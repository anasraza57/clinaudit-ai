"""
Audit API routes.

Endpoints for running the audit pipeline (single patient or batch)
and retrieving results.
"""

import asyncio
import gc
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.models.audit import AuditJob, AuditResult
from src.models.database import get_session, get_session_factory
from src.models.patient import Patient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["Audit"])


# ── Response schemas ─────────────────────────────────────────────────


class DiagnosisScoreSchema(BaseModel):
    diagnosis: str
    score: int = Field(description="+1 (adherent) or -1 (non-adherent)")
    explanation: str
    guidelines_followed: list[str]
    guidelines_not_followed: list[str]


class ScoringResultSchema(BaseModel):
    overall_score: float = Field(description="Proportion of adherent diagnoses (0.0–1.0)")
    total_diagnoses: int
    adherent_count: int
    non_adherent_count: int
    error_count: int
    scores: list[DiagnosisScoreSchema]


class AuditSingleSuccessResponse(BaseModel):
    status: str = "completed"
    pat_id: str
    result: ScoringResultSchema


class AuditSingleFailedResponse(BaseModel):
    status: str = "failed"
    pat_id: str
    error: str | None
    stage_reached: str | None


class BatchAcceptedResponse(BaseModel):
    status: str = "accepted"
    job_id: int
    total_patients: int
    provider: str = Field(description="AI provider used for this job")
    message: str


class JobStatusResponse(BaseModel):
    job_id: int
    status: str = Field(description="pending | running | completed | failed")
    provider: str | None = Field(None, description="AI provider used for this job")
    total_patients: int
    processed_patients: int
    failed_patients: int
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


class AuditResultItem(BaseModel):
    pat_id: str
    overall_score: float | None
    diagnoses_found: int
    guidelines_followed: int
    guidelines_not_followed: int
    status: str
    error_message: str | None = None
    details: dict | None = None


class JobResultsResponse(BaseModel):
    job_id: int
    total: int
    page: int
    page_size: int
    total_pages: int
    results: list[AuditResultItem]


# ── Helper: build pipeline ───────────────────────────────────────────


def _get_pipeline():
    """
    Build and return an AuditPipeline instance.

    Assembles the pipeline from singleton services and the configured
    AI provider. Called per-request to ensure fresh state.
    """
    from src.ai.factory import get_ai_provider
    from src.services.embedder import get_embedder
    from src.services.pipeline import AuditPipeline
    from src.services.vector_store import get_vector_store

    ai_provider = get_ai_provider()
    embedder = get_embedder()
    vector_store = get_vector_store()

    return AuditPipeline(
        ai_provider=ai_provider,
        embedder=embedder,
        vector_store=vector_store,
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.post(
    "/patient/{pat_id}",
    summary="Audit a single patient",
    responses={
        200: {"description": "Audit completed or failed with details"},
        404: {"description": "Patient not found"},
    },
)
async def audit_single_patient(
    pat_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Run the full 4-agent audit pipeline for a single patient.

    The pipeline runs synchronously and returns the result immediately:

    1. **Extractor** — groups clinical entries by episode, categorises each
    2. **Query** — generates search queries for each diagnosis
    3. **Retriever** — finds relevant NICE guidelines via FAISS + PubMedBERT
    4. **Scorer** — LLM evaluates adherence per diagnosis

    Returns an overall adherence score (0.0–1.0) and per-diagnosis
    breakdown with explanations.
    """
    # Verify patient exists
    result = await session.execute(
        select(Patient).where(Patient.pat_id == pat_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {pat_id} not found")

    pipeline = _get_pipeline()

    # Ensure embedder and vector store are loaded
    if not pipeline._retriever._embedder.is_loaded:
        pipeline._retriever._embedder.load()
    if not pipeline._retriever._vector_store.is_loaded:
        pipeline._retriever._vector_store.load()

    # Load SNOMED categories if needed
    if not pipeline.categories_loaded:
        await pipeline.load_categories_from_db(session)

    # Run pipeline
    pipeline_result = await pipeline.run_single(session, pat_id)

    if pipeline_result.success:
        return {
            "status": "completed",
            "pat_id": pat_id,
            "result": pipeline_result.scoring.summary(),
        }
    else:
        return {
            "status": "failed",
            "pat_id": pat_id,
            "error": pipeline_result.error,
            "stage_reached": pipeline_result.stage_reached,
        }


@router.post(
    "/batch",
    response_model=BatchAcceptedResponse,
    summary="Start a batch audit",
    responses={
        404: {"description": "One or more patients not found"},
        400: {"description": "No patients to audit"},
    },
)
async def start_batch_audit(
    background_tasks: BackgroundTasks,
    limit: int | None = Query(None, ge=1, description="Maximum number of patients to audit (default: all)"),
    pat_ids: list[str] | None = Query(None, description="Specific patient IDs to audit (default: all)"),
    skip_audited: bool = Query(False, description="Skip patients already audited by the current AI provider"),
    session: AsyncSession = Depends(get_session),
):
    """
    Start a batch audit job for multiple patients.

    The job runs in the background. Returns a job ID immediately for
    polling progress via `GET /audit/jobs/{job_id}`.

    **Options:**
    - Omit both parameters to audit all patients in the database
    - Use `limit` to audit a random subset (e.g. `?limit=50`)
    - Use `pat_ids` to audit specific patients
    - Use `skip_audited=true` to only audit patients without a completed result
      **for the current AI provider** (so switching from OpenAI to Ollama won't
      skip patients — each provider's audits are tracked independently)

    Each patient is processed through the full 4-agent pipeline.
    Progress is committed every 10 patients.
    """
    # Subquery: patient IDs that already have a completed audit result
    # for the CURRENT provider — switching providers won't skip patients
    current_provider = get_settings().ai_provider
    already_audited_subq = (
        select(Patient.id)
        .join(AuditResult, AuditResult.patient_id == Patient.id)
        .join(AuditJob, AuditResult.job_id == AuditJob.id)
        .where(AuditResult.status == "completed")
        .where(AuditJob.provider == current_provider)
    )

    # Resolve patient IDs
    if pat_ids is not None:
        ids = pat_ids
        # Verify all patients exist
        result = await session.execute(
            select(Patient.pat_id).where(Patient.pat_id.in_(ids))
        )
        found = {row[0] for row in result.all()}
        missing = set(ids) - found
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Patients not found: {sorted(missing)[:10]}",
            )
        # Filter out already-audited if requested
        if skip_audited:
            result = await session.execute(
                select(Patient.pat_id)
                .where(Patient.pat_id.in_(ids))
                .where(~Patient.id.in_(already_audited_subq))
            )
            ids = [row[0] for row in result.all()]
    else:
        query = select(Patient.pat_id)
        if skip_audited:
            query = query.where(~Patient.id.in_(already_audited_subq))
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        ids = [row[0] for row in result.all()]

    if not ids:
        raise HTTPException(status_code=400, detail="No patients to audit")

    # Create job record
    job = AuditJob(
        status="pending",
        total_patients=len(ids),
        processed_patients=0,
        failed_patients=0,
        provider=current_provider,
    )
    session.add(job)
    await session.flush()
    job_id = job.id

    # Schedule background processing
    background_tasks.add_task(_run_batch_background, job_id, ids)

    return {
        "status": "accepted",
        "job_id": job_id,
        "total_patients": len(ids),
        "provider": current_provider,
        "message": f"Batch audit started. Poll GET /api/v1/audit/jobs/{job_id} for status.",
    }


async def _run_batch_background(job_id: int, pat_ids: list[str]) -> None:
    """
    Background task that processes patients with per-patient session isolation.

    Each patient gets its own short-lived DB session so the SQLAlchemy
    identity map is cleared between patients, preventing memory growth.
    Per-patient timeouts ensure a single slow LLM call can never stall
    the whole batch.
    """
    factory = get_session_factory()
    settings = get_settings()
    patient_timeout = settings.pipeline_patient_timeout

    # ── Mark job as running ───────────────────────────────────────
    async with factory() as session:
        result = await session.execute(
            select(AuditJob).where(AuditJob.id == job_id)
        )
        job = result.scalar_one()
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await session.commit()

    logger.info("Batch job %d started (%d patients)", job_id, len(pat_ids))

    pipeline = _get_pipeline()

    if not pipeline._retriever._embedder.is_loaded:
        pipeline._retriever._embedder.load()
    if not pipeline._retriever._vector_store.is_loaded:
        pipeline._retriever._vector_store.load()

    # Pre-load categories in a short-lived session
    async with factory() as session:
        await pipeline.load_categories_from_db(session)

    # ── Process patients (one session per patient) ────────────────
    failed_count = 0
    processed = 0

    for i, pat_id in enumerate(pat_ids, 1):
        try:
            # Fresh session per patient — identity map is discarded on exit,
            # keeping memory usage constant regardless of batch size.
            async with factory() as session:
                pipeline_result = await asyncio.wait_for(
                    pipeline.run_single(session, pat_id, job_id=job_id),
                    timeout=patient_timeout,
                )
                if not pipeline_result.success:
                    failed_count += 1

                # Update progress atomically with the patient result
                job_row = await session.execute(
                    select(AuditJob).where(AuditJob.id == job_id)
                )
                job_obj = job_row.scalar_one()
                job_obj.processed_patients = i
                job_obj.failed_patients = failed_count
                await session.commit()

        except asyncio.TimeoutError:
            failed_count += 1
            logger.error(
                "Batch job %d: patient %s timed out after %ds",
                job_id, pat_id, int(patient_timeout),
            )
            await _save_patient_error_and_progress(
                factory, pat_id, job_id,
                f"Pipeline timed out after {int(patient_timeout)}s",
                i, failed_count,
            )

        except Exception as e:
            failed_count += 1
            logger.error(
                "Batch job %d: patient %s failed: %s", job_id, pat_id, e,
            )
            await _save_patient_error_and_progress(
                factory, pat_id, job_id,
                str(e)[:500], i, failed_count,
            )

        processed = i

        if i % 10 == 0:
            logger.info(
                "Batch job %d: %d/%d patients (%d failed)",
                job_id, i, len(pat_ids), failed_count,
            )
            gc.collect()

    # ── Finalise job ──────────────────────────────────────────────
    try:
        async with factory() as session:
            result = await session.execute(
                select(AuditJob).where(AuditJob.id == job_id)
            )
            job = result.scalar_one()
            job.status = "completed"
            job.processed_patients = processed
            job.failed_patients = failed_count
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
    except Exception as e:
        logger.error("Batch job %d: failed to finalize: %s", job_id, e)

    logger.info(
        "Batch job %d finished: %d/%d patients, %d failed",
        job_id, processed, len(pat_ids), failed_count,
    )
    gc.collect()


async def _save_patient_error_and_progress(
    factory,
    pat_id: str,
    job_id: int,
    error_msg: str,
    processed: int,
    failed_count: int,
) -> None:
    """Store a failed AuditResult and update job progress in a clean session."""
    try:
        async with factory() as session:
            # Find patient DB id to store the error result
            result = await session.execute(
                select(Patient.id).where(Patient.pat_id == pat_id)
            )
            row = result.first()
            if row:
                audit_result = AuditResult(
                    patient_id=row[0],
                    job_id=job_id,
                    overall_score=None,
                    diagnoses_found=0,
                    guidelines_followed=0,
                    guidelines_not_followed=0,
                    details_json=None,
                    status="failed",
                    error_message=error_msg,
                )
                session.add(audit_result)

            # Update job progress
            job_result = await session.execute(
                select(AuditJob).where(AuditJob.id == job_id)
            )
            job = job_result.scalar_one()
            job.processed_patients = processed
            job.failed_patients = failed_count
            await session.commit()
    except Exception as e:
        logger.warning(
            "Batch job %d: failed to save error for patient %s: %s",
            job_id, pat_id, e,
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Check batch job status",
    responses={404: {"description": "Job not found"}},
)
async def get_job_status(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Get the current status and progress of a batch audit job.

    Poll this endpoint to track progress. The job transitions through
    states: `pending` → `running` → `completed` (or `failed`).
    """
    result = await session.execute(
        select(AuditJob).where(AuditJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        provider=job.provider,
        total_patients=job.total_patients,
        processed_patients=job.processed_patients,
        failed_patients=job.failed_patients,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
    )


@router.get(
    "/jobs/{job_id}/results",
    response_model=JobResultsResponse,
    summary="Get batch job results",
    responses={404: {"description": "Job not found"}},
)
async def get_job_results(
    job_id: int,
    status: str | None = Query(None, description="Filter by result status (completed | failed)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    session: AsyncSession = Depends(get_session),
):
    """
    Get paginated audit results for a specific batch job.

    Each result includes the patient's overall adherence score,
    diagnosis counts, and the full scoring breakdown in `details`.
    """
    # Verify job exists
    result = await session.execute(
        select(AuditJob).where(AuditJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Count total results (with optional status filter)
    count_q = (
        select(func.count())
        .select_from(AuditResult)
        .where(AuditResult.job_id == job_id)
    )
    if status is not None:
        count_q = count_q.where(AuditResult.status == status)
    total = await session.scalar(count_q)
    total = total or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    # Fetch paginated results with patient info
    offset = (page - 1) * page_size
    results_q = (
        select(AuditResult, Patient.pat_id)
        .join(Patient, AuditResult.patient_id == Patient.id)
        .where(AuditResult.job_id == job_id)
    )
    if status is not None:
        results_q = results_q.where(AuditResult.status == status)
    result = await session.execute(
        results_q.order_by(AuditResult.id)
        .offset(offset)
        .limit(page_size)
    )
    rows = result.all()

    items = []
    for ar, pat_id in rows:
        details = None
        if ar.details_json:
            try:
                details = json.loads(ar.details_json)
            except json.JSONDecodeError:
                details = None

        items.append(AuditResultItem(
            pat_id=pat_id,
            overall_score=ar.overall_score,
            diagnoses_found=ar.diagnoses_found,
            guidelines_followed=ar.guidelines_followed,
            guidelines_not_followed=ar.guidelines_not_followed,
            status=ar.status,
            error_message=ar.error_message,
            details=details,
        ))

    return JobResultsResponse(
        job_id=job_id,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        results=items,
    )


class PatientResultsResponse(BaseModel):
    pat_id: str
    total_results: int
    results: list[AuditResultItem]


@router.get(
    "/results/{pat_id}",
    response_model=PatientResultsResponse,
    summary="Get all results for a patient",
    responses={404: {"description": "Patient not found"}},
)
async def get_patient_results(
    pat_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get all audit results for a specific patient across all jobs.

    A patient may have been audited multiple times (different batch runs).
    Returns all results ordered by most recent first, each with the full
    scoring breakdown.
    """
    # Find patient
    result = await session.execute(
        select(Patient).where(Patient.pat_id == pat_id)
    )
    patient = result.scalar_one_or_none()

    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {pat_id} not found")

    # Get all results for this patient
    result = await session.execute(
        select(AuditResult)
        .where(AuditResult.patient_id == patient.id)
        .order_by(AuditResult.id.desc())
    )
    audit_results = result.scalars().all()

    items = []
    for ar in audit_results:
        details = None
        if ar.details_json:
            try:
                details = json.loads(ar.details_json)
            except json.JSONDecodeError:
                details = None

        items.append(AuditResultItem(
            pat_id=pat_id,
            overall_score=ar.overall_score,
            diagnoses_found=ar.diagnoses_found,
            guidelines_followed=ar.guidelines_followed,
            guidelines_not_followed=ar.guidelines_not_followed,
            status=ar.status,
            error_message=ar.error_message,
            details=details,
        ))

    return PatientResultsResponse(
        pat_id=pat_id,
        total_results=len(items),
        results=items,
    )
