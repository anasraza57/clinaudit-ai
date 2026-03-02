"""
Audit API routes.

Endpoints for running the audit pipeline (single patient or batch)
and retrieving results.
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

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
    message: str


class JobStatusResponse(BaseModel):
    job_id: int
    status: str = Field(description="pending | running | completed | failed")
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

    # Ensure embedder is loaded
    if not pipeline._retriever._embedder.is_loaded:
        pipeline._retriever._embedder.load()

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

    Each patient is processed through the full 4-agent pipeline.
    Progress is committed every 10 patients.
    """
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
    else:
        query = select(Patient.pat_id)
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
        "message": f"Batch audit started. Poll GET /api/v1/audit/jobs/{job_id} for status.",
    }


async def _run_batch_background(job_id: int, pat_ids: list[str]) -> None:
    """Background task that runs the batch pipeline with its own DB session."""
    factory = get_session_factory()

    async with factory() as session:
        try:
            # Update job status — commit immediately so polling sees "running"
            result = await session.execute(
                select(AuditJob).where(AuditJob.id == job_id)
            )
            job = result.scalar_one()
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info("Batch job %d started (%d patients)", job_id, len(pat_ids))

            pipeline = _get_pipeline()

            # Ensure embedder is loaded
            if not pipeline._retriever._embedder.is_loaded:
                pipeline._retriever._embedder.load()

            # Load categories
            await pipeline.load_categories_from_db(session)

            # Process each patient
            for i, pat_id in enumerate(pat_ids, 1):
                try:
                    pipeline_result = await pipeline.run_single(
                        session, pat_id, job_id=job_id
                    )
                    if not pipeline_result.success:
                        job.failed_patients += 1
                except Exception as e:
                    logger.error("Batch job %d: patient %s failed: %s",
                                 job_id, pat_id, e)
                    job.failed_patients += 1

                job.processed_patients = i

                # Commit progress every patient so polling sees updates
                await session.commit()

                if i % 10 == 0:
                    logger.info(
                        "Batch job %d: %d/%d patients (%d failed)",
                        job_id, i, len(pat_ids), job.failed_patients,
                    )

            # Finalise
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()

            logger.info(
                "Batch job %d finished: %d/%d patients, %d failed",
                job_id, job.processed_patients, job.total_patients, job.failed_patients,
            )

        except Exception as e:
            logger.error("Batch job %d crashed: %s", job_id, e)
            try:
                # Re-fetch job in case session state is stale after error
                result = await session.execute(
                    select(AuditJob).where(AuditJob.id == job_id)
                )
                job = result.scalar_one()
                job.status = "failed"
                job.error_message = str(e)[:500]
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
            except Exception as inner_e:
                logger.error("Batch job %d: failed to save error state: %s",
                             job_id, inner_e)


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

    # Count total results
    total = await session.scalar(
        select(func.count())
        .select_from(AuditResult)
        .where(AuditResult.job_id == job_id)
    )
    total = total or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    # Fetch paginated results with patient info
    offset = (page - 1) * page_size
    result = await session.execute(
        select(AuditResult, Patient.pat_id)
        .join(Patient, AuditResult.patient_id == Patient.id)
        .where(AuditResult.job_id == job_id)
        .order_by(AuditResult.id)
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
