"""
Audit API routes.

Endpoints for running the audit pipeline (single patient or batch)
and retrieving results.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.audit import AuditJob, AuditResult
from src.models.database import get_session, get_session_factory
from src.models.patient import Patient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


# ── Request / response schemas ───────────────────────────────────────


class AuditSingleRequest(BaseModel):
    """Request to audit a single patient."""
    pat_id: str


class BatchAuditRequest(BaseModel):
    """Request to audit multiple patients."""
    pat_ids: list[str] | None = None  # None = audit all patients


class AuditResultResponse(BaseModel):
    """Audit result for a single patient."""
    pat_id: str
    overall_score: float | None
    diagnoses_found: int
    guidelines_followed: int
    guidelines_not_followed: int
    status: str
    error_message: str | None = None
    details: dict | None = None


class JobStatusResponse(BaseModel):
    """Status of a batch audit job."""
    job_id: int
    status: str
    total_patients: int
    processed_patients: int
    failed_patients: int
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


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


@router.post("/patient/{pat_id}")
async def audit_single_patient(
    pat_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Run the full audit pipeline for a single patient.

    Returns the scoring result immediately (synchronous).
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


@router.post("/batch")
async def start_batch_audit(
    request: BatchAuditRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """
    Start a batch audit job for multiple patients.

    The job runs in the background. Returns the job ID immediately
    for polling via GET /audit/jobs/{job_id}.
    """
    # Resolve patient IDs
    if request.pat_ids is not None:
        pat_ids = request.pat_ids
        # Verify all patients exist
        result = await session.execute(
            select(Patient.pat_id).where(Patient.pat_id.in_(pat_ids))
        )
        found = {row[0] for row in result.all()}
        missing = set(pat_ids) - found
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Patients not found: {sorted(missing)[:10]}",
            )
    else:
        result = await session.execute(select(Patient.pat_id))
        pat_ids = [row[0] for row in result.all()]

    if not pat_ids:
        raise HTTPException(status_code=400, detail="No patients to audit")

    # Create job record
    job = AuditJob(
        status="pending",
        total_patients=len(pat_ids),
        processed_patients=0,
        failed_patients=0,
    )
    session.add(job)
    await session.flush()
    job_id = job.id

    # Schedule background processing
    background_tasks.add_task(_run_batch_background, job_id, pat_ids)

    return {
        "status": "accepted",
        "job_id": job_id,
        "total_patients": len(pat_ids),
        "message": f"Batch audit started. Poll GET /api/v1/audit/jobs/{job_id} for status.",
    }


async def _run_batch_background(job_id: int, pat_ids: list[str]) -> None:
    """Background task that runs the batch pipeline with its own DB session."""
    factory = get_session_factory()

    async with factory() as session:
        try:
            # Update job status
            result = await session.execute(
                select(AuditJob).where(AuditJob.id == job_id)
            )
            job = result.scalar_one()
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            await session.flush()

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

                if i % 10 == 0:
                    await session.commit()
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
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
            except Exception:
                pass


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get the status of a batch audit job."""
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


@router.get("/results/{pat_id}")
async def get_patient_results(
    pat_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get all audit results for a patient."""
    # Find patient
    result = await session.execute(
        select(Patient).where(Patient.pat_id == pat_id)
    )
    patient = result.scalar_one_or_none()

    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {pat_id} not found")

    # Get results
    result = await session.execute(
        select(AuditResult)
        .where(AuditResult.patient_id == patient.id)
        .order_by(AuditResult.id.desc())
    )
    audit_results = result.scalars().all()

    if not audit_results:
        return {"pat_id": pat_id, "results": [], "message": "No audit results found"}

    responses = []
    for ar in audit_results:
        details = None
        if ar.details_json:
            try:
                details = json.loads(ar.details_json)
            except json.JSONDecodeError:
                details = None

        responses.append(
            AuditResultResponse(
                pat_id=pat_id,
                overall_score=ar.overall_score,
                diagnoses_found=ar.diagnoses_found,
                guidelines_followed=ar.guidelines_followed,
                guidelines_not_followed=ar.guidelines_not_followed,
                status=ar.status,
                error_message=ar.error_message,
                details=details,
            )
        )

    return {"pat_id": pat_id, "results": [r.model_dump() for r in responses]}
