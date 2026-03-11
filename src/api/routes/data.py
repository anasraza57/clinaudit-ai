"""
Data management API routes.

Endpoints for importing patient records and guidelines into the database,
and viewing database statistics.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_session
from src.models.patient import ClinicalEntry, Patient
from src.models.guideline import Guideline
from src.services.data_import import import_guidelines, import_patients

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])


# ── Response schemas ─────────────────────────────────────────────────


class ImportSummary(BaseModel):
    status: str
    summary: dict


class DataStatsResponse(BaseModel):
    patients: int
    clinical_entries: int
    guidelines: int

    model_config = {"json_schema_extra": {"examples": [
        {"patients": 4327, "clinical_entries": 21530, "guidelines": 1656}
    ]}}


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/stats", response_model=DataStatsResponse, summary="Database statistics")
async def data_stats(
    session: AsyncSession = Depends(get_session),
):
    """
    Returns row counts for the main data tables: patients,
    clinical entries, and guidelines.

    Use this to verify data has been imported correctly.
    """
    try:
        patients = await session.scalar(select(func.count()).select_from(Patient))
        entries = await session.scalar(select(func.count()).select_from(ClinicalEntry))
        guidelines = await session.scalar(select(func.count()).select_from(Guideline))
    except ProgrammingError:
        # Tables don't exist yet (migrations not run)
        await session.rollback()
        return {"patients": 0, "clinical_entries": 0, "guidelines": 0}
    return {
        "patients": patients or 0,
        "clinical_entries": entries or 0,
        "guidelines": guidelines or 0,
    }


@router.post("/import/patients", response_model=ImportSummary, summary="Import patient data")
async def import_patients_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """
    Import patient records and clinical entries from the configured CSV file
    into the database.

    This is idempotent — re-running will skip existing patients (matched
    by `pat_id`) and only insert new ones.
    """
    summary = await import_patients(session)
    return {"status": "ok", "summary": summary}


@router.post("/import/guidelines", response_model=ImportSummary, summary="Import guidelines")
async def import_guidelines_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """
    Import NICE clinical guidelines from the configured CSV file into
    the database.

    This is idempotent — re-running will skip existing guidelines (matched
    by `guideline_id`) and only insert new ones.
    """
    summary = await import_guidelines(session)
    return {"status": "ok", "summary": summary}
