"""
Data management API routes.

Endpoints for importing patient records and guidelines into the database.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_session
from src.services.data_import import import_guidelines, import_patients

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/import/patients")
async def import_patients_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """Import patient records from the configured CSV file."""
    summary = await import_patients(session)
    return {"status": "ok", "summary": summary}


@router.post("/import/guidelines")
async def import_guidelines_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """Import guidelines from the configured CSV file."""
    summary = await import_guidelines(session)
    return {"status": "ok", "summary": summary}
