"""
Data import service.

Loads patient records and guidelines from CSV files into PostgreSQL.
Designed for idempotent re-runs: skips rows that already exist.
"""

import csv
import gzip
import io
import logging
import sys
from datetime import date
from pathlib import Path

# Some guideline texts are very large — increase CSV field size limit
csv.field_size_limit(sys.maxsize)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.models.patient import Patient, ClinicalEntry
from src.models.guideline import Guideline

logger = logging.getLogger(__name__)


async def import_patients(session: AsyncSession, csv_path: str | None = None) -> dict:
    """
    Import patient records from msk_valid_notes.csv.

    CSV columns: PatID, Index_date, Consdate, ConceptID, Term, Notes, ConceptDisplay

    Each unique PatID becomes a Patient row. Each CSV row becomes a
    ClinicalEntry linked to its Patient.

    Returns a summary dict with counts.
    """
    settings = get_settings()
    path = Path(csv_path or settings.patient_data_path)

    if not path.exists():
        raise FileNotFoundError(f"Patient data file not found: {path}")

    logger.info("Importing patients from %s", path)

    # Load existing patient IDs to skip duplicates
    result = await session.execute(select(Patient.pat_id))
    existing_pat_ids: set[str] = {row[0] for row in result.all()}
    logger.info("Found %d existing patients in database", len(existing_pat_ids))

    # First pass: collect unique patient IDs and their entries
    patients_data: dict[str, list[dict]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pat_id = row["PatID"].strip()
            if not pat_id:
                continue
            if pat_id not in patients_data:
                patients_data[pat_id] = []
            patients_data[pat_id].append(row)

    new_patients = 0
    new_entries = 0
    skipped_patients = 0

    for pat_id, rows in patients_data.items():
        if pat_id in existing_pat_ids:
            skipped_patients += 1
            continue

        patient = Patient(pat_id=pat_id)
        session.add(patient)
        # Flush to get the auto-generated id
        await session.flush()

        for row in rows:
            entry = ClinicalEntry(
                patient_id=patient.id,
                index_date=date.fromisoformat(row["Index_date"].strip()),
                cons_date=date.fromisoformat(row["Consdate"].strip()),
                concept_id=row["ConceptID"].strip(),
                term=row["Term"].strip(),
                concept_display=row["ConceptDisplay"].strip(),
                notes=row.get("Notes", "").strip() or None,
                category=None,  # Will be set by the Extractor agent later
            )
            session.add(entry)
            new_entries += 1

        new_patients += 1

        # Batch flush every 500 patients to manage memory
        if new_patients % 500 == 0:
            await session.flush()
            logger.info("Progress: %d patients imported...", new_patients)

    await session.flush()

    summary = {
        "total_in_csv": len(patients_data),
        "new_patients": new_patients,
        "new_entries": new_entries,
        "skipped_patients": skipped_patients,
    }
    logger.info(
        "Patient import complete: %d new patients, %d entries, %d skipped",
        new_patients, new_entries, skipped_patients,
    )
    return summary


async def import_guidelines(session: AsyncSession, csv_path: str | None = None) -> dict:
    """
    Import NICE guidelines from guidelines.csv.

    CSV columns: id, source, title, clean_text, raw_text, url, overview

    Each row becomes a Guideline record. Skips rows that already exist
    (matched by source_id).

    Returns a summary dict with counts.
    """
    settings = get_settings()
    path = Path(csv_path or settings.guidelines_csv_path)
    gz_path = Path(str(path) + ".gz")

    # Support both raw CSV and gzip-compressed versions
    if not path.exists() and gz_path.exists():
        logger.info("Found compressed guidelines at %s, will decompress on the fly", gz_path)
        use_gz = True
    elif path.exists():
        use_gz = False
    else:
        raise FileNotFoundError(
            f"Guidelines file not found: {path} (also checked {gz_path})"
        )

    logger.info("Importing guidelines from %s", gz_path if use_gz else path)

    # Load existing source IDs
    result = await session.execute(select(Guideline.source_id))
    existing_ids: set[str] = {row[0] for row in result.all()}
    logger.info("Found %d existing guidelines in database", len(existing_ids))

    new_guidelines = 0
    skipped = 0

    def _open_csv():
        if use_gz:
            return io.TextIOWrapper(gzip.open(gz_path, "rb"), encoding="utf-8", newline="")
        return open(path, newline="", encoding="utf-8")

    with _open_csv() as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row["id"].strip()
            if not source_id or source_id in existing_ids:
                skipped += 1
                continue

            guideline = Guideline(
                source_id=source_id,
                source=row.get("source", "nice").strip(),
                title=row["title"].strip(),
                clean_text=row["clean_text"].strip(),
                url=row.get("url", "").strip() or None,
                overview=row.get("overview", "").strip() or None,
            )
            session.add(guideline)
            new_guidelines += 1

            # Batch flush every 1000 rows
            if new_guidelines % 1000 == 0:
                await session.flush()
                logger.info("Progress: %d guidelines imported...", new_guidelines)

    await session.flush()

    summary = {
        "new_guidelines": new_guidelines,
        "skipped": skipped,
    }
    logger.info(
        "Guideline import complete: %d new, %d skipped",
        new_guidelines, skipped,
    )
    return summary
