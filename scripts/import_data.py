"""
Import patient records and guidelines from CSV into PostgreSQL.

Usage:
    DB_HOST=localhost python3 scripts/import_data.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.models.database import get_session_factory, close_db
from src.services.data_import import import_guidelines, import_patients


async def main():
    get_settings.cache_clear()
    factory = get_session_factory()

    print("=== Importing patient data ===")
    async with factory() as session:
        summary = await import_patients(session)
        await session.commit()
    print(f"  Patients: {summary['new_patients']} new, {summary['skipped_patients']} skipped")
    print(f"  Clinical entries: {summary['new_entries']}")

    print("\n=== Importing guidelines ===")
    async with factory() as session:
        summary = await import_guidelines(session)
        await session.commit()
    print(f"  Guidelines: {summary['new_guidelines']} new, {summary['skipped']} skipped")

    await close_db()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
