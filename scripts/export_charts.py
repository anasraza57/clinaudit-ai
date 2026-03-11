"""
Export audit charts as PNG images to a local folder.

Usage:
    DB_HOST=localhost python3 scripts/export_charts.py --output exports/charts
    DB_HOST=localhost python3 scripts/export_charts.py --output exports/charts --job-id 1 --dpi 300
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.models.database import close_db, get_session_factory
from src.services.export import export_charts_to_png


async def main():
    parser = argparse.ArgumentParser(
        description="Export audit charts as PNG images",
    )
    parser.add_argument(
        "--output", "-o",
        default="exports/charts",
        help="Output directory for PNG files (default: exports/charts)",
    )
    parser.add_argument(
        "--job-id", "-j",
        type=int,
        default=None,
        help="Scope to a specific batch job ID",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PNG output (default: 200)",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    factory = get_session_factory()

    print(f"Exporting charts to: {args.output}")
    if args.job_id:
        print(f"  Job ID: {args.job_id}")
    print(f"  DPI: {args.dpi}")

    async with factory() as session:
        saved = await export_charts_to_png(
            session,
            output_dir=args.output,
            job_id=args.job_id,
            dpi=args.dpi,
        )

    await close_db()

    if saved:
        print(f"\nSaved {len(saved)} chart(s):")
        for path in saved:
            print(f"  {path}")
    else:
        print("\nNo charts generated (no audit data found).")


if __name__ == "__main__":
    asyncio.run(main())
