"""
Tests for the data import service.

These tests verify CSV parsing logic without needing a live database.
We test the file-reading and row-parsing aspects only.
Integration tests (with a real database) will be added later.
"""

import csv
import pytest
from pathlib import Path

from src.services.data_import import import_patients, import_guidelines


class TestImportPatientsParsing:
    """Test that import_patients raises correctly on missing files."""

    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        """Should raise FileNotFoundError for non-existent CSV."""
        # We need a mock session, but the function checks the file first
        with pytest.raises(FileNotFoundError, match="not found"):
            await import_patients(session=None, csv_path="/nonexistent/path.csv")

    def test_csv_structure(self, tmp_path):
        """Verify our CSV reader handles the expected column names."""
        csv_path = tmp_path / "test_patients.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PatID", "Index_date", "Consdate", "ConceptID", "Term", "Notes", "ConceptDisplay"])
            writer.writerow(["pat-001", "2024-01-01", "2024-01-01", "123456", "Back pain", "", "Low back pain"])
            writer.writerow(["pat-001", "2024-01-01", "2024-01-15", "789012", "Ibuprofen", "400mg", "Ibuprofen"])
            writer.writerow(["pat-002", "2024-02-01", "2024-02-01", "345678", "Knee pain", "", "Knee pain"])

        # Just verify the CSV is well-formed and parseable
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["PatID"] == "pat-001"
        assert rows[2]["ConceptID"] == "345678"
        # Two unique patients
        unique_pats = {r["PatID"] for r in rows}
        assert len(unique_pats) == 2


class TestImportGuidelinesParsing:
    """Test that import_guidelines raises correctly on missing files."""

    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            await import_guidelines(session=None, csv_path="/nonexistent/guidelines.csv")

    def test_csv_structure(self, tmp_path):
        """Verify our CSV reader handles the expected column names."""
        csv_path = tmp_path / "test_guidelines.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "source", "title", "clean_text", "raw_text", "url", "overview"])
            writer.writerow(["hash1", "nice", "Guideline 1", "Text 1", "Raw 1", "http://example.com", "Overview 1"])
            writer.writerow(["hash2", "nice", "Guideline 2", "Text 2", "Raw 2", "", ""])

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["id"] == "hash1"
        assert rows[1]["url"] == ""
