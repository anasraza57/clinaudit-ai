"""
Tests for SQLAlchemy models.

These are pure unit tests that verify model instantiation and
relationships without needing a database connection.
"""

from datetime import date

from src.models.patient import Patient, ClinicalEntry
from src.models.audit import AuditJob, AuditResult
from src.models.guideline import Guideline


class TestPatientModel:
    def test_create_patient(self):
        patient = Patient(pat_id="abc-123-def")
        assert patient.pat_id == "abc-123-def"

    def test_patient_repr(self):
        patient = Patient(id=1, pat_id="abc-123")
        assert "abc-123" in repr(patient)

    def test_patient_tablename(self):
        assert Patient.__tablename__ == "patients"


class TestClinicalEntryModel:
    def test_create_entry(self):
        entry = ClinicalEntry(
            patient_id=1,
            index_date=date(2024, 9, 4),
            cons_date=date(2024, 9, 4),
            concept_id="18171007",
            term="Finger fracture",
            concept_display="Fracture of phalanx of finger",
        )
        assert entry.concept_id == "18171007"
        assert entry.term == "Finger fracture"
        assert entry.category is None

    def test_entry_tablename(self):
        assert ClinicalEntry.__tablename__ == "clinical_entries"


class TestAuditJobModel:
    def test_create_job(self):
        job = AuditJob(
            status="pending",
            total_patients=10,
            processed_patients=0,
            failed_patients=0,
        )
        assert job.status == "pending"
        assert job.total_patients == 10

    def test_job_repr(self):
        job = AuditJob(
            id=1, status="running",
            total_patients=5, processed_patients=3,
            failed_patients=0,
        )
        assert "running" in repr(job)
        assert "3/5" in repr(job)

    def test_job_tablename(self):
        assert AuditJob.__tablename__ == "audit_jobs"


class TestAuditResultModel:
    def test_create_result(self):
        result = AuditResult(
            patient_id=1,
            overall_score=0.85,
            diagnoses_found=3,
            guidelines_followed=2,
            guidelines_not_followed=1,
            status="completed",
        )
        assert result.overall_score == 0.85
        assert result.diagnoses_found == 3

    def test_result_repr(self):
        result = AuditResult(id=1, patient_id=5, overall_score=0.75)
        r = repr(result)
        assert "patient_id=5" in r
        assert "0.75" in r

    def test_result_tablename(self):
        assert AuditResult.__tablename__ == "audit_results"


class TestGuidelineModel:
    def test_create_guideline(self):
        guideline = Guideline(
            source_id="abc123hash",
            source="nice",
            title="Test Guideline Title",
            clean_text="Some guideline content here.",
        )
        assert guideline.source == "nice"
        assert guideline.url is None

    def test_guideline_repr(self):
        guideline = Guideline(
            id=1,
            title="A very long guideline title that should be truncated in repr",
        )
        assert "Guideline" in repr(guideline)

    def test_guideline_tablename(self):
        assert Guideline.__tablename__ == "guidelines"
