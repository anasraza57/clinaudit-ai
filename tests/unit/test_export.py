"""
Tests for the export service (CSV and HTML report generation).

Uses an in-memory SQLite database (via aiosqlite) to test actual
query and rendering logic without needing PostgreSQL.
"""

import csv
import io
import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.models.audit import AuditJob, AuditResult
from src.models.base import Base
from src.models.patient import Patient
from src.services.export import generate_csv, generate_html_report


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def async_session():
    """Create an in-memory SQLite database with all tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False,
    )
    async with factory() as session:
        yield session

    await engine.dispose()


# ── Helpers ───────────────────────────────────────────────────────────


def _make_patient(session: AsyncSession, pat_id: str) -> Patient:
    p = Patient(pat_id=pat_id)
    session.add(p)
    return p


def _make_details_json(pat_id: str, scores: list[dict]) -> str:
    adherent = sum(1 for s in scores if s.get("score") == 1)
    non_adherent = sum(1 for s in scores if s.get("score") == -1)
    total_scored = adherent + non_adherent
    return json.dumps({
        "pat_id": pat_id,
        "total_diagnoses": len(scores),
        "adherent": adherent,
        "non_adherent": non_adherent,
        "errors": 0,
        "aggregate_score": adherent / total_scored if total_scored > 0 else 0,
        "scores": scores,
    })


async def _add_completed_result(
    session: AsyncSession,
    patient: Patient,
    overall_score: float,
    details_scores: list[dict],
    job_id: int | None = None,
) -> AuditResult:
    await session.flush()
    adherent = sum(1 for s in details_scores if s.get("score") == 1)
    non_adherent_count = sum(1 for s in details_scores if s.get("score") == -1)
    result = AuditResult(
        patient_id=patient.id,
        job_id=job_id,
        overall_score=overall_score,
        diagnoses_found=len(details_scores),
        guidelines_followed=adherent,
        guidelines_not_followed=non_adherent_count,
        details_json=_make_details_json(patient.pat_id, details_scores),
        status="completed",
    )
    session.add(result)
    await session.flush()
    return result


async def _make_job(session: AsyncSession) -> AuditJob:
    job = AuditJob(
        status="completed",
        total_patients=2,
        processed_patients=2,
        failed_patients=0,
    )
    session.add(job)
    await session.flush()
    return job


# ── Test: CSV Export ──────────────────────────────────────────────────


class TestCSVExport:

    @pytest.mark.asyncio
    async def test_empty_returns_header_only(self, async_session):
        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # header only
        assert rows[0][0] == "pat_id"

    @pytest.mark.asyncio
    async def test_single_patient_one_diagnosis(self, async_session):
        p = _make_patient(async_session, "PAT-001")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Back pain",
                "index_date": "2024-01-15",
                "score": 1,
                "explanation": "Physio referral made",
                "guidelines_followed": ["Physiotherapy referral"],
                "guidelines_not_followed": [],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        data = rows[1]
        assert data[0] == "PAT-001"      # pat_id
        assert data[1] == "1.0"           # overall_score
        assert data[2] == "Back pain"     # diagnosis
        assert data[4] == "1"             # score
        assert "Physio" in data[5]        # explanation

    @pytest.mark.asyncio
    async def test_multiple_diagnoses_per_patient(self, async_session):
        p = _make_patient(async_session, "PAT-002")
        await _add_completed_result(async_session, p, 0.5, [
            {
                "diagnosis": "Hip pain",
                "score": 1,
                "explanation": "Referred to physio",
                "guidelines_followed": ["Physio"],
                "guidelines_not_followed": [],
            },
            {
                "diagnosis": "Shoulder pain",
                "score": -1,
                "explanation": "No actions taken",
                "guidelines_followed": [],
                "guidelines_not_followed": ["Exercise therapy"],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 3  # header + 2 diagnosis rows
        assert rows[1][2] == "Hip pain"
        assert rows[2][2] == "Shoulder pain"

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session):
        p1 = _make_patient(async_session, "PAT-A")
        p2 = _make_patient(async_session, "PAT-B")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])  # no job_id

        csv_str = await generate_csv(async_session, job_id=job.id)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 (only PAT-A)
        assert rows[1][0] == "PAT-A"

    @pytest.mark.asyncio
    async def test_guidelines_joined_with_semicolons(self, async_session):
        p = _make_patient(async_session, "PAT-003")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Knee pain",
                "score": 1,
                "explanation": "Multiple guidelines followed",
                "guidelines_followed": ["Physio referral", "Exercise advice", "Weight management"],
                "guidelines_not_followed": ["Imaging"],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        followed = rows[1][6]  # guidelines_followed column
        assert "Physio referral" in followed
        assert "Exercise advice" in followed
        assert ";" in followed

        not_followed = rows[1][7]  # guidelines_not_followed column
        assert not_followed == "Imaging"


# ── Test: HTML Report ─────────────────────────────────────────────────


class TestHTMLReport:

    @pytest.mark.asyncio
    async def test_empty_report(self, async_session):
        html = await generate_html_report(async_session)

        assert "<!DOCTYPE html>" in html
        assert "GuidelineGuard Audit Report" in html
        assert "0" in html  # 0 patients

    @pytest.mark.asyncio
    async def test_report_contains_stats(self, async_session):
        for pat_id, score in [("P1", 0.5), ("P2", 1.0)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
                 "guidelines_followed": ["Physio"], "guidelines_not_followed": []},
            ])

        html = await generate_html_report(async_session)

        assert "2" in html            # total patients
        assert "Patients Audited" in html
        assert "Mean Adherence" in html
        assert "Median Adherence" in html

    @pytest.mark.asyncio
    async def test_report_contains_conditions(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Low back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "explanation": "No referral",
             "guidelines_followed": [], "guidelines_not_followed": ["Physio"]},
        ])

        html = await generate_html_report(async_session)

        assert "Low back pain" in html
        assert "Knee pain" in html
        assert "Adherence by Condition" in html

    @pytest.mark.asyncio
    async def test_report_contains_patient_details(self, async_session):
        p = _make_patient(async_session, "abc123-test-patient-uuid")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Finger pain",
                "score": 1,
                "explanation": "Physiotherapy referral was appropriate",
                "guidelines_followed": ["Physiotherapy referral"],
                "guidelines_not_followed": [],
            },
        ])

        html = await generate_html_report(async_session)

        assert "abc123-test-" in html  # truncated pat_id
        assert "Finger pain" in html
        assert "Physiotherapy referral was appropriate" in html
        assert "+1 Adherent" in html

    @pytest.mark.asyncio
    async def test_report_has_score_badges(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])

        html = await generate_html_report(async_session)

        assert "+1 Adherent" in html
        assert "-1 Non-adherent" in html

    @pytest.mark.asyncio
    async def test_report_is_self_contained(self, async_session):
        """HTML report has inline CSS, no external dependencies."""
        html = await generate_html_report(async_session)

        assert "<style>" in html
        # No external stylesheet references
        assert 'rel="stylesheet"' not in html
        assert "<script" not in html

    @pytest.mark.asyncio
    async def test_job_id_scoping(self, async_session):
        p1 = _make_patient(async_session, "IN-JOB")
        p2 = _make_patient(async_session, "NOT-IN-JOB")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Hip pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])

        html = await generate_html_report(async_session, job_id=job.id)

        assert "IN-JOB" in html
        assert "NOT-IN-JOB" not in html
        assert "1" in html  # 1 patient
