"""
Tests for the reporting service.

Uses an in-memory SQLite database (via aiosqlite) to test actual
SQL queries and JSON parsing logic without needing PostgreSQL.
"""

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
from src.services.reporting import (
    _load_completed_results,
    compute_system_metrics,
    get_condition_breakdown,
    get_dashboard_stats,
    get_non_adherent_cases,
    get_score_distribution,
)


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
    """Create and add a Patient to the session."""
    p = Patient(pat_id=pat_id)
    session.add(p)
    return p


def _make_details_json(pat_id: str, scores: list[dict]) -> str:
    """Build a details_json string matching ScoringResult.summary() format."""
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
    """Add a completed AuditResult to the session."""
    await session.flush()  # ensure patient has an id

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


async def _add_failed_result(
    session: AsyncSession,
    patient: Patient,
    job_id: int | None = None,
    error_message: str = "Pipeline failed",
) -> AuditResult:
    """Add a failed AuditResult to the session."""
    await session.flush()

    result = AuditResult(
        patient_id=patient.id,
        job_id=job_id,
        overall_score=None,
        diagnoses_found=0,
        guidelines_followed=0,
        guidelines_not_followed=0,
        details_json=None,
        status="failed",
        error_message=error_message,
    )
    session.add(result)
    await session.flush()
    return result


async def _make_job(session: AsyncSession) -> AuditJob:
    """Create and flush an AuditJob, returning it with its id set."""
    job = AuditJob(
        status="completed",
        total_patients=1,
        processed_patients=1,
        failed_patients=0,
    )
    session.add(job)
    await session.flush()
    return job


# ── Test: _load_completed_results ─────────────────────────────────────


class TestLoadCompletedResults:

    @pytest.mark.asyncio
    async def test_returns_completed_only(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])
        await _add_failed_result(async_session, p)

        results = await _load_completed_results(async_session)
        assert len(results) == 1
        assert results[0].status == "completed"

    @pytest.mark.asyncio
    async def test_filters_by_job_id(self, async_session):
        p = _make_patient(async_session, "P1")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ], job_id=job.id)
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Missing"},
        ])

        results = await _load_completed_results(async_session, job_id=job.id)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_results(self, async_session):
        results = await _load_completed_results(async_session)
        assert results == []

    @pytest.mark.asyncio
    async def test_include_details_loads_patient(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])

        results = await _load_completed_results(
            async_session, include_details=True,
        )
        assert len(results) == 1
        assert results[0].patient.pat_id == "P1"


# ── Test: get_dashboard_stats ─────────────────────────────────────────


class TestDashboardStats:

    @pytest.mark.asyncio
    async def test_empty(self, async_session):
        stats = await get_dashboard_stats(async_session)
        assert stats["total_audited"] == 0
        assert stats["total_failed"] == 0
        assert stats["failure_rate"] == 0.0
        assert stats["score_stats"]["mean"] is None
        assert stats["score_stats"]["median"] is None
        assert stats["score_stats"]["min"] is None
        assert stats["score_stats"]["max"] is None

    @pytest.mark.asyncio
    async def test_single_completed(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.75, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])

        stats = await get_dashboard_stats(async_session)
        assert stats["total_audited"] == 1
        assert stats["total_failed"] == 0
        assert stats["failure_rate"] == 0.0
        assert stats["score_stats"]["mean"] == 0.75
        assert stats["score_stats"]["median"] == 0.75
        assert stats["score_stats"]["min"] == 0.75
        assert stats["score_stats"]["max"] == 0.75

    @pytest.mark.asyncio
    async def test_multiple_results(self, async_session):
        for pat_id, score in [("P1", 0.5), ("P2", 0.8), ("P3", 1.0)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Condition", "score": 1, "explanation": "OK"},
            ])

        stats = await get_dashboard_stats(async_session)
        assert stats["total_audited"] == 3
        assert stats["score_stats"]["min"] == 0.5
        assert stats["score_stats"]["max"] == 1.0
        assert stats["score_stats"]["median"] == 0.8

    @pytest.mark.asyncio
    async def test_with_failures(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")
        await _add_completed_result(async_session, p1, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])
        await _add_failed_result(async_session, p2)

        stats = await get_dashboard_stats(async_session)
        assert stats["total_audited"] == 1
        assert stats["total_failed"] == 1
        assert stats["failure_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.9, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Missing"},
        ])

        stats = await get_dashboard_stats(async_session, job_id=job.id)
        assert stats["total_audited"] == 1
        assert stats["score_stats"]["mean"] == 0.9

    @pytest.mark.asyncio
    async def test_median_even_count(self, async_session):
        for pat_id, score in [("P1", 0.2), ("P2", 0.4), ("P3", 0.6), ("P4", 0.8)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Condition", "score": 1, "explanation": "OK"},
            ])

        stats = await get_dashboard_stats(async_session)
        assert stats["score_stats"]["median"] == 0.5  # (0.4 + 0.6) / 2


# ── Test: get_condition_breakdown ─────────────────────────────────────


class TestConditionBreakdown:

    @pytest.mark.asyncio
    async def test_empty(self, async_session):
        breakdown = await get_condition_breakdown(async_session)
        assert breakdown == []

    @pytest.mark.asyncio
    async def test_basic_grouping_legacy(self, async_session):
        """Legacy binary format (no judgement key) still works."""
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")

        await _add_completed_result(async_session, p1, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Missing"},
        ])
        await _add_completed_result(async_session, p2, 0.0, [
            {"diagnosis": "Back pain", "score": -1, "explanation": "Missing"},
        ])

        breakdown = await get_condition_breakdown(async_session)
        assert len(breakdown) == 2

        back = next(b for b in breakdown if b["diagnosis"] == "Back pain")
        assert back["total_cases"] == 2
        assert back["compliant"] == 1
        assert back["non_compliant"] == 1
        assert back["adherent"] == 1
        assert back["non_adherent"] == 1
        assert back["adherence_rate"] == 0.5

        knee = next(b for b in breakdown if b["diagnosis"] == "Knee pain")
        assert knee["total_cases"] == 1
        assert knee["adherence_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_basic_grouping_new_format(self, async_session):
        """New 5-level format with judgement key."""
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")

        await _add_completed_result(async_session, p1, 0.75, [
            {"diagnosis": "Back pain", "score": 2, "judgement": "COMPLIANT",
             "confidence": 0.85, "explanation": "OK"},
            {"diagnosis": "Knee pain", "score": -1, "judgement": "NON-COMPLIANT",
             "confidence": 0.9, "explanation": "Missing"},
        ])
        await _add_completed_result(async_session, p2, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "judgement": "PARTIALLY COMPLIANT",
             "confidence": 0.7, "explanation": "Partial"},
        ])

        breakdown = await get_condition_breakdown(async_session)
        assert len(breakdown) == 2

        back = next(b for b in breakdown if b["diagnosis"] == "Back pain")
        assert back["total_cases"] == 2
        assert back["compliant"] == 1
        assert back["partial"] == 1
        assert back["adherent"] == 2
        assert back["non_adherent"] == 0
        assert back["adherence_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_min_count_filter(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")

        await _add_completed_result(async_session, p1, 1.0, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
            {"diagnosis": "Rare condition", "score": -1, "explanation": "Missing"},
        ])
        await _add_completed_result(async_session, p2, 1.0, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])

        breakdown = await get_condition_breakdown(async_session, min_count=2)
        assert len(breakdown) == 1
        assert breakdown[0]["diagnosis"] == "Back pain"

    @pytest.mark.asyncio
    async def test_sort_by_adherence_rate(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")

        await _add_completed_result(async_session, p1, 0.5, [
            {"diagnosis": "Good condition", "score": 1, "explanation": "OK"},
            {"diagnosis": "Bad condition", "score": -1, "explanation": "Missing"},
        ])
        await _add_completed_result(async_session, p2, 0.5, [
            {"diagnosis": "Good condition", "score": 1, "explanation": "OK"},
            {"diagnosis": "Bad condition", "score": -1, "explanation": "Missing"},
        ])

        breakdown = await get_condition_breakdown(
            async_session, sort_by="adherence_rate",
        )
        assert breakdown[0]["diagnosis"] == "Bad condition"   # 0%
        assert breakdown[1]["diagnosis"] == "Good condition"  # 100%

    @pytest.mark.asyncio
    async def test_sort_by_count_default(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")
        p3 = _make_patient(async_session, "P3")

        await _add_completed_result(async_session, p1, 1.0, [
            {"diagnosis": "Common", "score": 1, "explanation": "OK"},
            {"diagnosis": "Rare", "score": 1, "explanation": "OK"},
        ])
        await _add_completed_result(async_session, p2, 1.0, [
            {"diagnosis": "Common", "score": 1, "explanation": "OK"},
        ])
        await _add_completed_result(async_session, p3, 1.0, [
            {"diagnosis": "Common", "score": 1, "explanation": "OK"},
        ])

        breakdown = await get_condition_breakdown(async_session)
        assert breakdown[0]["diagnosis"] == "Common"
        assert breakdown[0]["total_cases"] == 3


# ── Test: get_non_adherent_cases ──────────────────────────────────────


class TestNonAdherentCases:

    @pytest.mark.asyncio
    async def test_empty(self, async_session):
        result = await get_non_adherent_cases(async_session)
        assert result["total"] == 0
        assert result["cases"] == []
        assert result["total_pages"] == 0

    @pytest.mark.asyncio
    async def test_basic(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1,
             "explanation": "No physio referral",
             "guidelines_not_followed": ["Physiotherapy referral"]},
        ])

        result = await get_non_adherent_cases(async_session)
        assert result["total"] == 1
        assert len(result["cases"]) == 1
        assert result["cases"][0]["pat_id"] == "P1"
        assert result["cases"][0]["diagnosis"] == "Knee pain"
        assert result["cases"][0]["explanation"] == "No physio referral"
        assert result["cases"][0]["guidelines_not_followed"] == [
            "Physiotherapy referral",
        ]

    @pytest.mark.asyncio
    async def test_new_format_includes_fields(self, async_session):
        """New format cases should include judgement, confidence, and cited text."""
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.25, [
            {"diagnosis": "Knee pain", "score": -1, "judgement": "NON-COMPLIANT",
             "confidence": 0.9, "cited_guideline_text": "Exercise therapy recommended.",
             "explanation": "No physio referral",
             "guidelines_not_followed": ["Physiotherapy referral"]},
            {"diagnosis": "Hip pain", "score": -2, "judgement": "RISKY NON-COMPLIANT",
             "confidence": 0.95, "cited_guideline_text": "Do not offer opioids.",
             "explanation": "Opioids prescribed",
             "guidelines_not_followed": ["Avoid opioids"]},
        ])

        result = await get_non_adherent_cases(async_session)
        assert result["total"] == 2

        knee = next(c for c in result["cases"] if c["diagnosis"] == "Knee pain")
        assert knee["score"] == -1
        assert knee["judgement"] == "NON-COMPLIANT"
        assert knee["confidence"] == 0.9
        assert knee["cited_guideline_text"] == "Exercise therapy recommended."

        hip = next(c for c in result["cases"] if c["diagnosis"] == "Hip pain")
        assert hip["score"] == -2
        assert hip["judgement"] == "RISKY NON-COMPLIANT"
        assert hip["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_no_non_adherent(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 1.0, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK"},
        ])

        result = await get_non_adherent_cases(async_session)
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_pagination(self, async_session):
        p = _make_patient(async_session, "P1")
        scores = [
            {"diagnosis": f"Condition {i}", "score": -1,
             "explanation": f"Exp {i}",
             "guidelines_not_followed": [f"Guideline {i}"]}
            for i in range(5)
        ]
        await _add_completed_result(async_session, p, 0.0, scores)

        # Page 1 with page_size=2
        result = await get_non_adherent_cases(
            async_session, page=1, page_size=2,
        )
        assert result["total"] == 5
        assert len(result["cases"]) == 2
        assert result["total_pages"] == 3
        assert result["page"] == 1

        # Page 3 (last page, 1 item)
        result = await get_non_adherent_cases(
            async_session, page=3, page_size=2,
        )
        assert len(result["cases"]) == 1

    @pytest.mark.asyncio
    async def test_page_beyond_data(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.0, [
            {"diagnosis": "Back pain", "score": -1, "explanation": "Bad",
             "guidelines_not_followed": ["Rest"]},
        ])

        result = await get_non_adherent_cases(async_session, page=999)
        assert result["total"] == 1
        assert result["cases"] == []

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.0, [
            {"diagnosis": "Back pain", "score": -1, "explanation": "Bad",
             "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.0, [
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Bad",
             "guidelines_not_followed": []},
        ])

        result = await get_non_adherent_cases(async_session, job_id=job.id)
        assert result["total"] == 1
        assert result["cases"][0]["pat_id"] == "P1"


# ── Test: get_score_distribution ──────────────────────────────────────


class TestScoreDistribution:

    @pytest.mark.asyncio
    async def test_empty(self, async_session):
        dist = await get_score_distribution(async_session)
        assert dist["bins"] == []
        assert dist["total"] == 0

    @pytest.mark.asyncio
    async def test_basic(self, async_session):
        for pat_id, score in [("P1", 0.0), ("P2", 0.5), ("P3", 1.0)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Cond", "score": 1, "explanation": "OK"},
            ])

        dist = await get_score_distribution(async_session, bins=10)
        assert dist["total"] == 3
        assert len(dist["bins"]) == 10
        # 0.0 in first bin [0.0, 0.1)
        assert dist["bins"][0]["count"] == 1
        # 0.5 in bin [0.5, 0.6)
        assert dist["bins"][5]["count"] == 1
        # 1.0 in last bin [0.9, 1.0]
        assert dist["bins"][9]["count"] == 1

    @pytest.mark.asyncio
    async def test_custom_bins(self, async_session):
        for pat_id, score in [("P1", 0.25), ("P2", 0.75)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Cond", "score": 1, "explanation": "OK"},
            ])

        dist = await get_score_distribution(async_session, bins=4)
        assert len(dist["bins"]) == 4
        # [0.0, 0.25)=0, [0.25, 0.5)=1, [0.5, 0.75)=0, [0.75, 1.0]=1
        assert dist["bins"][0]["count"] == 0
        assert dist["bins"][1]["count"] == 1
        assert dist["bins"][2]["count"] == 0
        assert dist["bins"][3]["count"] == 1

    @pytest.mark.asyncio
    async def test_boundary_scores(self, async_session):
        """Scores at exact boundaries (0.0 and 1.0) are counted correctly."""
        for pat_id, score in [("P1", 0.0), ("P2", 1.0)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Cond", "score": 1, "explanation": "OK"},
            ])

        dist = await get_score_distribution(async_session, bins=5)
        assert dist["total"] == 2
        assert dist["bins"][0]["count"] == 1  # 0.0 in [0.0, 0.2)
        assert dist["bins"][4]["count"] == 1  # 1.0 in [0.8, 1.0]

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session):
        p1 = _make_patient(async_session, "P1")
        p2 = _make_patient(async_session, "P2")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.8, [
            {"diagnosis": "Cond", "score": 1, "explanation": "OK"},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Cond", "score": -1, "explanation": "Bad"},
        ])

        dist = await get_score_distribution(async_session, job_id=job.id)
        assert dist["total"] == 1


# ── Test: compute_system_metrics ─────────────────────────────────────


class TestComputeSystemMetrics:

    @pytest.mark.asyncio
    async def test_class_distribution(self, async_session):
        """Score class distribution should count each level correctly."""
        p = _make_patient(async_session, "P1")
        job = await _make_job(async_session)
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "A", "score": 2, "judgement": "Compliant", "confidence": 0.9},
            {"diagnosis": "B", "score": 1, "judgement": "Partial", "confidence": 0.7},
            {"diagnosis": "C", "score": 0, "judgement": "Not Relevant", "confidence": 0.5},
            {"diagnosis": "D", "score": -1, "judgement": "Non-compliant", "confidence": 0.8},
            {"diagnosis": "E", "score": -2, "judgement": "Risky", "confidence": 0.6},
        ], job_id=job.id)
        await async_session.flush()

        metrics = await compute_system_metrics(async_session, job.id)
        assert metrics["score_class_distribution"] == {
            "+2": 1, "+1": 1, "0": 1, "-1": 1, "-2": 1,
        }
        assert metrics["total_diagnoses"] == 5

    @pytest.mark.asyncio
    async def test_adherence_rate(self, async_session):
        """Adherence rate excludes not_relevant and errors."""
        p = _make_patient(async_session, "P1")
        job = await _make_job(async_session)
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "A", "score": 2, "judgement": "Compliant", "confidence": 0.9},
            {"diagnosis": "B", "score": 1, "judgement": "Partial", "confidence": 0.8},
            {"diagnosis": "C", "score": 0, "judgement": "Not Relevant", "confidence": 0.5},
            {"diagnosis": "D", "score": -1, "judgement": "Non-compliant", "confidence": 0.7},
        ], job_id=job.id)
        await async_session.flush()

        metrics = await compute_system_metrics(async_session, job.id)
        # adherent=2 (compliant+partial), total_scored=3 (excl not_relevant)
        assert metrics["adherence_rate"] == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_confidence_stats(self, async_session):
        """Confidence stats should compute mean, median, min, max, std."""
        p = _make_patient(async_session, "P1")
        job = await _make_job(async_session)
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "A", "score": 1, "judgement": "Partial", "confidence": 0.6},
            {"diagnosis": "B", "score": -1, "judgement": "Non-compliant", "confidence": 0.8},
        ], job_id=job.id)
        await async_session.flush()

        metrics = await compute_system_metrics(async_session, job.id)
        conf = metrics["confidence_stats"]
        assert conf["mean"] == pytest.approx(0.7, abs=0.01)
        assert conf["median"] == pytest.approx(0.7, abs=0.01)
        assert conf["min"] == pytest.approx(0.6, abs=0.01)
        assert conf["max"] == pytest.approx(0.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_job(self, async_session):
        """Empty job should return zero counts."""
        job = await _make_job(async_session)
        await async_session.flush()

        metrics = await compute_system_metrics(async_session, job.id)
        assert metrics["total_patients"] == 0
        assert metrics["total_diagnoses"] == 0
        assert metrics["adherence_rate"] == 0.0
