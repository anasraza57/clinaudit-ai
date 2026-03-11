"""
Tests for the model comparison service.

Uses an in-memory SQLite database (via aiosqlite) to test comparison
logic including Cohen's kappa and Pearson correlation.
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
from src.services.comparison import (
    ComparisonResult,
    _compute_auroc,
    compute_cohen_kappa,
    compute_cross_model_classification,
    compute_pearson,
    compare_jobs,
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


def _make_details_json(pat_id: str, scores: list[dict]) -> str:
    """Build a details_json string matching ScoringResult.summary() format."""
    return json.dumps({
        "pat_id": pat_id,
        "total_diagnoses": len(scores),
        "scores": scores,
    })


async def _create_job(
    session: AsyncSession,
    provider: str | None = None,
) -> AuditJob:
    """Create an AuditJob and return it."""
    job = AuditJob(
        status="completed",
        total_patients=0,
        processed_patients=0,
        failed_patients=0,
        provider=provider,
    )
    session.add(job)
    await session.flush()
    return job


async def _add_result(
    session: AsyncSession,
    pat_id: str,
    job: AuditJob,
    overall_score: float,
    diagnosis_scores: list[dict],
) -> AuditResult:
    """Add a completed AuditResult for a patient to a job."""
    # Find or create patient
    from sqlalchemy import select
    result = await session.execute(
        select(Patient).where(Patient.pat_id == pat_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        patient = Patient(pat_id=pat_id)
        session.add(patient)
        await session.flush()

    ar = AuditResult(
        patient_id=patient.id,
        job_id=job.id,
        overall_score=overall_score,
        diagnoses_found=len(diagnosis_scores),
        guidelines_followed=sum(1 for s in diagnosis_scores if s.get("score", 0) >= 1),
        guidelines_not_followed=sum(1 for s in diagnosis_scores if s.get("score", 0) <= -1),
        details_json=_make_details_json(pat_id, diagnosis_scores),
        status="completed",
    )
    session.add(ar)
    await session.flush()
    return ar


# ── Cohen's Kappa Tests ──────────────────────────────────────────────


class TestCohenKappa:
    def test_perfect_agreement(self):
        """Identical labels should give kappa ~1.0."""
        labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        kappa = compute_cohen_kappa(labels, labels)
        assert kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        """Completely opposite labels should give negative kappa."""
        labels_a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        labels_b = [2, 2, 2, 0, 0, 0, 1, 1, 1]
        kappa = compute_cohen_kappa(labels_a, labels_b)
        assert kappa < 0.0

    def test_partial_agreement(self):
        """Mixed agreement should give intermediate kappa."""
        labels_a = [0, 1, 2, 0, 1, 2]
        labels_b = [0, 1, 2, 1, 0, 2]
        kappa = compute_cohen_kappa(labels_a, labels_b)
        assert 0.0 < kappa < 1.0

    def test_all_same_class(self):
        """All same class should give kappa 1.0 (trivial agreement)."""
        labels = [1, 1, 1, 1]
        kappa = compute_cohen_kappa(labels, labels)
        assert kappa == pytest.approx(1.0)

    def test_empty_labels(self):
        """Empty labels should return 0.0."""
        assert compute_cohen_kappa([], []) == 0.0

    def test_mismatched_lengths(self):
        """Different-length lists should return 0.0."""
        assert compute_cohen_kappa([1, 2], [1]) == 0.0


# ── Pearson Correlation Tests ─────────────────────────────────────────


class TestPearson:
    def test_perfect_positive(self):
        """Identical values should give correlation 1.0."""
        values = [0.1, 0.5, 0.9]
        r = compute_pearson(values, values)
        assert r == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Perfectly inverted values should give correlation -1.0."""
        a = [0.0, 0.5, 1.0]
        b = [1.0, 0.5, 0.0]
        r = compute_pearson(a, b)
        assert r == pytest.approx(-1.0)

    def test_no_correlation(self):
        """Uncorrelated values should give correlation ~0.0."""
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 1.0, 3.0]
        r = compute_pearson(a, b)
        assert -0.5 < r < 0.5

    def test_zero_variance(self):
        """Constant values should give correlation 0.0."""
        a = [0.5, 0.5, 0.5]
        b = [0.1, 0.5, 0.9]
        assert compute_pearson(a, b) == 0.0

    def test_too_few_points(self):
        """Less than 2 points should return 0.0."""
        assert compute_pearson([1.0], [1.0]) == 0.0
        assert compute_pearson([], []) == 0.0

    def test_mismatched_lengths(self):
        """Different-length lists should return 0.0."""
        assert compute_pearson([1.0, 2.0], [1.0]) == 0.0


# ── Compare Jobs Tests ────────────────────────────────────────────────


class TestCompareJobs:
    @pytest.mark.asyncio
    async def test_compare_matching_patients(self, async_session):
        """Two jobs with the same patients produce valid comparison."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        scores_a = [
            {"diagnosis": "Low back pain", "index_date": "2024-01-01",
             "score": 2, "judgement": "COMPLIANT"},
            {"diagnosis": "Osteoarthritis", "index_date": "2024-01-01",
             "score": 1, "judgement": "PARTIALLY COMPLIANT"},
        ]
        scores_b = [
            {"diagnosis": "Low back pain", "index_date": "2024-01-01",
             "score": 1, "judgement": "PARTIALLY COMPLIANT"},
            {"diagnosis": "Osteoarthritis", "index_date": "2024-01-01",
             "score": -1, "judgement": "NON-COMPLIANT"},
        ]

        await _add_result(async_session, "PAT001", job_a, 0.875, scores_a)
        await _add_result(async_session, "PAT001", job_b, 0.5, scores_b)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)

        assert isinstance(result, ComparisonResult)
        assert result.job_a_provider == "openai"
        assert result.job_b_provider == "ollama"
        assert result.total_patients_compared == 1
        assert len(result.patients) == 1

        patient = result.patients[0]
        assert patient.pat_id == "PAT001"
        assert patient.score_a == pytest.approx(0.875)
        assert patient.score_b == pytest.approx(0.5)
        assert patient.score_diff == pytest.approx(0.375)
        assert len(patient.per_diagnosis) == 2

        # Low back pain: both adherent (+2 and +1) → agree
        lbp = next(d for d in patient.per_diagnosis if d.diagnosis == "Low back pain")
        assert lbp.score_a == 2
        assert lbp.score_b == 1
        assert lbp.agreement is True

        # Osteoarthritis: +1 vs -1 → disagree
        oa = next(d for d in patient.per_diagnosis if d.diagnosis == "Osteoarthritis")
        assert oa.score_a == 1
        assert oa.score_b == -1
        assert oa.agreement is False

    @pytest.mark.asyncio
    async def test_compare_partial_overlap(self, async_session):
        """Only overlapping patients are included in comparison."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        scores = [{"diagnosis": "Knee pain", "index_date": "2024-01-01",
                    "score": 1, "judgement": "PARTIALLY COMPLIANT"}]

        # PAT001 in both, PAT002 only in A, PAT003 only in B
        await _add_result(async_session, "PAT001", job_a, 0.75, scores)
        await _add_result(async_session, "PAT002", job_a, 0.5, scores)
        await _add_result(async_session, "PAT001", job_b, 0.75, scores)
        await _add_result(async_session, "PAT003", job_b, 0.25, scores)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        assert result.total_patients_compared == 1
        assert result.patients[0].pat_id == "PAT001"

    @pytest.mark.asyncio
    async def test_compare_with_no_overlap(self, async_session):
        """Jobs with no shared patients produce empty comparison."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        scores = [{"diagnosis": "Sciatica", "index_date": "2024-01-01",
                    "score": 0, "judgement": "NOT RELEVANT"}]

        await _add_result(async_session, "PAT001", job_a, 0.5, scores)
        await _add_result(async_session, "PAT002", job_b, 0.5, scores)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        assert result.total_patients_compared == 0
        assert result.mean_score_a == 0.0
        assert result.mean_score_b == 0.0
        assert result.agreement_rate == 0.0

    @pytest.mark.asyncio
    async def test_compare_with_no_results(self, async_session):
        """Empty jobs produce zero metrics."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        assert result.total_patients_compared == 0
        assert result.cohen_kappa == 0.0
        assert result.score_correlation == 0.0

    @pytest.mark.asyncio
    async def test_compare_nonexistent_job(self, async_session):
        """Comparing with a nonexistent job raises ValueError."""
        job_a = await _create_job(async_session)
        await async_session.commit()

        with pytest.raises(ValueError, match="not found"):
            await compare_jobs(async_session, job_a.id, 9999)

    @pytest.mark.asyncio
    async def test_per_condition_comparison(self, async_session):
        """Per-condition adherence deltas are computed correctly."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        # Patient 1: LBP compliant in both
        scores_1a = [{"diagnosis": "Low back pain", "index_date": "2024-01-01",
                       "score": 2, "judgement": "COMPLIANT"}]
        scores_1b = [{"diagnosis": "Low back pain", "index_date": "2024-01-01",
                       "score": 1, "judgement": "PARTIALLY COMPLIANT"}]

        # Patient 2: LBP non-compliant in B only
        scores_2a = [{"diagnosis": "Low back pain", "index_date": "2024-02-01",
                       "score": 1, "judgement": "PARTIALLY COMPLIANT"}]
        scores_2b = [{"diagnosis": "Low back pain", "index_date": "2024-02-01",
                       "score": -1, "judgement": "NON-COMPLIANT"}]

        await _add_result(async_session, "PAT001", job_a, 1.0, scores_1a)
        await _add_result(async_session, "PAT001", job_b, 0.75, scores_1b)
        await _add_result(async_session, "PAT002", job_a, 0.75, scores_2a)
        await _add_result(async_session, "PAT002", job_b, 0.25, scores_2b)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)

        # Both patients have LBP → 2 cases for that condition
        assert len(result.per_condition) == 1
        lbp = result.per_condition[0]
        assert lbp.condition == "Low back pain"
        assert lbp.count == 2

        # Job A: 2/2 adherent (scores +2, +1)
        assert lbp.adherence_rate_a == pytest.approx(1.0)
        # Job B: 1/2 adherent (scores +1, -1)
        assert lbp.adherence_rate_b == pytest.approx(0.5)
        # Delta
        assert lbp.diff == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_perfect_agreement_gives_high_kappa(self, async_session):
        """Identical scores across jobs produce kappa ~1.0."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        # Use varying overall_scores so Pearson correlation is meaningful
        patient_data = [
            ("PAT001", 0.25, [{"diagnosis": "Low back pain", "index_date": "2024-01-01",
                                "score": -1, "judgement": "NON-COMPLIANT"}]),
            ("PAT002", 0.5, [{"diagnosis": "Knee pain", "index_date": "2024-01-01",
                               "score": 0, "judgement": "NOT RELEVANT"}]),
            ("PAT003", 1.0, [{"diagnosis": "Shoulder pain", "index_date": "2024-01-01",
                               "score": 2, "judgement": "COMPLIANT"}]),
        ]

        for pat_id, overall, scores in patient_data:
            await _add_result(async_session, pat_id, job_a, overall, scores)
            await _add_result(async_session, pat_id, job_b, overall, scores)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        assert result.cohen_kappa == pytest.approx(1.0)
        assert result.agreement_rate == pytest.approx(1.0)
        assert result.score_correlation == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_summary_serializable(self, async_session):
        """ComparisonResult.summary() produces a JSON-serializable dict."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        scores = [{"diagnosis": "Hip pain", "index_date": "2024-01-01",
                    "score": 0, "judgement": "NOT RELEVANT"}]
        await _add_result(async_session, "PAT001", job_a, 0.5, scores)
        await _add_result(async_session, "PAT001", job_b, 0.5, scores)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        summary = result.summary()

        # Should be JSON-serializable
        serialized = json.dumps(summary)
        assert serialized is not None

        # Check structure
        assert "job_a_id" in summary
        assert "cohen_kappa" in summary
        assert "patients" in summary
        assert "per_condition" in summary
        assert len(summary["patients"]) == 1

    @pytest.mark.asyncio
    async def test_compare_multiple_patients_aggregate(self, async_session):
        """Aggregate metrics computed correctly across multiple patients."""
        job_a = await _create_job(async_session, provider="openai")
        job_b = await _create_job(async_session, provider="ollama")

        scores_template = [{"diagnosis": "Shoulder pain", "index_date": "2024-01-01",
                            "score": 1, "judgement": "PARTIALLY COMPLIANT"}]

        # PAT001: 0.8 vs 0.6, PAT002: 0.4 vs 0.2
        await _add_result(async_session, "PAT001", job_a, 0.8, scores_template)
        await _add_result(async_session, "PAT001", job_b, 0.6, scores_template)
        await _add_result(async_session, "PAT002", job_a, 0.4, scores_template)
        await _add_result(async_session, "PAT002", job_b, 0.2, scores_template)
        await async_session.commit()

        result = await compare_jobs(async_session, job_a.id, job_b.id)
        assert result.total_patients_compared == 2
        assert result.mean_score_a == pytest.approx(0.6)  # (0.8 + 0.4) / 2
        assert result.mean_score_b == pytest.approx(0.4)  # (0.6 + 0.2) / 2
        assert result.mean_abs_diff == pytest.approx(0.2)  # (0.2 + 0.2) / 2


# ── AUROC Tests ──────────────────────────────────────────────────────


class TestComputeAuroc:
    def test_perfect_separator(self):
        """Perfect separation should give AUROC ~1.0."""
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        assert _compute_auroc(labels, scores) == pytest.approx(1.0)

    def test_inverse_separator(self):
        """Inverse perfect separation should give AUROC ~0.0."""
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        assert _compute_auroc(labels, scores) == pytest.approx(0.0)

    def test_non_discriminative_gives_around_half(self):
        """Non-predictive scores should give AUROC ~0.5."""
        # Alternating labels with ascending scores = no discrimination
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        auroc = _compute_auroc(labels, scores)
        assert auroc == pytest.approx(0.5, abs=0.2)

    def test_empty_returns_none(self):
        assert _compute_auroc([], []) is None

    def test_single_class_returns_none(self):
        """Only one class should return None."""
        assert _compute_auroc([1, 1, 1], [0.9, 0.8, 0.7]) is None
        assert _compute_auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None


# ── Cross-Model Classification Tests ────────────────────────────────


class TestCrossModelClassification:

    @pytest.mark.asyncio
    async def test_confusion_matrix(self, async_session):
        """Confusion matrix should correctly count score pairs."""
        job_a = await _create_job(async_session, "openai")
        job_b = await _create_job(async_session, "ollama")

        scores_a = [
            {"diagnosis": "Pain", "index_date": "2024-01-01",
             "score": 1, "judgement": "Partial", "confidence": 0.8},
            {"diagnosis": "Fracture", "index_date": "2024-01-01",
             "score": -1, "judgement": "Non-compliant", "confidence": 0.7},
        ]
        scores_b = [
            {"diagnosis": "Pain", "index_date": "2024-01-01",
             "score": 1, "judgement": "Partial", "confidence": 0.9},
            {"diagnosis": "Fracture", "index_date": "2024-01-01",
             "score": 0, "judgement": "Not Relevant", "confidence": 0.5},
        ]

        await _add_result(async_session, "P1", job_a, 0.5, scores_a)
        await _add_result(async_session, "P1", job_b, 0.5, scores_b)
        await async_session.commit()

        result = await compute_cross_model_classification(
            async_session, job_a.id, job_b.id,
        )

        assert result["total_diagnoses_compared"] == 2
        assert result["confusion_matrix"]["labels"] == ["-2", "-1", "0", "+1", "+2"]
        # Pain: A=+1, B=+1 → matrix[3][3] = 1
        # Fracture: A=-1, B=0 → matrix[1][2] = 1
        matrix = result["confusion_matrix"]["matrix"]
        assert matrix[3][3] == 1  # +1 vs +1
        assert matrix[1][2] == 1  # -1 vs 0

    @pytest.mark.asyncio
    async def test_exact_match_accuracy(self, async_session):
        """Exact match accuracy should count identical scores."""
        job_a = await _create_job(async_session, "openai")
        job_b = await _create_job(async_session, "ollama")

        # Both agree on this diagnosis
        scores = [
            {"diagnosis": "Pain", "index_date": "2024-01-01",
             "score": 1, "judgement": "Partial", "confidence": 0.8},
        ]

        await _add_result(async_session, "P1", job_a, 0.5, scores)
        await _add_result(async_session, "P1", job_b, 0.5, scores)
        await async_session.commit()

        result = await compute_cross_model_classification(
            async_session, job_a.id, job_b.id,
        )
        assert result["exact_match_accuracy"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_providers_included(self, async_session):
        """Result should include provider names."""
        job_a = await _create_job(async_session, "openai")
        job_b = await _create_job(async_session, "ollama")

        scores = [{"diagnosis": "X", "index_date": "2024-01-01",
                    "score": 0, "judgement": "Not Relevant", "confidence": 0.5}]
        await _add_result(async_session, "P1", job_a, 0.5, scores)
        await _add_result(async_session, "P1", job_b, 0.5, scores)
        await async_session.commit()

        result = await compute_cross_model_classification(
            async_session, job_a.id, job_b.id,
        )
        assert result["job_a_provider"] == "openai"
        assert result["job_b_provider"] == "ollama"
