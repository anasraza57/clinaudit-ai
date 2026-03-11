"""
Tests for missing care opportunity detection.

Tests parsing of missing_care_opportunities from LLM responses,
backward compatibility, and reporting aggregation.
"""

import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.agents.scorer import (
    DiagnosisScore,
    ScoringResult,
    parse_scoring_response,
)
from src.models.audit import AuditJob, AuditResult
from src.models.base import Base
from src.models.patient import Patient
from src.services.reporting import get_missing_care_summary


# ── Parsing Tests ─────────────────────────────────────────────────────


class TestMissingCareParsing:
    def test_parse_with_missing_care(self):
        """Extract missing care opportunities from LLM response."""
        response = """Score: +1
Judgement: PARTIALLY COMPLIANT
Confidence: 0.75
Cited Guideline: "Consider referral to physiotherapy"
Explanation: The GP referred the patient but did not prescribe NSAIDs.
Guidelines Followed: Physiotherapy referral
Guidelines Not Followed: NSAID prescription
Missing Care Opportunities: Exercise therapy advice, Weight management referral"""

        parsed = parse_scoring_response(response)
        assert parsed["missing_care_opportunities"] == [
            "Exercise therapy advice",
            "Weight management referral",
        ]

    def test_parse_with_none_missing(self):
        """'None' response produces empty list."""
        response = """Score: +2
Judgement: COMPLIANT
Confidence: 0.9
Cited Guideline: "Refer to physiotherapy"
Explanation: Fully compliant management.
Guidelines Followed: Physiotherapy referral, NSAID prescription
Guidelines Not Followed: None
Missing Care Opportunities: None"""

        parsed = parse_scoring_response(response)
        assert parsed["missing_care_opportunities"] == []

    def test_parse_without_field(self):
        """Backward compat: old responses without the field produce empty list."""
        response = """Score: +1
Judgement: PARTIALLY COMPLIANT
Confidence: 0.7
Cited Guideline: "Some guideline"
Explanation: Explanation here.
Guidelines Followed: Some action
Guidelines Not Followed: Some other action"""

        parsed = parse_scoring_response(response)
        assert parsed["missing_care_opportunities"] == []

    def test_parse_single_opportunity(self):
        """Single missing care opportunity parsed correctly."""
        response = """Score: -1
Judgement: NON-COMPLIANT
Confidence: 0.8
Cited Guideline: "Offer structured exercise programme"
Explanation: No treatment documented.
Guidelines Followed: None
Guidelines Not Followed: Structured exercise programme
Missing Care Opportunities: Structured exercise programme referral"""

        parsed = parse_scoring_response(response)
        assert parsed["missing_care_opportunities"] == [
            "Structured exercise programme referral",
        ]

    def test_guidelines_not_followed_still_parsed_with_missing_care(self):
        """Guidelines Not Followed is still parsed when Missing Care field follows."""
        response = """Score: -1
Judgement: NON-COMPLIANT
Confidence: 0.6
Cited Guideline: "Offer exercise therapy"
Explanation: No actions documented.
Guidelines Followed: None
Guidelines Not Followed: Exercise therapy, NSAID prescription
Missing Care Opportunities: Exercise therapy advice"""

        parsed = parse_scoring_response(response)
        assert parsed["guidelines_not_followed"] == [
            "Exercise therapy",
            "NSAID prescription",
        ]
        assert parsed["missing_care_opportunities"] == [
            "Exercise therapy advice",
        ]


# ── Data Class Tests ──────────────────────────────────────────────────


class TestMissingCareInDataClasses:
    def test_diagnosis_score_has_field(self):
        """DiagnosisScore includes missing_care_opportunities."""
        ds = DiagnosisScore(
            diagnosis_term="Low back pain",
            concept_id="123",
            index_date="2024-01-01",
            score=1,
            judgement="PARTIALLY COMPLIANT",
            explanation="Partial compliance.",
            missing_care_opportunities=["Exercise therapy", "Weight management"],
        )
        assert ds.missing_care_opportunities == ["Exercise therapy", "Weight management"]

    def test_diagnosis_score_default_empty(self):
        """Missing care defaults to empty list."""
        ds = DiagnosisScore(
            diagnosis_term="Test",
            concept_id="456",
            index_date="2024-01-01",
            score=2,
            judgement="COMPLIANT",
            explanation="Full compliance.",
        )
        assert ds.missing_care_opportunities == []

    def test_summary_includes_missing_care(self):
        """ScoringResult.summary() includes missing_care_opportunities."""
        ds = DiagnosisScore(
            diagnosis_term="Low back pain",
            concept_id="123",
            index_date="2024-01-01",
            score=-1,
            judgement="NON-COMPLIANT",
            explanation="Non-compliant.",
            missing_care_opportunities=["Physiotherapy referral"],
        )
        result = ScoringResult(
            pat_id="PAT001",
            diagnosis_scores=[ds],
            total_diagnoses=1,
            non_compliant_count=1,
        )
        summary = result.summary()
        assert summary["scores"][0]["missing_care_opportunities"] == ["Physiotherapy referral"]


# ── Reporting Tests ───────────────────────────────────────────────────


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


def _make_details_json(pat_id: str, scores: list[dict]) -> str:
    return json.dumps({
        "pat_id": pat_id,
        "total_diagnoses": len(scores),
        "scores": scores,
    })


class TestMissingCareReporting:
    @pytest.mark.asyncio
    async def test_aggregation_groups_by_condition(self, async_session):
        """Missing care opportunities are grouped by condition."""
        patient = Patient(pat_id="PAT001")
        async_session.add(patient)
        await async_session.flush()

        scores = [
            {
                "diagnosis": "Low back pain",
                "index_date": "2024-01-01",
                "score": -1,
                "judgement": "NON-COMPLIANT",
                "missing_care_opportunities": ["Exercise therapy", "NSAID prescription"],
            },
            {
                "diagnosis": "Knee pain",
                "index_date": "2024-01-01",
                "score": 1,
                "judgement": "PARTIALLY COMPLIANT",
                "missing_care_opportunities": ["Weight management"],
            },
        ]

        result = AuditResult(
            patient_id=patient.id,
            overall_score=0.375,
            diagnoses_found=2,
            guidelines_followed=1,
            guidelines_not_followed=1,
            details_json=_make_details_json("PAT001", scores),
            status="completed",
        )
        async_session.add(result)
        await async_session.commit()

        summary = await get_missing_care_summary(async_session)

        assert summary["total_patients"] == 1
        assert summary["total_opportunities"] == 3
        assert len(summary["opportunities_by_condition"]) == 2

        # LBP should be first (2 opportunities > 1)
        lbp = summary["opportunities_by_condition"][0]
        assert lbp["condition"] == "Low back pain"
        assert lbp["total_opportunities"] == 2
        assert len(lbp["opportunities"]) == 2

    @pytest.mark.asyncio
    async def test_no_missing_care_returns_empty(self, async_session):
        """Patients without missing care produce empty summary."""
        patient = Patient(pat_id="PAT002")
        async_session.add(patient)
        await async_session.flush()

        scores = [
            {
                "diagnosis": "Hip pain",
                "index_date": "2024-01-01",
                "score": 2,
                "judgement": "COMPLIANT",
                # No missing_care_opportunities field
            },
        ]

        result = AuditResult(
            patient_id=patient.id,
            overall_score=1.0,
            diagnoses_found=1,
            guidelines_followed=1,
            guidelines_not_followed=0,
            details_json=_make_details_json("PAT002", scores),
            status="completed",
        )
        async_session.add(result)
        await async_session.commit()

        summary = await get_missing_care_summary(async_session)
        assert summary["total_opportunities"] == 0
        assert summary["opportunities_by_condition"] == []
        assert summary["cases"] == []

    @pytest.mark.asyncio
    async def test_job_id_scoping(self, async_session):
        """Missing care respects job_id filter."""
        job = AuditJob(
            status="completed",
            total_patients=1,
            processed_patients=1,
            failed_patients=0,
        )
        async_session.add(job)
        await async_session.flush()

        patient = Patient(pat_id="PAT003")
        async_session.add(patient)
        await async_session.flush()

        # Result in the job
        scores_in = [{"diagnosis": "Sciatica", "score": -1, "judgement": "NON-COMPLIANT",
                       "missing_care_opportunities": ["MRI referral"]}]
        r1 = AuditResult(
            patient_id=patient.id, job_id=job.id,
            overall_score=0.25, diagnoses_found=1,
            guidelines_followed=0, guidelines_not_followed=1,
            details_json=_make_details_json("PAT003", scores_in),
            status="completed",
        )

        # Result NOT in the job
        scores_out = [{"diagnosis": "Ankle pain", "score": -1, "judgement": "NON-COMPLIANT",
                        "missing_care_opportunities": ["X-ray"]}]
        r2 = AuditResult(
            patient_id=patient.id, job_id=None,
            overall_score=0.25, diagnoses_found=1,
            guidelines_followed=0, guidelines_not_followed=1,
            details_json=_make_details_json("PAT003", scores_out),
            status="completed",
        )
        async_session.add_all([r1, r2])
        await async_session.commit()

        summary = await get_missing_care_summary(async_session, job_id=job.id)
        assert summary["total_opportunities"] == 1
        assert summary["opportunities_by_condition"][0]["condition"] == "Sciatica"

    @pytest.mark.asyncio
    async def test_frequency_counting(self, async_session):
        """Same opportunity across patients is counted correctly."""
        for i, pat_id in enumerate(["PAT010", "PAT011", "PAT012"]):
            patient = Patient(pat_id=pat_id)
            async_session.add(patient)
            await async_session.flush()

            scores = [{"diagnosis": "Low back pain", "score": -1,
                        "judgement": "NON-COMPLIANT",
                        "missing_care_opportunities": ["Exercise therapy"]}]
            r = AuditResult(
                patient_id=patient.id,
                overall_score=0.25, diagnoses_found=1,
                guidelines_followed=0, guidelines_not_followed=1,
                details_json=_make_details_json(pat_id, scores),
                status="completed",
            )
            async_session.add(r)

        await async_session.commit()

        summary = await get_missing_care_summary(async_session)
        assert summary["total_opportunities"] == 3
        lbp = summary["opportunities_by_condition"][0]
        assert lbp["opportunities"][0]["action"] == "Exercise therapy"
        assert lbp["opportunities"][0]["count"] == 3
