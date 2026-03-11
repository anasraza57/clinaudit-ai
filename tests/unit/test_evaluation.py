"""
Tests for LLM-as-Judge evaluation service.

Tests rating parsing, per-agent evaluation functions (extractor weak
supervision, query/retriever/scorer LLM-as-Judge), pipeline-level
orchestration, aggregation, and stored-data reconstruction.
"""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from src.agents.extractor import CategorisedEntry, ExtractionResult, PatientEpisode
from src.agents.query import DiagnosisQueries, QueryResult
from src.agents.retriever import DiagnosisGuidelines, GuidelineMatch, RetrievalResult
from src.agents.scorer import DiagnosisScore, ScoringResult
from src.services.evaluation import (
    AggregateEvaluation,
    PipelineEvaluation,
    ScorerMetrics,
    _parse_rating,
    aggregate_evaluations,
    evaluate_extractor,
    evaluate_extractor_from_db,
    evaluate_patient,
    evaluate_queries,
    evaluate_retrieval,
    evaluate_retrieval_ir,
    evaluate_scoring,
    scoring_from_stored,
)
from src.services.pipeline import PipelineResult


# ── Mock AI provider ─────────────────────────────────────────────────


class MockAIProvider:
    """Deterministic mock for AI provider."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[str] = []

    async def chat_simple(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response

    @property
    def provider_name(self) -> str:
        return "mock"


# ── Rating parsing ───────────────────────────────────────────────────


class TestParseRating:
    def test_parse_simple_rating(self):
        text = "Relevance: 4\nCoverage: 3"
        assert _parse_rating(text, "Relevance") == 4
        assert _parse_rating(text, "Coverage") == 3

    def test_parse_missing_field_defaults_to_3(self):
        assert _parse_rating("Nothing here", "Relevance") == 3

    def test_parse_clamps_to_range(self):
        assert _parse_rating("Relevance: 9", "Relevance") == 5
        assert _parse_rating("Relevance: 0", "Relevance") == 1

    def test_parse_case_insensitive(self):
        assert _parse_rating("relevance: 4", "Relevance") == 4
        assert _parse_rating("RELEVANCE: 4", "Relevance") == 4

    def test_parse_multiword_field(self):
        text = "Reasoning Quality: 5\nCitation Accuracy: 3\nScore Calibration: 4"
        assert _parse_rating(text, "Reasoning Quality") == 5
        assert _parse_rating(text, "Citation Accuracy") == 3
        assert _parse_rating(text, "Score Calibration") == 4

    def test_parse_slash_format(self):
        """First number after colon is used, ignoring 'N/5' denominator."""
        assert _parse_rating("Relevance: 4/5", "Relevance") == 4


# ── Extractor evaluation ────────────────────────────────────────────


class TestExtractorEvaluation:
    def test_perfect_match(self):
        """All extractor categories match SNOMED rules."""
        extraction = ExtractionResult(
            pat_id="PAT001",
            episodes=[
                PatientEpisode(
                    index_date=date(2024, 1, 1),
                    entries=[
                        CategorisedEntry(
                            concept_id="1",
                            term="pain",
                            concept_display="Low back pain",
                            cons_date=date(2024, 1, 1),
                            category="diagnosis",
                        ),
                        CategorisedEntry(
                            concept_id="2",
                            term="referral",
                            concept_display="Referral to physiotherapy",
                            cons_date=date(2024, 1, 1),
                            category="referral",
                        ),
                    ],
                ),
            ],
            total_entries=2,
            total_diagnoses=1,
        )
        raw_entries = [
            {"concept_display": "Low back pain"},
            {"concept_display": "Referral to physiotherapy"},
        ]

        metrics = evaluate_extractor(extraction, raw_entries)
        assert metrics.rule_match_rate == 1.0
        assert metrics.total_entries == 2
        assert metrics.rule_matched == 2

    def test_empty_entries(self):
        """Empty entries produce empty metrics."""
        extraction = ExtractionResult(pat_id="PAT001")
        metrics = evaluate_extractor(extraction, [])
        assert metrics.total_entries == 0
        assert metrics.rule_match_rate == 0.0

    def test_mismatch_tracked(self):
        """Mismatched categories affect P/R/F1."""
        extraction = ExtractionResult(
            pat_id="PAT001",
            episodes=[
                PatientEpisode(
                    index_date=date(2024, 1, 1),
                    entries=[
                        CategorisedEntry(
                            concept_id="1",
                            term="x-ray",
                            concept_display="X-ray of knee",
                            cons_date=date(2024, 1, 1),
                            category="treatment",  # Wrong — rules say investigation
                        ),
                    ],
                ),
            ],
            total_entries=1,
            total_diagnoses=0,
        )
        raw_entries = [{"concept_display": "X-ray of knee"}]

        metrics = evaluate_extractor(extraction, raw_entries)
        assert metrics.rule_match_rate == 0.0
        assert metrics.rule_matched == 0
        # investigation has a false negative, treatment has a false positive
        assert "investigation" in metrics.per_category
        assert metrics.per_category["investigation"]["fn"] == 1
        assert "treatment" in metrics.per_category
        assert metrics.per_category["treatment"]["fp"] == 1


# ── Query evaluation ─────────────────────────────────────────────────


class TestQueryEvaluation:
    @pytest.mark.asyncio
    async def test_good_queries(self):
        """High relevance and coverage ratings are captured."""
        provider = MockAIProvider("Relevance: 5\nCoverage: 4")
        query_result = QueryResult(
            pat_id="PAT001",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    queries=[
                        "NICE low back pain management",
                        "back pain treatment guidelines",
                    ],
                    source="template",
                ),
            ],
            total_diagnoses=1,
            total_queries=2,
        )

        metrics = await evaluate_queries(query_result, provider)
        assert metrics.mean_relevance == 5.0
        assert metrics.mean_coverage == 4.0
        assert len(metrics.per_diagnosis) == 1
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_queries(self):
        """Empty query result produces empty metrics."""
        provider = MockAIProvider()
        metrics = await evaluate_queries(
            QueryResult(pat_id="PAT001"), provider,
        )
        assert metrics.total_diagnoses == 0

    @pytest.mark.asyncio
    async def test_failed_judge_defaults_to_3(self):
        """Failed LLM judge call defaults ratings to 3."""
        provider = MockAIProvider()
        provider.chat_simple = AsyncMock(side_effect=Exception("API error"))

        query_result = QueryResult(
            pat_id="PAT001",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Sciatica",
                    concept_id="456",
                    index_date="2024-01-01",
                    queries=["sciatica treatment"],
                    source="template",
                ),
            ],
        )

        metrics = await evaluate_queries(query_result, provider)
        assert metrics.mean_relevance == 3.0
        assert metrics.mean_coverage == 3.0


# ── Retriever evaluation ─────────────────────────────────────────────


class TestRetrieverEvaluation:
    @pytest.mark.asyncio
    async def test_relevant_guidelines(self):
        """High relevance rating for good guidelines."""
        provider = MockAIProvider("Relevance: 5")
        retrieval = RetrievalResult(
            pat_id="PAT001",
            diagnosis_guidelines=[
                DiagnosisGuidelines(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    guidelines=[
                        GuidelineMatch(
                            guideline_id="NG59",
                            title="Low back pain and sciatica",
                            source="NICE",
                            url="https://nice.org.uk/ng59",
                            clean_text="Offer structured exercise programme...",
                            score=0.1,
                            rank=1,
                            matched_query="back pain treatment",
                        ),
                    ],
                ),
            ],
            total_diagnoses=1,
            total_guidelines=1,
        )

        metrics = await evaluate_retrieval(retrieval, provider)
        assert metrics.mean_relevance == 5.0
        assert metrics.total_guidelines == 1

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Empty retrieval produces empty metrics."""
        provider = MockAIProvider()
        metrics = await evaluate_retrieval(
            RetrievalResult(pat_id="PAT001"), provider,
        )
        assert metrics.total_diagnoses == 0


# ── Scorer evaluation ────────────────────────────────────────────────


class TestScorerEvaluation:
    @pytest.mark.asyncio
    async def test_good_scoring(self):
        """High quality scoring gets high ratings."""
        provider = MockAIProvider(
            "Reasoning Quality: 5\n"
            "Citation Accuracy: 4\n"
            "Score Calibration: 5",
        )
        scoring = ScoringResult(
            pat_id="PAT001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    score=2,
                    judgement="COMPLIANT",
                    explanation="All recommended guidelines were followed.",
                    cited_guideline_text="Offer exercise therapy",
                    guidelines_followed=["Exercise therapy", "NSAID"],
                    guidelines_not_followed=[],
                ),
            ],
            total_diagnoses=1,
            compliant_count=1,
        )

        metrics = await evaluate_scoring(scoring, provider)
        assert metrics.mean_reasoning_quality == 5.0
        assert metrics.mean_citation_accuracy == 4.0
        assert metrics.mean_score_calibration == 5.0

    @pytest.mark.asyncio
    async def test_skips_error_diagnoses(self):
        """Diagnoses with errors are skipped."""
        provider = MockAIProvider(
            "Reasoning Quality: 5\n"
            "Citation Accuracy: 5\n"
            "Score Calibration: 5",
        )
        scoring = ScoringResult(
            pat_id="PAT001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Error case",
                    concept_id="999",
                    index_date="2024-01-01",
                    score=0,
                    judgement="ERROR",
                    explanation="",
                    error="LLM call failed",
                ),
            ],
            total_diagnoses=1,
            error_count=1,
        )

        metrics = await evaluate_scoring(scoring, provider)
        assert metrics.total_diagnoses == 0
        assert len(provider.calls) == 0

    @pytest.mark.asyncio
    async def test_empty_scoring(self):
        """Empty scoring produces empty metrics."""
        provider = MockAIProvider()
        metrics = await evaluate_scoring(
            ScoringResult(pat_id="PAT001"), provider,
        )
        assert metrics.total_diagnoses == 0


# ── Pipeline evaluation ──────────────────────────────────────────────


class TestPipelineEvaluation:
    @pytest.mark.asyncio
    async def test_full_pipeline_evaluation(self):
        """All agents are evaluated when present in pipeline result."""
        provider = MockAIProvider(
            "Relevance: 4\n"
            "Coverage: 3\n"
            "Reasoning Quality: 5\n"
            "Citation Accuracy: 4\n"
            "Score Calibration: 5",
        )

        extraction = ExtractionResult(
            pat_id="PAT001",
            episodes=[
                PatientEpisode(
                    index_date=date(2024, 1, 1),
                    entries=[
                        CategorisedEntry(
                            concept_id="1",
                            term="pain",
                            concept_display="Low back pain",
                            cons_date=date(2024, 1, 1),
                            category="diagnosis",
                        ),
                    ],
                ),
            ],
            total_entries=1,
            total_diagnoses=1,
        )

        query_result = QueryResult(
            pat_id="PAT001",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    queries=["NICE low back pain"],
                    source="template",
                ),
            ],
            total_diagnoses=1,
            total_queries=1,
        )

        retrieval = RetrievalResult(
            pat_id="PAT001",
            diagnosis_guidelines=[
                DiagnosisGuidelines(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    guidelines=[
                        GuidelineMatch(
                            guideline_id="NG59",
                            title="Low back pain",
                            source="NICE",
                            url="https://nice.org.uk/ng59",
                            clean_text="Offer exercise...",
                            score=0.1,
                            rank=1,
                            matched_query="back pain",
                        ),
                    ],
                ),
            ],
            total_diagnoses=1,
            total_guidelines=1,
        )

        scoring = ScoringResult(
            pat_id="PAT001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    score=2,
                    judgement="COMPLIANT",
                    explanation="All guidelines followed.",
                ),
            ],
            total_diagnoses=1,
            compliant_count=1,
        )

        pipeline = PipelineResult(
            pat_id="PAT001",
            extraction=extraction,
            query_result=query_result,
            retrieval=retrieval,
            scoring=scoring,
            stage_reached="scoring",
        )

        raw_entries = [{"concept_display": "Low back pain"}]

        evaluation = await evaluate_patient(
            pipeline, raw_entries, provider,
        )
        assert evaluation.extractor is not None
        assert evaluation.query is not None
        assert evaluation.retriever is not None
        assert evaluation.scorer is not None

    @pytest.mark.asyncio
    async def test_selective_agent_evaluation(self):
        """Only selected agents are evaluated."""
        provider = MockAIProvider(
            "Reasoning Quality: 5\n"
            "Citation Accuracy: 4\n"
            "Score Calibration: 5",
        )

        scoring = ScoringResult(
            pat_id="PAT001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Low back pain",
                    concept_id="123",
                    index_date="2024-01-01",
                    score=2,
                    judgement="COMPLIANT",
                    explanation="All guidelines followed.",
                ),
            ],
            total_diagnoses=1,
            compliant_count=1,
        )

        pipeline = PipelineResult(
            pat_id="PAT001",
            scoring=scoring,
            stage_reached="scoring",
        )

        evaluation = await evaluate_patient(
            pipeline, [], provider, agents=["scorer"],
        )
        assert evaluation.scorer is not None
        assert evaluation.extractor is None
        assert evaluation.query is None
        assert evaluation.retriever is None


# ── Aggregation ──────────────────────────────────────────────────────


class TestAggregateEvaluations:
    def test_aggregate_multiple_patients(self):
        """Aggregate metrics from multiple patient evaluations."""
        ev1 = PipelineEvaluation(
            pat_id="PAT001",
            scorer=ScorerMetrics(
                total_diagnoses=1,
                mean_reasoning_quality=5.0,
                mean_citation_accuracy=4.0,
                mean_score_calibration=5.0,
                per_diagnosis=[{
                    "diagnosis": "Low back pain",
                    "score": 2,
                    "judgement": "COMPLIANT",
                    "reasoning_quality": 5,
                    "citation_accuracy": 4,
                    "score_calibration": 5,
                }],
            ),
        )
        ev2 = PipelineEvaluation(
            pat_id="PAT002",
            scorer=ScorerMetrics(
                total_diagnoses=1,
                mean_reasoning_quality=3.0,
                mean_citation_accuracy=2.0,
                mean_score_calibration=3.0,
                per_diagnosis=[{
                    "diagnosis": "Knee pain",
                    "score": -1,
                    "judgement": "NON-COMPLIANT",
                    "reasoning_quality": 3,
                    "citation_accuracy": 2,
                    "score_calibration": 3,
                }],
            ),
        )

        agg = aggregate_evaluations([ev1, ev2])
        assert agg.total_patients == 2
        assert agg.scorer is not None
        assert agg.scorer.mean_reasoning_quality == 4.0
        assert agg.scorer.mean_citation_accuracy == 3.0
        assert agg.scorer.mean_score_calibration == 4.0

    def test_aggregate_empty(self):
        """Empty list returns empty aggregate."""
        agg = aggregate_evaluations([])
        assert agg.total_patients == 0
        assert agg.scorer is None
        assert agg.extractor is None


# ── scoring_from_stored ──────────────────────────────────────────────


class TestScoringFromStored:
    def test_reconstructs_scoring_result(self):
        """ScoringResult is correctly reconstructed from stored JSON dict."""
        details = {
            "pat_id": "PAT001",
            "total_diagnoses": 2,
            "scores": [
                {
                    "diagnosis": "Low back pain",
                    "concept_id": "123",
                    "index_date": "2024-01-01",
                    "score": 2,
                    "judgement": "COMPLIANT",
                    "explanation": "All followed.",
                    "confidence": 0.9,
                    "cited_guideline_text": "Offer exercise",
                    "guidelines_followed": ["Exercise"],
                    "guidelines_not_followed": [],
                    "missing_care_opportunities": [],
                },
                {
                    "diagnosis": "Knee pain",
                    "score": -1,
                    "judgement": "NON-COMPLIANT",
                    "explanation": "No treatment documented.",
                    "error": None,
                },
            ],
        }

        scoring = scoring_from_stored(details)
        assert scoring.pat_id == "PAT001"
        assert scoring.total_diagnoses == 2
        assert len(scoring.diagnosis_scores) == 2
        assert scoring.diagnosis_scores[0].diagnosis_term == "Low back pain"
        assert scoring.diagnosis_scores[0].score == 2
        assert scoring.diagnosis_scores[1].score == -1

    def test_handles_missing_fields(self):
        """Missing optional fields default gracefully."""
        details = {
            "scores": [{"diagnosis": "Test"}],
        }

        scoring = scoring_from_stored(details)
        assert scoring.pat_id == "Unknown"
        ds = scoring.diagnosis_scores[0]
        assert ds.score == 0
        assert ds.judgement == ""
        assert ds.guidelines_followed == []
        assert ds.missing_care_opportunities == []


# ── Test: evaluate_extractor_from_db ────────────────────────────────


class TestExtractorFromDB:

    @pytest.mark.asyncio
    async def test_computes_per_category_metrics(self):
        """Should compute P/R/F1 from stored categories vs rules."""
        import pytest_asyncio
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from src.models.base import Base
        from src.models.patient import ClinicalEntry, Patient

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with factory() as session:
            # Create a patient and entries with known categories
            patient = Patient(pat_id="TEST1")
            session.add(patient)
            await session.flush()

            # "Osteoarthritis" should map to "diagnosis" by rules
            e1 = ClinicalEntry(
                patient_id=patient.id,
                index_date=date(2024, 1, 1),
                cons_date=date(2024, 1, 1),
                concept_id="396275006",
                term="Osteoarthritis",
                concept_display="Osteoarthritis",
                category="diagnosis",  # matches rules
            )
            # "X-ray" should map to "investigation" by rules
            e2 = ClinicalEntry(
                patient_id=patient.id,
                index_date=date(2024, 1, 1),
                cons_date=date(2024, 1, 1),
                concept_id="168537006",
                term="X-ray of knee",
                concept_display="X-ray of knee",
                category="investigation",  # matches rules
            )
            session.add_all([e1, e2])
            await session.flush()

            result = await evaluate_extractor_from_db(session)

            assert result["total_concepts"] >= 1
            assert "category_distribution" in result
            assert "per_category" in result

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero(self):
        """Empty database should return zero metrics."""
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from src.models.base import Base

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with factory() as session:
            result = await evaluate_extractor_from_db(session)
            assert result["total_concepts"] == 0
            assert result["rule_match_rate"] == 0.0

        await engine.dispose()


# ── Test: evaluate_retrieval_ir ──────────────────────────────────────


class TestRetrieverIR:

    @pytest.mark.asyncio
    async def test_all_relevant_gives_perfect_scores(self):
        """All guidelines rated ≥ threshold → precision=1, ndcg=1, mrr=1."""
        provider = MockAIProvider("Relevance: 5")
        retrieval = RetrievalResult(
            pat_id="P1",
            diagnosis_guidelines=[
                DiagnosisGuidelines(
                    diagnosis_term="Back pain",
                    concept_id="161891005",
                    index_date="2024-01-01",
                    guidelines=[
                        GuidelineMatch(
                            guideline_id="G1", title="NICE NG59",
                            source="NICE", url="", clean_text="Manage back pain...",
                            score=0.9, rank=1, matched_query="back pain",
                        ),
                    ],
                ),
            ],
            total_diagnoses=1,
            total_guidelines=1,
        )

        result = await evaluate_retrieval_ir(retrieval, provider)
        assert result["total_diagnoses"] == 1
        assert result["mean_precision_at_k"] == pytest.approx(1.0)
        assert result["mean_ndcg"] == pytest.approx(1.0)
        assert result["mean_mrr"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_irrelevant_gives_zero(self):
        """All guidelines rated below threshold → precision=0, mrr=0."""
        provider = MockAIProvider("Relevance: 1")
        retrieval = RetrievalResult(
            pat_id="P1",
            diagnosis_guidelines=[
                DiagnosisGuidelines(
                    diagnosis_term="Back pain",
                    concept_id="161891005",
                    index_date="2024-01-01",
                    guidelines=[
                        GuidelineMatch(
                            guideline_id="G1", title="Unrelated",
                            source="NICE", url="", clean_text="...",
                            score=0.5, rank=1, matched_query="q",
                        ),
                    ],
                ),
            ],
            total_diagnoses=1,
            total_guidelines=1,
        )

        result = await evaluate_retrieval_ir(retrieval, provider)
        assert result["mean_precision_at_k"] == pytest.approx(0.0)
        assert result["mean_mrr"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Empty retrieval should return zero metrics."""
        provider = MockAIProvider("")
        retrieval = RetrievalResult(pat_id="P1")

        result = await evaluate_retrieval_ir(retrieval, provider)
        assert result["total_diagnoses"] == 0
