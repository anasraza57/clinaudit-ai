"""
Tests for the Audit Pipeline orchestrator.

Uses mock agents and a mock DB session to test pipeline logic
without needing a real database, LLM, or FAISS index.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.extractor import (
    CategorisedEntry,
    ExtractionResult,
    PatientEpisode,
)
from src.agents.query import DiagnosisQueries, QueryResult
from src.agents.retriever import (
    DiagnosisGuidelines,
    GuidelineMatch,
    RetrievalResult,
)
from src.agents.scorer import DiagnosisScore, ScoringResult
from src.services.pipeline import AuditPipeline, PipelineResult


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_ai_provider():
    provider = AsyncMock()
    provider.provider_name = "mock"
    return provider


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.is_loaded = True
    return embedder


@pytest.fixture()
def mock_vector_store():
    store = MagicMock()
    store.is_loaded = True
    return store


@pytest.fixture()
def pipeline(mock_ai_provider, mock_embedder, mock_vector_store):
    return AuditPipeline(
        ai_provider=mock_ai_provider,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


@pytest.fixture()
def sample_extraction():
    return ExtractionResult(
        pat_id="pat-001",
        episodes=[
            PatientEpisode(
                index_date=date(2024, 1, 15),
                entries=[
                    CategorisedEntry(
                        concept_id="279039007",
                        term="Low back pain",
                        concept_display="Low back pain",
                        cons_date=date(2024, 1, 15),
                        category="diagnosis",
                    ),
                    CategorisedEntry(
                        concept_id="12345",
                        term="Ibuprofen",
                        concept_display="Ibuprofen",
                        cons_date=date(2024, 1, 15),
                        category="treatment",
                    ),
                ],
            )
        ],
        total_entries=2,
        total_diagnoses=1,
    )


@pytest.fixture()
def sample_query_result():
    return QueryResult(
        pat_id="pat-001",
        diagnosis_queries=[
            DiagnosisQueries(
                diagnosis_term="Low back pain",
                concept_id="279039007",
                index_date="2024-01-15",
                queries=["low back pain guidelines"],
                source="template",
            ),
        ],
        total_diagnoses=1,
        total_queries=1,
    )


@pytest.fixture()
def sample_retrieval():
    return RetrievalResult(
        pat_id="pat-001",
        diagnosis_guidelines=[
            DiagnosisGuidelines(
                diagnosis_term="Low back pain",
                concept_id="279039007",
                index_date="2024-01-15",
                guidelines=[
                    GuidelineMatch(
                        guideline_id="ng59-1",
                        title="Low back pain guideline",
                        source="nice",
                        url="",
                        clean_text="Exercise therapy recommended.",
                        score=0.1,
                        rank=1,
                        matched_query="q",
                    ),
                ],
            )
        ],
        total_diagnoses=1,
        total_guidelines=1,
    )


@pytest.fixture()
def sample_scoring():
    return ScoringResult(
        pat_id="pat-001",
        diagnosis_scores=[
            DiagnosisScore(
                diagnosis_term="Low back pain",
                concept_id="279039007",
                index_date="2024-01-15",
                score=2,
                judgement="COMPLIANT",
                explanation="Good adherence.",
                confidence=0.85,
                cited_guideline_text="Exercise therapy recommended.",
                guidelines_followed=["Exercise"],
                guidelines_not_followed=[],
            ),
        ],
        total_diagnoses=1,
        compliant_count=1,
    )


@pytest.fixture()
def mock_session():
    """Mock async DB session."""
    session = AsyncMock()
    return session


# ── PipelineResult tests ─────────────────────────────────────────────


class TestPipelineResult:
    def test_success_when_scoring_present(self):
        pr = PipelineResult(
            pat_id="pat-001",
            scoring=ScoringResult(pat_id="pat-001"),
        )
        assert pr.success is True

    def test_failure_when_error(self):
        pr = PipelineResult(
            pat_id="pat-001",
            error="Something broke",
        )
        assert pr.success is False

    def test_failure_when_no_scoring(self):
        pr = PipelineResult(pat_id="pat-001")
        assert pr.success is False

    def test_summary_success(self, sample_scoring):
        pr = PipelineResult(
            pat_id="pat-001",
            scoring=sample_scoring,
            stage_reached="scoring",
        )
        summary = pr.summary()
        assert summary["pat_id"] == "pat-001"
        assert summary["success"] is True
        assert "scoring" in summary

    def test_summary_failure(self):
        pr = PipelineResult(
            pat_id="pat-001",
            error="No data",
            stage_reached="load",
        )
        summary = pr.summary()
        assert summary["success"] is False
        assert summary["error"] == "No data"


# ── AuditPipeline tests ─────────────────────────────────────────────


class TestAuditPipeline:
    def test_init(self, pipeline):
        assert pipeline.categories_loaded is False

    @pytest.mark.asyncio
    async def test_load_categories(self, pipeline):
        concepts = ["Low back pain", "Ibuprofen", "Physiotherapy referral"]
        await pipeline.load_categories(concepts)
        assert pipeline.categories_loaded is True

    @pytest.mark.asyncio
    async def test_run_single_success(
        self,
        pipeline,
        mock_session,
        sample_extraction,
        sample_query_result,
        sample_retrieval,
        sample_scoring,
    ):
        """Full pipeline: all 4 stages succeed."""
        # Mock _load_patient_entries
        entries = [
            {
                "concept_id": "279039007",
                "term": "Low back pain",
                "concept_display": "Low back pain",
                "index_date": date(2024, 1, 15),
                "cons_date": date(2024, 1, 15),
                "notes": None,
            },
        ]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)

        # Mock each agent
        pipeline._extractor.extract = MagicMock(return_value=sample_extraction)
        pipeline._query_agent.generate_queries = AsyncMock(return_value=sample_query_result)
        pipeline._retriever.retrieve = MagicMock(return_value=sample_retrieval)
        pipeline._scorer.score = AsyncMock(return_value=sample_scoring)

        # Mock _store_result
        pipeline._store_result = AsyncMock()

        # Pre-load categories
        await pipeline.load_categories(["Low back pain"])

        result = await pipeline.run_single(mock_session, "pat-001")

        assert result.success
        assert result.pat_id == "pat-001"
        assert result.stage_reached == "scoring"
        assert result.scoring.aggregate_score == 1.0

    @pytest.mark.asyncio
    async def test_run_single_no_entries(self, pipeline, mock_session):
        """Pipeline should fail gracefully when patient has no entries."""
        pipeline._load_patient_entries = AsyncMock(return_value=[])
        pipeline._store_result = AsyncMock()

        result = await pipeline.run_single(mock_session, "pat-missing")

        assert not result.success
        assert result.stage_reached == "load"
        assert "No clinical entries" in result.error

    @pytest.mark.asyncio
    async def test_run_single_no_diagnoses(
        self, pipeline, mock_session
    ):
        """Pipeline should fail when extractor finds no diagnoses."""
        entries = [
            {
                "concept_id": "12345",
                "term": "Ibuprofen",
                "concept_display": "Ibuprofen",
                "index_date": date(2024, 1, 15),
                "cons_date": date(2024, 1, 15),
                "notes": None,
            },
        ]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)
        pipeline._store_result = AsyncMock()

        # Extractor returns result with 0 diagnoses
        no_diag = ExtractionResult(
            pat_id="pat-001", episodes=[], total_entries=1, total_diagnoses=0
        )
        pipeline._extractor.extract = MagicMock(return_value=no_diag)

        await pipeline.load_categories(["Ibuprofen"])
        result = await pipeline.run_single(mock_session, "pat-001")

        assert not result.success
        assert "No diagnoses" in result.error

    @pytest.mark.asyncio
    async def test_run_single_scorer_error(
        self,
        pipeline,
        mock_session,
        sample_extraction,
        sample_query_result,
        sample_retrieval,
    ):
        """Pipeline should handle scorer failures gracefully."""
        entries = [
            {
                "concept_id": "279039007",
                "term": "Low back pain",
                "concept_display": "Low back pain",
                "index_date": date(2024, 1, 15),
                "cons_date": date(2024, 1, 15),
                "notes": None,
            },
        ]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)
        pipeline._store_result = AsyncMock()
        pipeline._extractor.extract = MagicMock(return_value=sample_extraction)
        pipeline._query_agent.generate_queries = AsyncMock(return_value=sample_query_result)
        pipeline._retriever.retrieve = MagicMock(return_value=sample_retrieval)
        pipeline._scorer.score = AsyncMock(side_effect=Exception("LLM down"))

        await pipeline.load_categories(["Low back pain"])
        result = await pipeline.run_single(mock_session, "pat-001")

        assert not result.success
        assert "LLM down" in result.error
        assert result.stage_reached == "retrieval"

    @pytest.mark.asyncio
    async def test_run_single_calls_store_result(
        self,
        pipeline,
        mock_session,
        sample_extraction,
        sample_query_result,
        sample_retrieval,
        sample_scoring,
    ):
        """Pipeline should always store result in DB."""
        entries = [{"concept_id": "1", "term": "T", "concept_display": "T",
                     "index_date": date(2024, 1, 1), "cons_date": date(2024, 1, 1),
                     "notes": None}]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)
        pipeline._store_result = AsyncMock()
        pipeline._extractor.extract = MagicMock(return_value=sample_extraction)
        pipeline._query_agent.generate_queries = AsyncMock(return_value=sample_query_result)
        pipeline._retriever.retrieve = MagicMock(return_value=sample_retrieval)
        pipeline._scorer.score = AsyncMock(return_value=sample_scoring)

        await pipeline.load_categories(["T"])
        await pipeline.run_single(mock_session, "pat-001")

        pipeline._store_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_passes_job_id(
        self,
        pipeline,
        mock_session,
        sample_extraction,
        sample_query_result,
        sample_retrieval,
        sample_scoring,
    ):
        """Job ID should be passed through to _store_result."""
        entries = [{"concept_id": "1", "term": "T", "concept_display": "T",
                     "index_date": date(2024, 1, 1), "cons_date": date(2024, 1, 1),
                     "notes": None}]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)
        pipeline._store_result = AsyncMock()
        pipeline._extractor.extract = MagicMock(return_value=sample_extraction)
        pipeline._query_agent.generate_queries = AsyncMock(return_value=sample_query_result)
        pipeline._retriever.retrieve = MagicMock(return_value=sample_retrieval)
        pipeline._scorer.score = AsyncMock(return_value=sample_scoring)

        await pipeline.load_categories(["T"])
        await pipeline.run_single(mock_session, "pat-001", job_id=42)

        call_args = pipeline._store_result.call_args
        assert call_args[0][3] == 42  # 4th positional arg is job_id

    @pytest.mark.asyncio
    async def test_pipeline_stages_called_in_order(
        self,
        pipeline,
        mock_session,
        sample_extraction,
        sample_query_result,
        sample_retrieval,
        sample_scoring,
    ):
        """Verify the 4 stages are called in sequence."""
        call_order = []

        entries = [{"concept_id": "1", "term": "T", "concept_display": "T",
                     "index_date": date(2024, 1, 1), "cons_date": date(2024, 1, 1),
                     "notes": None}]
        pipeline._load_patient_entries = AsyncMock(return_value=entries)
        pipeline._store_result = AsyncMock()

        def mock_extract(*a, **kw):
            call_order.append("extract")
            return sample_extraction

        async def mock_query(*a, **kw):
            call_order.append("query")
            return sample_query_result

        def mock_retrieve(*a, **kw):
            call_order.append("retrieve")
            return sample_retrieval

        async def mock_score(*a, **kw):
            call_order.append("score")
            return sample_scoring

        pipeline._extractor.extract = mock_extract
        pipeline._query_agent.generate_queries = mock_query
        pipeline._retriever.retrieve = mock_retrieve
        pipeline._scorer.score = mock_score

        await pipeline.load_categories(["T"])
        await pipeline.run_single(mock_session, "pat-001")

        assert call_order == ["extract", "query", "retrieve", "score"]
