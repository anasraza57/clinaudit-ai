"""
Tests for the Retriever Agent.

Uses mock embedder and vector store to test retrieval logic
without loading the real PubMedBERT model or FAISS index.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.query import DiagnosisQueries, QueryResult
from src.agents.retriever import (
    DiagnosisGuidelines,
    GuidelineMatch,
    RetrievalResult,
    RetrieverAgent,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_embedder():
    """Mock embedder that returns random 768-dim vectors."""
    embedder = MagicMock()
    embedder.is_loaded = True
    embedder.encode.side_effect = lambda text: np.random.rand(768).astype(np.float32)
    return embedder


@pytest.fixture()
def mock_vector_store():
    """Mock vector store that returns fake guideline results."""
    store = MagicMock()
    store.is_loaded = True

    def fake_search(query_embedding, top_k=5):
        return [
            {
                "id": f"guide-{i}",
                "title": f"Guideline {i}",
                "source": "nice",
                "url": f"https://nice.org.uk/guidance/{i}",
                "clean_text": f"This is guideline text for result {i}.",
                "score": float(i) * 0.1,
                "rank": i + 1,
            }
            for i in range(min(top_k, 3))  # Return up to 3 results
        ]

    store.search.side_effect = fake_search
    return store


@pytest.fixture()
def single_query_result():
    """QueryResult with one diagnosis and its queries."""
    return QueryResult(
        pat_id="pat-001",
        diagnosis_queries=[
            DiagnosisQueries(
                diagnosis_term="Low back pain",
                concept_id="279039007",
                index_date="2024-01-15",
                queries=[
                    "NICE guidelines for low back pain management",
                    "low back pain treatment options",
                    "low back pain referral criteria",
                ],
                source="template",
            ),
        ],
        total_diagnoses=1,
        total_queries=3,
    )


@pytest.fixture()
def multi_diagnosis_result():
    """QueryResult with two diagnoses."""
    return QueryResult(
        pat_id="pat-002",
        diagnosis_queries=[
            DiagnosisQueries(
                diagnosis_term="Low back pain",
                concept_id="279039007",
                index_date="2024-01-15",
                queries=["low back pain guidelines"],
                source="template",
            ),
            DiagnosisQueries(
                diagnosis_term="Osteoarthritis of knee",
                concept_id="239873007",
                index_date="2024-06-01",
                queries=["osteoarthritis management guidelines"],
                source="template",
            ),
        ],
        total_diagnoses=2,
        total_queries=2,
    )


@pytest.fixture()
def empty_query_result():
    """QueryResult with no diagnoses."""
    return QueryResult(
        pat_id="pat-003",
        diagnosis_queries=[],
        total_diagnoses=0,
        total_queries=0,
    )


# ── Retriever Agent tests ────────────────────────────────────────────


class TestRetrieverAgent:
    def test_retrieve_single_diagnosis(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=5,
        )
        result = agent.retrieve(single_query_result)

        assert isinstance(result, RetrievalResult)
        assert result.pat_id == "pat-001"
        assert result.total_diagnoses == 1
        assert result.total_guidelines >= 1

    def test_retrieve_embeds_each_query(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        agent.retrieve(single_query_result)

        # encode_batch is called once per diagnosis (batches all queries together)
        assert mock_embedder.encode_batch.call_count == 1

    def test_retrieve_searches_for_each_query(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        agent.retrieve(single_query_result)

        # Should search FAISS once per query
        assert mock_vector_store.search.call_count == 3

    def test_retrieve_deduplicates_guidelines(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        """Multiple queries returning the same guideline should be deduped."""
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=5,
        )
        result = agent.retrieve(single_query_result)

        dg = result.diagnosis_guidelines[0]
        # Mock returns guide-0, guide-1, guide-2 for each query
        # After dedup, should have at most 3 unique guidelines
        guideline_ids = [g.guideline_id for g in dg.guidelines]
        assert len(guideline_ids) == len(set(guideline_ids)), "Duplicates found!"

    def test_retrieve_keeps_best_score_on_dedup(self, mock_embedder):
        """When deduplicating, keep the result with the better score."""
        store = MagicMock()
        call_count = [0]

        def search_with_varying_scores(query_embedding, top_k=5):
            call_count[0] += 1
            # First query: guide-A with score 0.5
            # Second query: guide-A with score 0.2 (better!)
            if call_count[0] == 1:
                return [{"id": "guide-A", "title": "Guide A", "source": "nice",
                         "url": "", "clean_text": "text", "score": 0.5}]
            else:
                return [{"id": "guide-A", "title": "Guide A", "source": "nice",
                         "url": "", "clean_text": "text", "score": 0.2}]

        store.search.side_effect = search_with_varying_scores

        qr = QueryResult(
            pat_id="pat-X",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Test",
                    concept_id="1",
                    index_date="2024-01-01",
                    queries=["query 1", "query 2"],
                    source="default",
                ),
            ],
            total_diagnoses=1,
            total_queries=2,
        )

        agent = RetrieverAgent(embedder=mock_embedder, vector_store=store, top_k=5)
        result = agent.retrieve(qr)

        dg = result.diagnosis_guidelines[0]
        assert len(dg.guidelines) == 1
        assert dg.guidelines[0].score == 0.2  # Better score kept

    def test_retrieve_multi_diagnosis(
        self, mock_embedder, mock_vector_store, multi_diagnosis_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(multi_diagnosis_result)

        assert result.total_diagnoses == 2
        assert len(result.diagnosis_guidelines) == 2
        assert result.diagnosis_guidelines[0].diagnosis_term == "Low back pain"
        assert result.diagnosis_guidelines[1].diagnosis_term == "Osteoarthritis of knee"

    def test_retrieve_empty_queries(
        self, mock_embedder, mock_vector_store, empty_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(empty_query_result)

        assert result.total_diagnoses == 0
        assert result.total_guidelines == 0

    def test_retrieve_respects_top_k(self, mock_embedder):
        """Agent should limit results to top_k per diagnosis."""
        store = MagicMock()
        store.search.return_value = [
            {"id": f"g-{i}", "title": f"G{i}", "source": "nice",
             "url": "", "clean_text": f"text {i}", "score": float(i) * 0.1}
            for i in range(10)
        ]

        qr = QueryResult(
            pat_id="pat-X",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Test",
                    concept_id="1",
                    index_date="2024-01-01",
                    queries=["single query"],
                    source="default",
                ),
            ],
            total_diagnoses=1,
            total_queries=1,
        )

        agent = RetrieverAgent(embedder=mock_embedder, vector_store=store, top_k=3)
        result = agent.retrieve(qr)

        assert len(result.diagnosis_guidelines[0].guidelines) == 3

    def test_guidelines_have_ranks(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(single_query_result)

        dg = result.diagnosis_guidelines[0]
        ranks = [g.rank for g in dg.guidelines]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_summary_output(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(single_query_result)
        summary = result.summary()

        assert summary["pat_id"] == "pat-001"
        assert summary["total_diagnoses"] == 1
        assert "titles" in summary["diagnoses"][0]

    def test_duplicate_diagnosis_encodes_once(self, mock_embedder, mock_vector_store):
        """Same diagnosis term in 2 entries should only encode once."""
        qr = QueryResult(
            pat_id="pat-dedup",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Finger pain",
                    concept_id="1",
                    index_date="2024-01-15",
                    queries=["finger pain guidelines"],
                    source="llm",
                ),
                DiagnosisQueries(
                    diagnosis_term="Finger pain",
                    concept_id="1",
                    index_date="2024-06-01",
                    queries=["finger pain guidelines"],
                    source="llm",
                ),
            ],
            total_diagnoses=2,
            total_queries=2,
        )

        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(qr)

        # 2 DiagnosisGuidelines produced (one per entry)
        assert result.total_diagnoses == 2
        # But encode_batch called only once (cached for second occurrence)
        assert mock_embedder.encode_batch.call_count == 1
        # Both should have the same guidelines
        assert (
            result.diagnosis_guidelines[0].guidelines
            is result.diagnosis_guidelines[1].guidelines
        )

    def test_duplicate_diagnosis_same_date_skipped(self, mock_embedder, mock_vector_store):
        """Same (diagnosis_term, index_date) should produce only one entry."""
        qr = QueryResult(
            pat_id="pat-dedup2",
            diagnosis_queries=[
                DiagnosisQueries(
                    diagnosis_term="Finger pain",
                    concept_id="1",
                    index_date="2024-03-07",
                    queries=["finger pain guidelines"],
                    source="llm",
                ),
                DiagnosisQueries(
                    diagnosis_term="Finger pain",
                    concept_id="1",
                    index_date="2024-03-07",
                    queries=["finger pain guidelines"],
                    source="llm",
                ),
            ],
            total_diagnoses=2,
            total_queries=2,
        )

        agent = RetrieverAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        result = agent.retrieve(qr)

        # Only 1 unique (term, date) → only 1 DiagnosisGuidelines entry
        assert result.total_diagnoses == 1
        assert len(result.diagnosis_guidelines) == 1
        # encode_batch called only once
        assert mock_embedder.encode_batch.call_count == 1


# ── Data class tests ──────────────────────────────────────────────────


class TestGuidelineMatch:
    def test_creation(self):
        gm = GuidelineMatch(
            guideline_id="abc",
            title="Test Guideline",
            source="nice",
            url="https://example.com",
            clean_text="Guideline content here.",
            score=0.15,
            rank=1,
            matched_query="test query",
        )
        assert gm.guideline_id == "abc"
        assert gm.score == 0.15
        assert gm.rank == 1


class TestDiagnosisGuidelines:
    def test_guideline_texts(self):
        dg = DiagnosisGuidelines(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            guidelines=[
                GuidelineMatch(
                    guideline_id="a", title="Guide A", source="nice",
                    url="", clean_text="Text A", score=0.1, rank=2,
                    matched_query="q",
                ),
                GuidelineMatch(
                    guideline_id="b", title="Guide B", source="nice",
                    url="", clean_text="Text B", score=0.05, rank=1,
                    matched_query="q",
                ),
            ],
        )
        # Should be sorted by rank
        assert dg.guideline_texts == ["Text B", "Text A"]
        assert dg.guideline_titles == ["Guide B", "Guide A"]

    def test_empty_guidelines(self):
        dg = DiagnosisGuidelines(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
        )
        assert dg.guideline_texts == []
        assert dg.guideline_titles == []


class TestRetrievalResult:
    def test_empty_result(self):
        rr = RetrievalResult(pat_id="pat-000")
        summary = rr.summary()
        assert summary["total_diagnoses"] == 0
        assert summary["total_guidelines"] == 0
        assert summary["diagnoses"] == []
