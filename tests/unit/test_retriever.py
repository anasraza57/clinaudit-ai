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
    GuidelineEvidenceFinder,
    _diagnosis_topics,
    _title_is_excluded,
    _title_topics,
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


class TestGuidelineEvidenceFinder:
    def test_retrieve_single_diagnosis(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = GuidelineEvidenceFinder(
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
        agent = GuidelineEvidenceFinder(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        agent.retrieve(single_query_result)

        # encode_batch is called once per diagnosis (batches all queries together)
        assert mock_embedder.encode_batch.call_count == 1

    def test_retrieve_searches_for_each_query(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = GuidelineEvidenceFinder(
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
        agent = GuidelineEvidenceFinder(
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

        agent = GuidelineEvidenceFinder(embedder=mock_embedder, vector_store=store, top_k=5)
        result = agent.retrieve(qr)

        dg = result.diagnosis_guidelines[0]
        assert len(dg.guidelines) == 1
        assert dg.guidelines[0].score == 0.2  # Better score kept

    def test_retrieve_multi_diagnosis(
        self, mock_embedder, mock_vector_store, multi_diagnosis_result
    ):
        agent = GuidelineEvidenceFinder(
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
        agent = GuidelineEvidenceFinder(
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

        agent = GuidelineEvidenceFinder(embedder=mock_embedder, vector_store=store, top_k=3)
        result = agent.retrieve(qr)

        assert len(result.diagnosis_guidelines[0].guidelines) == 3

    def test_guidelines_have_ranks(
        self, mock_embedder, mock_vector_store, single_query_result
    ):
        agent = GuidelineEvidenceFinder(
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
        agent = GuidelineEvidenceFinder(
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

        agent = GuidelineEvidenceFinder(
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

        agent = GuidelineEvidenceFinder(
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


# ── Relevance filter helper tests ────────────────────────────────────


class TestRelevanceHelpers:
    """Tests for the module-level relevance filtering functions."""

    def test_diagnosis_topics_back_pain(self):
        topics = _diagnosis_topics("Low back pain")
        assert "spine" in topics
        assert "pain" in topics

    def test_diagnosis_topics_carpal_tunnel(self):
        topics = _diagnosis_topics("Carpal tunnel syndrome")
        assert "hand_wrist" in topics

    def test_diagnosis_topics_knee(self):
        topics = _diagnosis_topics("Osteoarthritis of knee")
        assert "knee" in topics
        assert "osteoarthritis" in topics

    def test_diagnosis_topics_unknown_condition(self):
        topics = _diagnosis_topics("Xylophonia syndrome")
        assert topics == set()

    def test_title_topics_matches(self):
        topics = _title_topics("Low back pain and sciatica: management")
        assert "spine" in topics
        assert "pain" in topics

    def test_title_is_excluded_cancer(self):
        assert _title_is_excluded("Lorlatinib for untreated ALK-positive non-small-cell lung cancer")

    def test_title_is_excluded_diabetes(self):
        assert _title_is_excluded("Diabetic foot problems: prevention and management")

    def test_title_is_excluded_chest_pain(self):
        assert _title_is_excluded("Recent-onset chest pain of suspected cardiac origin")

    def test_title_not_excluded_msk(self):
        assert not _title_is_excluded("Osteoarthritis: care and management in adults")

    def test_title_not_excluded_back_pain(self):
        assert not _title_is_excluded("Low back pain and sciatica in over 16s")


# ── Relevance filter integration tests ───────────────────────────────


class TestRelevanceFiltering:
    """Tests for the _filter_irrelevant method on GuidelineEvidenceFinder."""

    @pytest.fixture()
    def finder(self, mock_embedder, mock_vector_store):
        return GuidelineEvidenceFinder(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=5,
        )

    def _make_match(self, title: str, score: float = 0.3) -> GuidelineMatch:
        return GuidelineMatch(
            guideline_id=f"g-{hash(title)}",
            title=title,
            source="nice",
            url="",
            clean_text=f"Guideline text for {title}",
            score=score,
            rank=0,
            matched_query="test",
        )

    def test_filters_excluded_title_terms(self, finder):
        """Guidelines from clearly irrelevant specialties are removed."""
        matches = [
            self._make_match("Low back pain and sciatica", 0.2),
            self._make_match("Lorlatinib for non-small-cell lung cancer", 0.3),
            self._make_match("Diabetic foot problems", 0.4),
        ]
        filtered = finder._filter_irrelevant("Low back pain", matches)
        titles = [m.title for m in filtered]
        assert "Low back pain and sciatica" in titles
        assert "Lorlatinib for non-small-cell lung cancer" not in titles
        assert "Diabetic foot problems" not in titles

    def test_filters_topic_mismatch(self, finder):
        """A knee guideline should be filtered for a shoulder diagnosis."""
        matches = [
            self._make_match("Shoulder impingement management", 0.2),
            self._make_match("Knee osteoarthritis treatment", 0.3),
        ]
        filtered = finder._filter_irrelevant("Shoulder pain", matches)
        titles = [m.title for m in filtered]
        assert "Shoulder impingement management" in titles
        assert "Knee osteoarthritis treatment" not in titles

    def test_filters_high_distance(self, finder):
        """Guidelines with L2 distance above threshold are removed."""
        matches = [
            self._make_match("Back pain management", 0.3),
            self._make_match("Spine rehabilitation guidelines", 1.5),  # above default 1.2
        ]
        filtered = finder._filter_irrelevant("Low back pain", matches)
        assert len(filtered) == 1
        assert filtered[0].title == "Back pain management"

    def test_fallback_when_all_filtered(self, finder):
        """If all guidelines are filtered, best match is kept as fallback."""
        matches = [
            self._make_match("Lung cancer screening", 0.5),
            self._make_match("Breast cancer management", 0.7),
        ]
        filtered = finder._filter_irrelevant("Carpal tunnel syndrome", matches)
        # All are cancer → excluded, but fallback keeps best match
        assert len(filtered) == 1
        assert filtered[0].title == "Lung cancer screening"

    def test_unknown_diagnosis_no_topic_filtering(self, finder):
        """Rare/novel diagnoses with no topic match skip topic filtering."""
        matches = [
            self._make_match("Chest pain cardiac assessment", 0.3),
            self._make_match("Generic MSK management guide", 0.4),
        ]
        # "Xylophonia syndrome" has no topic tags, so topic filter is skipped.
        # But "chest pain" is in the exclude list, so it gets filtered.
        filtered = finder._filter_irrelevant("Xylophonia syndrome", matches)
        titles = [m.title for m in filtered]
        assert "Chest pain cardiac assessment" not in titles
        assert "Generic MSK management guide" in titles

    def test_relevant_guidelines_pass_through(self, finder):
        """Topically relevant guidelines are not filtered."""
        matches = [
            self._make_match("Osteoarthritis: care and management", 0.2),
            self._make_match("Joint replacement referral criteria", 0.3),
        ]
        filtered = finder._filter_irrelevant("Osteoarthritis of knee", matches)
        assert len(filtered) == 2

    def test_carpal_tunnel_vs_chest_pain(self, finder):
        """Regression: carpal tunnel should NOT match chest pain guidelines."""
        matches = [
            self._make_match("Recent-onset chest pain of suspected cardiac origin", 0.4),
            self._make_match("Carpal tunnel syndrome: management", 0.5),
        ]
        filtered = finder._filter_irrelevant("Carpal tunnel syndrome", matches)
        titles = [m.title for m in filtered]
        assert "Recent-onset chest pain of suspected cardiac origin" not in titles
        assert "Carpal tunnel syndrome: management" in titles

    def test_filtering_integrated_in_retrieve(self, mock_embedder):
        """End-to-end: irrelevant guidelines are filtered during retrieval."""
        store = MagicMock()
        store.search.return_value = [
            {"id": "g-1", "title": "Low back pain and sciatica", "source": "nice",
             "url": "", "clean_text": "Relevant guideline.", "score": 0.2},
            {"id": "g-2", "title": "Breast cancer screening programme", "source": "nice",
             "url": "", "clean_text": "Irrelevant guideline.", "score": 0.3},
            {"id": "g-3", "title": "Diabetic foot problems", "source": "nice",
             "url": "", "clean_text": "Irrelevant guideline.", "score": 0.4},
        ]

        qr = QueryResult(
            pat_id="pat-filter",
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

        agent = GuidelineEvidenceFinder(embedder=mock_embedder, vector_store=store, top_k=5)
        result = agent.retrieve(qr)

        dg = result.diagnosis_guidelines[0]
        titles = [g.title for g in dg.guidelines]
        assert "Low back pain and sciatica" in titles
        assert "Breast cancer screening programme" not in titles
        assert "Diabetic foot problems" not in titles
