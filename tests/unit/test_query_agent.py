"""
Tests for the Query Agent.

Verifies that the agent correctly generates search queries from
extracted diagnoses using templates, LLM, and default fallbacks.
"""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from src.agents.extractor import (
    CategorisedEntry,
    ExtractionResult,
    PatientEpisode,
)
from src.agents.query import (
    DiagnosisQueries,
    QueryAgent,
    QueryResult,
    _find_template,
    generate_default_queries,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def single_episode_extraction():
    """Extraction with one episode containing one diagnosis."""
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
                        concept_id="183545006",
                        term="Orthopaedic referral",
                        concept_display="Referral to orthopaedic service",
                        cons_date=date(2024, 2, 1),
                        category="referral",
                    ),
                ],
            ),
        ],
        total_entries=2,
        total_diagnoses=1,
    )


@pytest.fixture()
def multi_episode_extraction():
    """Extraction with two episodes and different diagnoses."""
    return ExtractionResult(
        pat_id="pat-002",
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
                ],
            ),
            PatientEpisode(
                index_date=date(2024, 6, 1),
                entries=[
                    CategorisedEntry(
                        concept_id="239873007",
                        term="Osteoarthritis of knee",
                        concept_display="Osteoarthritis of knee",
                        cons_date=date(2024, 6, 1),
                        category="diagnosis",
                    ),
                    CategorisedEntry(
                        concept_id="12345",
                        term="Steroid injection",
                        concept_display="Injection of steroid into knee joint",
                        cons_date=date(2024, 6, 15),
                        category="treatment",
                    ),
                ],
            ),
        ],
        total_entries=3,
        total_diagnoses=2,
    )


@pytest.fixture()
def no_diagnosis_extraction():
    """Extraction with entries but no diagnoses."""
    return ExtractionResult(
        pat_id="pat-003",
        episodes=[
            PatientEpisode(
                index_date=date(2024, 1, 1),
                entries=[
                    CategorisedEntry(
                        concept_id="183545006",
                        term="Telephone consultation",
                        concept_display="Telephone consultation",
                        cons_date=date(2024, 1, 1),
                        category="administrative",
                    ),
                ],
            ),
        ],
        total_entries=1,
        total_diagnoses=0,
    )


@pytest.fixture()
def unusual_diagnosis_extraction():
    """Extraction with a diagnosis that has no template."""
    return ExtractionResult(
        pat_id="pat-004",
        episodes=[
            PatientEpisode(
                index_date=date(2024, 3, 1),
                entries=[
                    CategorisedEntry(
                        concept_id="999999",
                        term="Acquired hallux valgus",
                        concept_display="Acquired hallux valgus",
                        cons_date=date(2024, 3, 1),
                        category="diagnosis",
                    ),
                ],
            ),
        ],
        total_entries=1,
        total_diagnoses=1,
    )


# ── Template lookup tests ────────────────────────────────────────────


class TestTemplateMatching:
    """Test the rule-based template query generation."""

    def test_exact_match(self):
        result = _find_template("Low back pain")
        assert result is not None
        assert len(result) == 3
        assert any("low back pain" in q.lower() for q in result)

    def test_exact_match_case_insensitive(self):
        result = _find_template("LOW BACK PAIN")
        assert result is not None

    def test_substring_match(self):
        # "osteoarthritis of knee" should match "osteoarthritis" template
        result = _find_template("Osteoarthritis of knee")
        assert result is not None
        assert any("osteoarthritis" in q.lower() for q in result)

    def test_no_match(self):
        result = _find_template("Acquired hallux valgus")
        assert result is None

    @pytest.mark.parametrize("term", [
        "Low back pain",
        "Osteoarthritis of knee",
        "Carpal tunnel syndrome",
        "Plantar fasciitis",
        "Gout",
        "Fibromyalgia",
        "Osteoporosis",
        "Shoulder pain",
        "Hip pain",
        "Sciatica",
    ])
    def test_common_msk_conditions_have_templates(self, term):
        result = _find_template(term)
        assert result is not None, f"No template for common MSK diagnosis: {term}"
        assert len(result) >= 1

    def test_template_queries_are_strings(self):
        result = _find_template("Low back pain")
        assert all(isinstance(q, str) for q in result)
        assert all(len(q) > 10 for q in result)  # Not empty/trivial


# ── Default query tests ──────────────────────────────────────────────


class TestDefaultQueries:
    def test_generates_correct_count(self):
        queries = generate_default_queries("Hallux valgus", max_queries=3)
        assert len(queries) == 3

    def test_respects_max_queries(self):
        queries = generate_default_queries("Hallux valgus", max_queries=1)
        assert len(queries) == 1

    def test_includes_diagnosis_term(self):
        queries = generate_default_queries("Hallux valgus")
        assert all("Hallux valgus" in q for q in queries)

    def test_includes_nice_guidelines(self):
        queries = generate_default_queries("Some condition")
        assert any("NICE" in q for q in queries)


# ── QueryAgent tests (no LLM) ────────────────────────────────────────


class TestQueryAgentNoLLM:
    """Test the Query Agent without an LLM provider."""

    @pytest.mark.asyncio
    async def test_generates_queries_for_template_diagnosis(
        self, single_episode_extraction
    ):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(single_episode_extraction)

        assert isinstance(result, QueryResult)
        assert result.pat_id == "pat-001"
        assert result.total_diagnoses == 1
        assert result.total_queries >= 1
        assert result.diagnosis_queries[0].source == "template"

    @pytest.mark.asyncio
    async def test_generates_queries_for_multiple_episodes(
        self, multi_episode_extraction
    ):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(multi_episode_extraction)

        assert result.total_diagnoses == 2
        assert len(result.diagnosis_queries) == 2
        # Low back pain should use template
        assert result.diagnosis_queries[0].diagnosis_term == "Low back pain"
        assert result.diagnosis_queries[0].source == "template"
        # Osteoarthritis of knee should also use template (substring match)
        assert result.diagnosis_queries[1].diagnosis_term == "Osteoarthritis of knee"
        assert result.diagnosis_queries[1].source == "template"

    @pytest.mark.asyncio
    async def test_no_diagnoses_produces_empty_result(
        self, no_diagnosis_extraction
    ):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(no_diagnosis_extraction)

        assert result.total_diagnoses == 0
        assert result.total_queries == 0
        assert len(result.diagnosis_queries) == 0

    @pytest.mark.asyncio
    async def test_unusual_diagnosis_uses_defaults(
        self, unusual_diagnosis_extraction
    ):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(unusual_diagnosis_extraction)

        assert result.total_diagnoses == 1
        dq = result.diagnosis_queries[0]
        assert dq.source == "default"
        assert len(dq.queries) >= 1
        assert "Acquired hallux valgus" in dq.queries[0]

    @pytest.mark.asyncio
    async def test_summary_output(self, single_episode_extraction):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(single_episode_extraction)
        summary = result.summary()

        assert summary["pat_id"] == "pat-001"
        assert summary["total_diagnoses"] == 1
        assert len(summary["diagnoses"]) == 1
        assert "queries" in summary["diagnoses"][0]
        assert summary["diagnoses"][0]["source"] == "template"

    @pytest.mark.asyncio
    async def test_all_queries_helper(self, multi_episode_extraction):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(multi_episode_extraction)
        all_q = result.all_queries()

        assert len(all_q) == result.total_queries
        # Each tuple is (diagnosis_term, query_text)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in all_q)

    @pytest.mark.asyncio
    async def test_index_date_preserved(self, multi_episode_extraction):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(multi_episode_extraction)

        assert result.diagnosis_queries[0].index_date == "2024-01-15"
        assert result.diagnosis_queries[1].index_date == "2024-06-01"

    @pytest.mark.asyncio
    async def test_concept_id_preserved(self, single_episode_extraction):
        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(single_episode_extraction)

        assert result.diagnosis_queries[0].concept_id == "279039007"


# ── QueryAgent tests (with mock LLM) ─────────────────────────────────


class TestQueryAgentWithLLM:
    """Test the Query Agent with a mock LLM provider."""

    @pytest.mark.asyncio
    async def test_template_preferred_over_llm(self, single_episode_extraction):
        """Template matches should not call the LLM."""
        mock_provider = AsyncMock()
        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(single_episode_extraction)

        # Low back pain has a template — LLM should NOT be called
        assert result.diagnosis_queries[0].source == "template"
        mock_provider.chat_simple.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_used_for_unmatched_diagnosis(
        self, unusual_diagnosis_extraction
    ):
        """Diagnoses without templates should use the LLM."""
        mock_provider = AsyncMock()
        mock_provider.chat_simple.return_value = (
            "NICE guidelines for hallux valgus surgical management\n"
            "bunion deformity assessment and conservative treatment\n"
            "hallux valgus referral criteria for orthopaedic surgery"
        )

        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(unusual_diagnosis_extraction)

        dq = result.diagnosis_queries[0]
        assert dq.source == "llm"
        assert len(dq.queries) == 3
        mock_provider.chat_simple.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_default(
        self, unusual_diagnosis_extraction
    ):
        """If the LLM fails, should fall back to default queries."""
        mock_provider = AsyncMock()
        mock_provider.chat_simple.side_effect = Exception("API error")

        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(unusual_diagnosis_extraction)

        dq = result.diagnosis_queries[0]
        assert dq.source == "default"
        assert len(dq.queries) >= 1

    @pytest.mark.asyncio
    async def test_llm_empty_response_falls_back_to_default(
        self, unusual_diagnosis_extraction
    ):
        """If the LLM returns empty text, should fall back to default."""
        mock_provider = AsyncMock()
        mock_provider.chat_simple.return_value = ""

        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(unusual_diagnosis_extraction)

        dq = result.diagnosis_queries[0]
        assert dq.source == "default"

    @pytest.mark.asyncio
    async def test_llm_respects_max_queries(self, unusual_diagnosis_extraction):
        """LLM results should be capped at max_queries_per_diagnosis."""
        mock_provider = AsyncMock()
        mock_provider.chat_simple.return_value = (
            "query one\nquery two\nquery three\nquery four\nquery five"
        )

        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(unusual_diagnosis_extraction)

        dq = result.diagnosis_queries[0]
        assert len(dq.queries) <= 3  # max_queries_per_diagnosis default is 3

    @pytest.mark.asyncio
    async def test_duplicate_diagnosis_across_episodes_calls_llm_once(self):
        """Same diagnosis term in 2 episodes should only call LLM once."""
        extraction = ExtractionResult(
            pat_id="pat-dedup",
            episodes=[
                PatientEpisode(
                    index_date=date(2024, 1, 15),
                    entries=[
                        CategorisedEntry(
                            concept_id="999999",
                            term="Acquired hallux valgus",
                            concept_display="Acquired hallux valgus",
                            cons_date=date(2024, 1, 15),
                            category="diagnosis",
                        ),
                    ],
                ),
                PatientEpisode(
                    index_date=date(2024, 6, 1),
                    entries=[
                        CategorisedEntry(
                            concept_id="999999",
                            term="Acquired hallux valgus",
                            concept_display="Acquired hallux valgus",
                            cons_date=date(2024, 6, 1),
                            category="diagnosis",
                        ),
                    ],
                ),
            ],
            total_entries=2,
            total_diagnoses=2,
        )

        mock_provider = AsyncMock()
        mock_provider.chat_simple.return_value = (
            "hallux valgus guidelines\nbunion management\nhallux referral"
        )

        agent = QueryAgent(ai_provider=mock_provider)
        result = await agent.generate_queries(extraction)

        # 2 DiagnosisQueries produced (one per episode)
        assert result.total_diagnoses == 2
        # But LLM called only once (cached for second occurrence)
        assert mock_provider.chat_simple.call_count == 1
        # Both should have identical queries
        assert result.diagnosis_queries[0].queries == result.diagnosis_queries[1].queries
        # But different index_dates
        assert result.diagnosis_queries[0].index_date == "2024-01-15"
        assert result.diagnosis_queries[1].index_date == "2024-06-01"

    @pytest.mark.asyncio
    async def test_duplicate_diagnosis_same_episode_skipped(self):
        """Same diagnosis term + same index_date should produce only one entry."""
        extraction = ExtractionResult(
            pat_id="pat-dedup2",
            episodes=[
                PatientEpisode(
                    index_date=date(2024, 3, 7),
                    entries=[
                        CategorisedEntry(
                            concept_id="1",
                            term="Finger pain",
                            concept_display="Finger pain",
                            cons_date=date(2024, 3, 7),
                            category="diagnosis",
                        ),
                        CategorisedEntry(
                            concept_id="1",
                            term="Finger pain",
                            concept_display="Finger pain",
                            cons_date=date(2024, 3, 7),
                            category="diagnosis",
                        ),
                    ],
                ),
            ],
            total_entries=2,
            total_diagnoses=2,
        )

        agent = QueryAgent(ai_provider=None)
        result = await agent.generate_queries(extraction)

        # Only 1 unique (term, date) → only 1 DiagnosisQueries entry
        assert result.total_diagnoses == 1
        assert len(result.diagnosis_queries) == 1
        assert result.diagnosis_queries[0].diagnosis_term == "Finger pain"


# ── DiagnosisQueries / QueryResult dataclass tests ────────────────────


class TestDataClasses:
    def test_diagnosis_queries_defaults(self):
        dq = DiagnosisQueries(
            diagnosis_term="Test",
            concept_id="123",
            index_date="2024-01-01",
        )
        assert dq.queries == []
        assert dq.source == ""

    def test_query_result_summary_empty(self):
        qr = QueryResult(pat_id="pat-000")
        summary = qr.summary()
        assert summary["total_diagnoses"] == 0
        assert summary["total_queries"] == 0
        assert summary["diagnoses"] == []

    def test_query_result_all_queries_empty(self):
        qr = QueryResult(pat_id="pat-000")
        assert qr.all_queries() == []
