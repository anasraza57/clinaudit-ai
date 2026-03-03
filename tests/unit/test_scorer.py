"""
Tests for the Scorer Agent.

Uses a mock AI provider to test scoring logic without calling a real LLM.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.extractor import CategorisedEntry, ExtractionResult, PatientEpisode
from src.agents.retriever import DiagnosisGuidelines, GuidelineMatch, RetrievalResult
from src.agents.scorer import (
    DiagnosisScore,
    ScorerAgent,
    ScoringResult,
    parse_scoring_response,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_ai_provider():
    """Mock AI provider that returns a well-formed scoring response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: +1\n"
        "Explanation: The documented treatments align with NICE guidelines for low back pain.\n"
        "Guidelines Followed: Exercise therapy recommended, NSAIDs prescribed\n"
        "Guidelines Not Followed: None"
    )
    return provider


@pytest.fixture()
def mock_ai_provider_non_adherent():
    """Mock AI provider that returns a non-adherent response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: -1\n"
        "Explanation: No treatments or referrals documented for this diagnosis.\n"
        "Guidelines Followed: None\n"
        "Guidelines Not Followed: Exercise therapy, physiotherapy referral"
    )
    return provider


@pytest.fixture()
def sample_extraction():
    """ExtractionResult with one episode containing diagnosis + treatments."""
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
                        term="Ibuprofen 400mg tablets",
                        concept_display="Ibuprofen",
                        cons_date=date(2024, 1, 15),
                        category="treatment",
                    ),
                    CategorisedEntry(
                        concept_id="67890",
                        term="Physiotherapy referral",
                        concept_display="Physiotherapy",
                        cons_date=date(2024, 1, 15),
                        category="referral",
                    ),
                ],
            )
        ],
        total_entries=3,
        total_diagnoses=1,
    )


@pytest.fixture()
def sample_retrieval():
    """RetrievalResult with guidelines for one diagnosis."""
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
                        title="Low back pain and sciatica in over 16s",
                        source="nice",
                        url="https://nice.org.uk/guidance/ng59",
                        clean_text="Offer exercise therapy as first-line treatment. "
                        "Consider NSAIDs for short-term pain relief.",
                        score=0.12,
                        rank=1,
                        matched_query="NICE guidelines for low back pain",
                    ),
                    GuidelineMatch(
                        guideline_id="ng59-2",
                        title="Low back pain and sciatica in over 16s",
                        source="nice",
                        url="https://nice.org.uk/guidance/ng59",
                        clean_text="Consider referral to physiotherapy. "
                        "Do not offer opioids for chronic low back pain.",
                        score=0.18,
                        rank=2,
                        matched_query="low back pain treatment referral",
                    ),
                ],
            )
        ],
        total_diagnoses=1,
        total_guidelines=2,
    )


@pytest.fixture()
def empty_extraction():
    """ExtractionResult with no episodes."""
    return ExtractionResult(
        pat_id="pat-002",
        episodes=[],
        total_entries=0,
        total_diagnoses=0,
    )


@pytest.fixture()
def empty_retrieval():
    """RetrievalResult with no guidelines."""
    return RetrievalResult(
        pat_id="pat-002",
        diagnosis_guidelines=[],
        total_diagnoses=0,
        total_guidelines=0,
    )


@pytest.fixture()
def multi_diagnosis_extraction():
    """ExtractionResult with two episodes."""
    return ExtractionResult(
        pat_id="pat-003",
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
                        term="Naproxen",
                        concept_display="Naproxen",
                        cons_date=date(2024, 1, 15),
                        category="treatment",
                    ),
                ],
            ),
            PatientEpisode(
                index_date=date(2024, 6, 1),
                entries=[
                    CategorisedEntry(
                        concept_id="239873007",
                        term="Osteoarthritis of knee",
                        concept_display="OA knee",
                        cons_date=date(2024, 6, 1),
                        category="diagnosis",
                    ),
                ],
            ),
        ],
        total_entries=3,
        total_diagnoses=2,
    )


@pytest.fixture()
def multi_diagnosis_retrieval():
    """RetrievalResult with guidelines for two diagnoses."""
    return RetrievalResult(
        pat_id="pat-003",
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
            ),
            DiagnosisGuidelines(
                diagnosis_term="Osteoarthritis of knee",
                concept_id="239873007",
                index_date="2024-06-01",
                guidelines=[
                    GuidelineMatch(
                        guideline_id="cg177-1",
                        title="Osteoarthritis guideline",
                        source="nice",
                        url="",
                        clean_text="Offer exercise and weight management.",
                        score=0.15,
                        rank=1,
                        matched_query="q",
                    ),
                ],
            ),
        ],
        total_diagnoses=2,
        total_guidelines=2,
    )


# ── parse_scoring_response tests ─────────────────────────────────────


class TestParseScoringResponse:
    def test_parse_adherent_response(self):
        response = (
            "Score: +1\n"
            "Explanation: Treatments align with guidelines.\n"
            "Guidelines Followed: Exercise therapy, NSAID prescription\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)

        assert result["score"] == 1
        assert "align" in result["explanation"]
        assert len(result["guidelines_followed"]) == 2
        assert result["guidelines_not_followed"] == []

    def test_parse_non_adherent_response(self):
        response = (
            "Score: -1\n"
            "Explanation: No treatments documented.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: Exercise therapy, physiotherapy referral"
        )
        result = parse_scoring_response(response)

        assert result["score"] == -1
        assert "No treatments" in result["explanation"]
        assert result["guidelines_followed"] == []
        assert len(result["guidelines_not_followed"]) == 2

    def test_parse_score_without_plus_sign(self):
        """Score: 1 (no plus sign) should still parse as adherent."""
        response = (
            "Score: 1\n"
            "Explanation: Good.\n"
            "Guidelines Followed: Something\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)
        assert result["score"] == 1

    def test_parse_defaults_to_non_adherent(self):
        """If score can't be parsed, default to -1."""
        result = parse_scoring_response("Some garbage response")
        assert result["score"] == -1
        assert result["explanation"] == ""
        assert result["guidelines_followed"] == []
        assert result["guidelines_not_followed"] == []

    def test_parse_multiline_explanation(self):
        response = (
            "Score: +1\n"
            "Explanation: The GP prescribed appropriate treatment.\n"
            "The referral was timely.\n"
            "Guidelines Followed: NSAID prescription\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)
        assert "prescribed appropriate" in result["explanation"]
        assert "referral was timely" in result["explanation"]

    def test_parse_case_insensitive(self):
        response = (
            "score: +1\n"
            "explanation: Good care.\n"
            "guidelines followed: Treatment A\n"
            "guidelines not followed: none"
        )
        result = parse_scoring_response(response)
        assert result["score"] == 1
        assert result["explanation"] == "Good care."
        assert result["guidelines_followed"] == ["Treatment A"]
        assert result["guidelines_not_followed"] == []

    def test_parse_extra_whitespace(self):
        response = (
            "Score:   +1  \n"
            "Explanation:   Some explanation here.  \n"
            "Guidelines Followed:   Item A ,  Item B  \n"
            "Guidelines Not Followed:   None  "
        )
        result = parse_scoring_response(response)
        assert result["score"] == 1
        assert result["explanation"] == "Some explanation here."
        assert result["guidelines_followed"] == ["Item A", "Item B"]

    def test_parse_single_guideline_items(self):
        response = (
            "Score: -1\n"
            "Explanation: Missing referral.\n"
            "Guidelines Followed: NSAIDs\n"
            "Guidelines Not Followed: Physiotherapy referral"
        )
        result = parse_scoring_response(response)
        assert result["guidelines_followed"] == ["NSAIDs"]
        assert result["guidelines_not_followed"] == ["Physiotherapy referral"]

    def test_parse_score_with_square_brackets(self):
        """LLMs sometimes output Score: [+1] with brackets. Must parse correctly."""
        adherent = (
            "Score: [+1]\n"
            "Explanation: Referral was appropriate.\n"
            "Guidelines Followed: Physiotherapy referral\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(adherent)
        assert result["score"] == 1

        non_adherent = (
            "Score: [-1]\n"
            "Explanation: No actions taken.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: Treatment"
        )
        result = parse_scoring_response(non_adherent)
        assert result["score"] == -1


# ── DiagnosisScore tests ─────────────────────────────────────────────


class TestDiagnosisScore:
    def test_creation_adherent(self):
        ds = DiagnosisScore(
            diagnosis_term="Low back pain",
            concept_id="279039007",
            index_date="2024-01-15",
            score=1,
            explanation="Good adherence.",
            guidelines_followed=["Exercise", "NSAIDs"],
            guidelines_not_followed=[],
        )
        assert ds.score == 1
        assert ds.error is None
        assert len(ds.guidelines_followed) == 2

    def test_creation_with_error(self):
        ds = DiagnosisScore(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            score=-1,
            explanation="Scoring failed.",
            error="API timeout",
        )
        assert ds.error == "API timeout"
        assert ds.score == -1


# ── ScoringResult tests ─────────────────────────────────────────────


class TestScoringResult:
    def test_aggregate_score_all_adherent(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 1, "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", 1, "ok"),
            ],
            total_diagnoses=2,
            adherent_count=2,
            non_adherent_count=0,
        )
        assert sr.aggregate_score == 1.0

    def test_aggregate_score_all_non_adherent(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", -1, "bad"),
                DiagnosisScore("D2", "2", "2024-01-01", -1, "bad"),
            ],
            total_diagnoses=2,
            adherent_count=0,
            non_adherent_count=2,
        )
        assert sr.aggregate_score == 0.0

    def test_aggregate_score_mixed(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 1, "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", -1, "bad"),
            ],
            total_diagnoses=2,
            adherent_count=1,
            non_adherent_count=1,
        )
        assert sr.aggregate_score == 0.5

    def test_aggregate_score_with_errors(self):
        """Errors should not count toward the aggregate."""
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 1, "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", -1, "failed", error="timeout"),
            ],
            total_diagnoses=2,
            adherent_count=1,
            non_adherent_count=0,
            error_count=1,
        )
        # Only 1 scored (adherent), errors excluded
        assert sr.aggregate_score == 1.0

    def test_aggregate_score_no_diagnoses(self):
        sr = ScoringResult(pat_id="pat-001")
        assert sr.aggregate_score == 0.0

    def test_summary_structure(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Low back pain",
                    concept_id="279039007",
                    index_date="2024-01-15",
                    score=1,
                    explanation="Good.",
                    guidelines_followed=["Exercise"],
                    guidelines_not_followed=[],
                ),
            ],
            total_diagnoses=1,
            adherent_count=1,
            non_adherent_count=0,
        )
        summary = sr.summary()

        assert summary["pat_id"] == "pat-001"
        assert summary["total_diagnoses"] == 1
        assert summary["adherent"] == 1
        assert summary["non_adherent"] == 0
        assert summary["errors"] == 0
        assert summary["aggregate_score"] == 1.0
        assert len(summary["scores"]) == 1
        assert summary["scores"][0]["diagnosis"] == "Low back pain"
        assert summary["scores"][0]["score"] == 1


# ── ScorerAgent tests ────────────────────────────────────────────────


class TestScorerAgent:
    @pytest.mark.asyncio
    async def test_score_single_diagnosis(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert isinstance(result, ScoringResult)
        assert result.pat_id == "pat-001"
        assert result.total_diagnoses == 1
        assert result.adherent_count == 1
        assert result.non_adherent_count == 0

    @pytest.mark.asyncio
    async def test_score_calls_llm(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        # Should call LLM once per diagnosis
        assert mock_ai_provider.chat_simple.call_count == 1

    @pytest.mark.asyncio
    async def test_score_prompt_contains_diagnosis(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        call_args = mock_ai_provider.chat_simple.call_args
        prompt = call_args[0][0]  # First positional arg

        assert "Low back pain" in prompt
        assert "Ibuprofen 400mg tablets" in prompt
        assert "Physiotherapy referral" in prompt

    @pytest.mark.asyncio
    async def test_score_prompt_contains_guidelines(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        call_args = mock_ai_provider.chat_simple.call_args
        prompt = call_args[0][0]

        assert "exercise therapy" in prompt.lower()
        assert "consider nsaids" in prompt.lower()

    @pytest.mark.asyncio
    async def test_score_uses_temperature_zero(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        call_kwargs = mock_ai_provider.chat_simple.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_score_multi_diagnosis(
        self,
        mock_ai_provider,
        multi_diagnosis_extraction,
        multi_diagnosis_retrieval,
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(
            multi_diagnosis_extraction, multi_diagnosis_retrieval
        )

        assert result.total_diagnoses == 2
        assert mock_ai_provider.chat_simple.call_count == 2
        assert result.adherent_count == 2

    @pytest.mark.asyncio
    async def test_score_empty_inputs(
        self, mock_ai_provider, empty_extraction, empty_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(empty_extraction, empty_retrieval)

        assert result.total_diagnoses == 0
        assert result.adherent_count == 0
        assert result.aggregate_score == 0.0
        assert mock_ai_provider.chat_simple.call_count == 0

    @pytest.mark.asyncio
    async def test_score_non_adherent(
        self,
        mock_ai_provider_non_adherent,
        sample_extraction,
        sample_retrieval,
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider_non_adherent)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert result.non_adherent_count == 1
        assert result.adherent_count == 0
        ds = result.diagnosis_scores[0]
        assert ds.score == -1
        assert len(ds.guidelines_not_followed) == 2

    @pytest.mark.asyncio
    async def test_score_llm_error_handled(
        self, sample_extraction, sample_retrieval
    ):
        """If the LLM raises an exception, it should be caught and logged."""
        provider = AsyncMock()
        provider.chat_simple.side_effect = Exception("API timeout")

        agent = ScorerAgent(ai_provider=provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert result.error_count == 1
        ds = result.diagnosis_scores[0]
        assert ds.error == "API timeout"
        assert ds.score == -1

    @pytest.mark.asyncio
    async def test_score_no_episode_for_diagnosis(
        self, mock_ai_provider, sample_retrieval
    ):
        """If extraction has no matching episode, use 'None documented'."""
        # Extraction with different date than retrieval expects
        extraction = ExtractionResult(
            pat_id="pat-001",
            episodes=[
                PatientEpisode(
                    index_date=date(2023, 6, 1),  # Different date
                    entries=[],
                ),
            ],
            total_entries=0,
            total_diagnoses=0,
        )

        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(extraction, sample_retrieval)

        call_args = mock_ai_provider.chat_simple.call_args
        prompt = call_args[0][0]
        assert "None documented" in prompt

    @pytest.mark.asyncio
    async def test_score_stores_guideline_titles(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        ds = result.diagnosis_scores[0]
        assert "Low back pain and sciatica in over 16s" in ds.guideline_titles_used

    @pytest.mark.asyncio
    async def test_score_diagnosis_fields(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        ds = result.diagnosis_scores[0]
        assert ds.diagnosis_term == "Low back pain"
        assert ds.concept_id == "279039007"
        assert ds.index_date == "2024-01-15"
        assert ds.explanation != ""

    @pytest.mark.asyncio
    async def test_duplicate_diagnosis_same_episode_skipped(
        self, mock_ai_provider, sample_extraction
    ):
        """Same (diagnosis_term, index_date) should be scored once; duplicate skipped."""
        # Retrieval with duplicate diagnosis in the same episode
        retrieval = RetrievalResult(
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
                ),
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
                ),
            ],
            total_diagnoses=2,
            total_guidelines=2,
        )

        agent = ScorerAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, retrieval)

        # Only 1 unique (diagnosis, date) → only 1 score entry
        assert result.total_diagnoses == 1
        assert len(result.diagnosis_scores) == 1
        # LLM called only once
        assert mock_ai_provider.chat_simple.call_count == 1


# ── _format_guidelines tests ─────────────────────────────────────────


class TestFormatGuidelines:
    def test_format_with_guidelines(self, mock_ai_provider, sample_retrieval):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        dg = sample_retrieval.diagnosis_guidelines[0]

        text = agent._format_guidelines(dg)
        assert "Low back pain and sciatica" in text
        assert "exercise therapy" in text.lower()

    def test_format_no_guidelines(self, mock_ai_provider):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        dg = DiagnosisGuidelines(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            guidelines=[],
        )

        text = agent._format_guidelines(dg)
        assert text == "No relevant guidelines found."

    def test_format_respects_max_chars(self, mock_ai_provider):
        """Guidelines should be truncated to scorer_max_guideline_chars."""
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        # Override max chars to a small value
        agent._max_guideline_chars = 100

        dg = DiagnosisGuidelines(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            guidelines=[
                GuidelineMatch(
                    guideline_id="g1",
                    title="Very Long Guideline",
                    source="nice",
                    url="",
                    clean_text="A" * 500,
                    score=0.1,
                    rank=1,
                    matched_query="q",
                ),
            ],
        )

        text = agent._format_guidelines(dg)
        assert len(text) <= 150  # Some overhead for header + ellipsis

    def test_format_sorts_by_rank(self, mock_ai_provider):
        agent = ScorerAgent(ai_provider=mock_ai_provider)
        dg = DiagnosisGuidelines(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            guidelines=[
                GuidelineMatch(
                    guideline_id="g2",
                    title="Second",
                    source="nice",
                    url="",
                    clean_text="Second guideline text.",
                    score=0.2,
                    rank=2,
                    matched_query="q",
                ),
                GuidelineMatch(
                    guideline_id="g1",
                    title="First",
                    source="nice",
                    url="",
                    clean_text="First guideline text.",
                    score=0.1,
                    rank=1,
                    matched_query="q",
                ),
            ],
        )

        text = agent._format_guidelines(dg)
        # First should appear before Second
        assert text.index("First") < text.index("Second")
