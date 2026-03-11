"""
Tests for the Compliance Auditor Agent (scorer).

Uses a mock AI provider to test scoring logic without calling a real LLM.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.extractor import CategorisedEntry, ExtractionResult, PatientEpisode
from src.agents.retriever import DiagnosisGuidelines, GuidelineMatch, RetrievalResult
from src.agents.scorer import (
    AuditJudgement,
    DiagnosisScore,
    JUDGEMENT_LABELS,
    ComplianceAuditorAgent,
    ScoringResult,
    parse_scoring_response,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_ai_provider():
    """Mock AI provider that returns a compliant (5-level) scoring response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: +2\n"
        "Judgement: COMPLIANT\n"
        "Confidence: 0.85\n"
        'Cited Guideline: "Offer exercise therapy as first-line treatment."\n'
        "Explanation: The documented treatments align with NICE guidelines for low back pain.\n"
        "Guidelines Followed: Exercise therapy recommended, NSAIDs prescribed\n"
        "Guidelines Not Followed: None"
    )
    return provider


@pytest.fixture()
def mock_ai_provider_partial():
    """Mock AI provider that returns a partially compliant response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: +1\n"
        "Judgement: PARTIALLY COMPLIANT\n"
        "Confidence: 0.7\n"
        'Cited Guideline: "Consider referral to physiotherapy."\n'
        "Explanation: Physiotherapy referral was made but no NSAID was prescribed.\n"
        "Guidelines Followed: Physiotherapy referral\n"
        "Guidelines Not Followed: NSAID prescription"
    )
    return provider


@pytest.fixture()
def mock_ai_provider_non_adherent():
    """Mock AI provider that returns a non-compliant response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: -1\n"
        "Judgement: NON-COMPLIANT\n"
        "Confidence: 0.9\n"
        'Cited Guideline: "Offer exercise therapy as first-line treatment."\n'
        "Explanation: No treatments or referrals documented for this diagnosis.\n"
        "Guidelines Followed: None\n"
        "Guidelines Not Followed: Exercise therapy, physiotherapy referral"
    )
    return provider


@pytest.fixture()
def mock_ai_provider_risky():
    """Mock AI provider that returns a risky non-compliant response."""
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.chat_simple.return_value = (
        "Score: -2\n"
        "Judgement: RISKY NON-COMPLIANT\n"
        "Confidence: 0.95\n"
        'Cited Guideline: "Do not offer opioids for chronic non-specific low back pain."\n'
        "Explanation: Opioids prescribed contrary to NICE guidelines.\n"
        "Guidelines Followed: None\n"
        "Guidelines Not Followed: Avoid opioids for chronic LBP"
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


# ── AuditJudgement enum tests ───────────────────────────────────────


class TestAuditJudgement:
    def test_values(self):
        assert AuditJudgement.RISKY_NON_COMPLIANT == -2
        assert AuditJudgement.NON_COMPLIANT == -1
        assert AuditJudgement.NOT_RELEVANT == 0
        assert AuditJudgement.PARTIALLY_COMPLIANT == 1
        assert AuditJudgement.COMPLIANT == 2

    def test_labels_cover_all_values(self):
        for member in AuditJudgement:
            assert member.value in JUDGEMENT_LABELS


# ── parse_scoring_response tests ─────────────────────────────────────


class TestParseScoringResponse:
    def test_parse_compliant_response(self):
        response = (
            "Score: +2\n"
            "Judgement: COMPLIANT\n"
            "Confidence: 0.85\n"
            'Cited Guideline: "Offer exercise therapy as first-line treatment."\n'
            "Explanation: Treatments align with guidelines.\n"
            "Guidelines Followed: Exercise therapy, NSAID prescription\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)

        assert result["score"] == 2
        assert result["judgement"] == "COMPLIANT"
        assert result["confidence"] == 0.85
        assert "exercise therapy" in result["cited_guideline_text"].lower()
        assert "align" in result["explanation"]
        assert len(result["guidelines_followed"]) == 2
        assert result["guidelines_not_followed"] == []

    def test_parse_partially_compliant(self):
        response = (
            "Score: +1\n"
            "Judgement: PARTIALLY COMPLIANT\n"
            "Confidence: 0.7\n"
            'Cited Guideline: "Consider referral to physiotherapy."\n'
            "Explanation: Referral made but no NSAID.\n"
            "Guidelines Followed: Physio referral\n"
            "Guidelines Not Followed: NSAID prescription"
        )
        result = parse_scoring_response(response)

        assert result["score"] == 1
        assert result["judgement"] == "PARTIALLY COMPLIANT"
        assert result["confidence"] == 0.7

    def test_parse_not_relevant(self):
        response = (
            "Score: 0\n"
            "Judgement: NOT RELEVANT\n"
            "Confidence: 0.6\n"
            "Cited Guideline: None\n"
            "Explanation: Guideline not applicable.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)

        assert result["score"] == 0
        assert result["confidence"] == 0.6
        assert result["cited_guideline_text"] == ""

    def test_parse_non_compliant(self):
        response = (
            "Score: -1\n"
            "Judgement: NON-COMPLIANT\n"
            "Confidence: 0.9\n"
            'Cited Guideline: "Exercise therapy recommended."\n'
            "Explanation: No treatments documented.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: Exercise therapy, physiotherapy referral"
        )
        result = parse_scoring_response(response)

        assert result["score"] == -1
        assert "No treatments" in result["explanation"]
        assert result["guidelines_followed"] == []
        assert len(result["guidelines_not_followed"]) == 2

    def test_parse_risky_non_compliant(self):
        response = (
            "Score: -2\n"
            "Judgement: RISKY NON-COMPLIANT\n"
            "Confidence: 0.95\n"
            'Cited Guideline: "Do not offer opioids for chronic low back pain."\n'
            "Explanation: Opioids prescribed contrary to NICE guidelines.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: Avoid opioids"
        )
        result = parse_scoring_response(response)

        assert result["score"] == -2
        assert result["confidence"] == 0.95
        assert "opioids" in result["cited_guideline_text"].lower()

    def test_parse_score_without_plus_sign(self):
        """Score: 2 (no plus sign) should still parse as compliant."""
        response = (
            "Score: 2\n"
            "Judgement: COMPLIANT\n"
            "Confidence: 0.8\n"
            "Cited Guideline: None\n"
            "Explanation: Good.\n"
            "Guidelines Followed: Something\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(response)
        assert result["score"] == 2

    def test_parse_defaults_to_non_compliant(self):
        """If score can't be parsed, default to -1."""
        result = parse_scoring_response("Some garbage response")
        assert result["score"] == -1
        assert result["judgement"] == "NON-COMPLIANT"
        assert result["confidence"] == 0.0
        assert result["cited_guideline_text"] == ""
        assert result["explanation"] == ""
        assert result["guidelines_followed"] == []
        assert result["guidelines_not_followed"] == []

    def test_parse_multiline_explanation(self):
        response = (
            "Score: +1\n"
            "Judgement: PARTIALLY COMPLIANT\n"
            "Confidence: 0.75\n"
            "Cited Guideline: None\n"
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
            "judgement: PARTIALLY COMPLIANT\n"
            "confidence: 0.8\n"
            "cited guideline: None\n"
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
            "Score:   +2  \n"
            "Judgement: COMPLIANT\n"
            "Confidence: 0.9\n"
            "Cited Guideline: None\n"
            "Explanation:   Some explanation here.  \n"
            "Guidelines Followed:   Item A ,  Item B  \n"
            "Guidelines Not Followed:   None  "
        )
        result = parse_scoring_response(response)
        assert result["score"] == 2
        assert result["explanation"] == "Some explanation here."
        assert result["guidelines_followed"] == ["Item A", "Item B"]

    def test_parse_single_guideline_items(self):
        response = (
            "Score: -1\n"
            "Judgement: NON-COMPLIANT\n"
            "Confidence: 0.8\n"
            'Cited Guideline: "Exercise."\n'
            "Explanation: Missing referral.\n"
            "Guidelines Followed: NSAIDs\n"
            "Guidelines Not Followed: Physiotherapy referral"
        )
        result = parse_scoring_response(response)
        assert result["guidelines_followed"] == ["NSAIDs"]
        assert result["guidelines_not_followed"] == ["Physiotherapy referral"]

    def test_parse_score_with_square_brackets(self):
        """LLMs sometimes output Score: [+2] with brackets."""
        adherent = (
            "Score: [+2]\n"
            "Judgement: COMPLIANT\n"
            "Confidence: 0.85\n"
            "Cited Guideline: None\n"
            "Explanation: Referral was appropriate.\n"
            "Guidelines Followed: Physiotherapy referral\n"
            "Guidelines Not Followed: None"
        )
        result = parse_scoring_response(adherent)
        assert result["score"] == 2

        non_adherent = (
            "Score: [-1]\n"
            "Judgement: NON-COMPLIANT\n"
            "Confidence: 0.9\n"
            "Cited Guideline: None\n"
            "Explanation: No actions taken.\n"
            "Guidelines Followed: None\n"
            "Guidelines Not Followed: Treatment"
        )
        result = parse_scoring_response(non_adherent)
        assert result["score"] == -1

    def test_parse_confidence_edge_values(self):
        """Confidence of 0 and 1 should parse correctly."""
        for conf_val, expected in [("0.0", 0.0), ("1.0", 1.0), ("0", 0.0), ("1", 1.0)]:
            response = (
                "Score: 0\n"
                "Judgement: NOT RELEVANT\n"
                f"Confidence: {conf_val}\n"
                "Cited Guideline: None\n"
                "Explanation: Test.\n"
                "Guidelines Followed: None\n"
                "Guidelines Not Followed: None"
            )
            result = parse_scoring_response(response)
            assert result["confidence"] == expected


# ── DiagnosisScore tests ─────────────────────────────────────────────


class TestDiagnosisScore:
    def test_creation_compliant(self):
        ds = DiagnosisScore(
            diagnosis_term="Low back pain",
            concept_id="279039007",
            index_date="2024-01-15",
            score=2,
            judgement="COMPLIANT",
            explanation="Good adherence.",
            confidence=0.85,
            cited_guideline_text="Exercise therapy recommended.",
            guidelines_followed=["Exercise", "NSAIDs"],
            guidelines_not_followed=[],
        )
        assert ds.score == 2
        assert ds.judgement == "COMPLIANT"
        assert ds.confidence == 0.85
        assert ds.error is None
        assert len(ds.guidelines_followed) == 2

    def test_creation_with_error(self):
        ds = DiagnosisScore(
            diagnosis_term="Test",
            concept_id="1",
            index_date="2024-01-01",
            score=-1,
            judgement="NON-COMPLIANT",
            explanation="Scoring failed.",
            error="API timeout",
        )
        assert ds.error == "API timeout"
        assert ds.score == -1


# ── ScoringResult tests ─────────────────────────────────────────────


class TestScoringResult:
    def test_aggregate_score_all_compliant(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 2, "COMPLIANT", "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", 2, "COMPLIANT", "ok"),
            ],
            total_diagnoses=2,
            compliant_count=2,
        )
        assert sr.aggregate_score == 1.0

    def test_aggregate_score_all_risky(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", -2, "RISKY NON-COMPLIANT", "bad"),
                DiagnosisScore("D2", "2", "2024-01-01", -2, "RISKY NON-COMPLIANT", "bad"),
            ],
            total_diagnoses=2,
            risky_count=2,
        )
        assert sr.aggregate_score == 0.0

    def test_aggregate_score_mixed(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 2, "COMPLIANT", "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", -2, "RISKY NON-COMPLIANT", "bad"),
            ],
            total_diagnoses=2,
            compliant_count=1,
            risky_count=1,
        )
        # (1.0 + 0.0) / 2 = 0.5
        assert sr.aggregate_score == 0.5

    def test_aggregate_score_partial_and_non_compliant(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 1, "PARTIALLY COMPLIANT", "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", -1, "NON-COMPLIANT", "bad"),
            ],
            total_diagnoses=2,
            partial_count=1,
            non_compliant_count=1,
        )
        # (0.75 + 0.25) / 2 = 0.5
        assert sr.aggregate_score == 0.5

    def test_aggregate_score_not_relevant(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 0, "NOT RELEVANT", "n/a"),
            ],
            total_diagnoses=1,
            not_relevant_count=1,
        )
        # (0 + 2) / 4 = 0.5
        assert sr.aggregate_score == 0.5

    def test_aggregate_score_with_errors(self):
        """Errors should not count toward the aggregate."""
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore("D1", "1", "2024-01-01", 2, "COMPLIANT", "ok"),
                DiagnosisScore("D2", "2", "2024-01-01", -1, "NON-COMPLIANT", "failed", error="timeout"),
            ],
            total_diagnoses=2,
            compliant_count=1,
            error_count=1,
        )
        # Only 1 scored (compliant), errors excluded → 1.0
        assert sr.aggregate_score == 1.0

    def test_aggregate_score_no_diagnoses(self):
        sr = ScoringResult(pat_id="pat-001")
        assert sr.aggregate_score == 0.0

    def test_adherent_count_property(self):
        sr = ScoringResult(
            pat_id="p", compliant_count=3, partial_count=2,
        )
        assert sr.adherent_count == 5

    def test_non_adherent_count_property(self):
        sr = ScoringResult(
            pat_id="p", non_compliant_count=2, risky_count=1,
        )
        assert sr.non_adherent_count == 3

    def test_summary_structure(self):
        sr = ScoringResult(
            pat_id="pat-001",
            diagnosis_scores=[
                DiagnosisScore(
                    diagnosis_term="Low back pain",
                    concept_id="279039007",
                    index_date="2024-01-15",
                    score=2,
                    judgement="COMPLIANT",
                    explanation="Good.",
                    confidence=0.85,
                    cited_guideline_text="Exercise therapy recommended.",
                    guidelines_followed=["Exercise"],
                    guidelines_not_followed=[],
                ),
            ],
            total_diagnoses=1,
            compliant_count=1,
        )
        summary = sr.summary()

        assert summary["pat_id"] == "pat-001"
        assert summary["total_diagnoses"] == 1
        assert summary["compliant"] == 1
        assert summary["partial"] == 0
        assert summary["adherent"] == 1
        assert summary["non_adherent"] == 0
        assert summary["errors"] == 0
        assert summary["aggregate_score"] == 1.0
        assert len(summary["scores"]) == 1
        assert summary["scores"][0]["diagnosis"] == "Low back pain"
        assert summary["scores"][0]["score"] == 2
        assert summary["scores"][0]["judgement"] == "COMPLIANT"
        assert summary["scores"][0]["confidence"] == 0.85
        assert summary["scores"][0]["cited_guideline_text"] == "Exercise therapy recommended."


# ── ComplianceAuditorAgent tests ────────────────────────────────────────────────


class TestComplianceAuditorAgent:
    @pytest.mark.asyncio
    async def test_score_single_diagnosis(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert isinstance(result, ScoringResult)
        assert result.pat_id == "pat-001"
        assert result.total_diagnoses == 1
        assert result.compliant_count == 1
        assert result.non_compliant_count == 0

    @pytest.mark.asyncio
    async def test_score_calls_llm(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        # Should call LLM once per diagnosis
        assert mock_ai_provider.chat_simple.call_count == 1

    @pytest.mark.asyncio
    async def test_score_prompt_contains_diagnosis(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        await agent.score(sample_extraction, sample_retrieval)

        call_args = mock_ai_provider.chat_simple.call_args
        prompt = call_args[0][0]

        assert "exercise therapy" in prompt.lower()
        assert "consider nsaids" in prompt.lower()

    @pytest.mark.asyncio
    async def test_score_uses_temperature_zero(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(
            multi_diagnosis_extraction, multi_diagnosis_retrieval
        )

        assert result.total_diagnoses == 2
        assert mock_ai_provider.chat_simple.call_count == 2
        assert result.compliant_count == 2

    @pytest.mark.asyncio
    async def test_score_empty_inputs(
        self, mock_ai_provider, empty_extraction, empty_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider_non_adherent)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert result.non_compliant_count == 1
        assert result.adherent_count == 0
        ds = result.diagnosis_scores[0]
        assert ds.score == -1
        assert ds.judgement == "NON-COMPLIANT"
        assert ds.confidence == 0.9
        assert len(ds.guidelines_not_followed) == 2

    @pytest.mark.asyncio
    async def test_score_risky_non_compliant(
        self,
        mock_ai_provider_risky,
        sample_extraction,
        sample_retrieval,
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider_risky)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert result.risky_count == 1
        assert result.non_adherent_count == 1
        ds = result.diagnosis_scores[0]
        assert ds.score == -2
        assert ds.confidence == 0.95
        assert "opioids" in ds.cited_guideline_text.lower()

    @pytest.mark.asyncio
    async def test_score_partial_compliant(
        self,
        mock_ai_provider_partial,
        sample_extraction,
        sample_retrieval,
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider_partial)
        result = await agent.score(sample_extraction, sample_retrieval)

        assert result.partial_count == 1
        assert result.adherent_count == 1
        ds = result.diagnosis_scores[0]
        assert ds.score == 1
        assert ds.confidence == 0.7

    @pytest.mark.asyncio
    async def test_score_llm_error_handled(
        self, sample_extraction, sample_retrieval
    ):
        """If the LLM raises an exception, it should be caught and logged."""
        provider = AsyncMock()
        provider.chat_simple.side_effect = Exception("API timeout")

        agent = ComplianceAuditorAgent(ai_provider=provider)
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

        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(extraction, sample_retrieval)

        call_args = mock_ai_provider.chat_simple.call_args
        prompt = call_args[0][0]
        assert "None documented" in prompt

    @pytest.mark.asyncio
    async def test_score_stores_guideline_titles(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        ds = result.diagnosis_scores[0]
        assert "Low back pain and sciatica in over 16s" in ds.guideline_titles_used

    @pytest.mark.asyncio
    async def test_score_diagnosis_fields(
        self, mock_ai_provider, sample_extraction, sample_retrieval
    ):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, sample_retrieval)

        ds = result.diagnosis_scores[0]
        assert ds.diagnosis_term == "Low back pain"
        assert ds.concept_id == "279039007"
        assert ds.index_date == "2024-01-15"
        assert ds.explanation != ""
        assert ds.confidence > 0
        assert ds.cited_guideline_text != ""

    @pytest.mark.asyncio
    async def test_duplicate_diagnosis_same_episode_skipped(
        self, mock_ai_provider, sample_extraction
    ):
        """Same (diagnosis_term, index_date) should be scored once; duplicate skipped."""
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

        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        result = await agent.score(sample_extraction, retrieval)

        assert result.total_diagnoses == 1
        assert len(result.diagnosis_scores) == 1
        assert mock_ai_provider.chat_simple.call_count == 1


# ── _format_guidelines tests ─────────────────────────────────────────


class TestFormatGuidelines:
    def test_format_with_guidelines(self, mock_ai_provider, sample_retrieval):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
        dg = sample_retrieval.diagnosis_guidelines[0]

        text = agent._format_guidelines(dg)
        assert "Low back pain and sciatica" in text
        assert "exercise therapy" in text.lower()

    def test_format_no_guidelines(self, mock_ai_provider):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        assert len(text) <= 150

    def test_format_sorts_by_rank(self, mock_ai_provider):
        agent = ComplianceAuditorAgent(ai_provider=mock_ai_provider)
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
        assert text.index("First") < text.index("Second")
