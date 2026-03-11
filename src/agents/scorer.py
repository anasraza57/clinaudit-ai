"""
Compliance Auditor Agent — Stage 4 (final) of the audit pipeline.

Takes the ExtractionResult (what the GP did) and the RetrievalResult
(what NICE guidelines recommend) and evaluates whether the documented
clinical care adheres to the guidelines.

The Compliance Auditor Agent's job:
1. For each diagnosis, combine the patient's actions (treatments,
   referrals, investigations) with the retrieved guideline texts
2. Ask the LLM to evaluate adherence using a 5-level grading scale
3. Parse the response into a structured score (-2 to +2) with
   confidence and NICE guideline citations
4. Produce an aggregate score for the entire patient

This agent requires an LLM provider — it cannot function without one.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum

from src.agents.extractor import ExtractionResult, PatientEpisode
from src.agents.retriever import DiagnosisGuidelines, RetrievalResult
from src.ai.base import AIProvider
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Judgement scale ─────────────────────────────────────────────────


class AuditJudgement(IntEnum):
    """5-level audit judgement scale."""

    RISKY_NON_COMPLIANT = -2
    NON_COMPLIANT = -1
    NOT_RELEVANT = 0
    PARTIALLY_COMPLIANT = 1
    COMPLIANT = 2


JUDGEMENT_LABELS: dict[int, str] = {
    2: "COMPLIANT",
    1: "PARTIALLY COMPLIANT",
    0: "NOT RELEVANT",
    -1: "NON-COMPLIANT",
    -2: "RISKY NON-COMPLIANT",
}

# ── Scoring prompt ───────────────────────────────────────────────────

SCORING_PROMPT = """You are a clinical audit expert evaluating whether a GP's management of a musculoskeletal condition adheres to NICE clinical guidelines.

## Patient Information

**Diagnosis:** {diagnosis}
**Index Date:** {index_date}

**Documented Actions (from coded SNOMED records):**
- Treatments: {treatments}
- Referrals: {referrals}
- Investigations: {investigations}
- Procedures: {procedures}

## Relevant NICE Guidelines

{guidelines}

## Task

Evaluate whether the documented clinical actions follow the NICE guidelines for this diagnosis.

Consider:
1. Were any appropriate management actions taken (treatments, referrals, or investigations)?
2. If a referral was made (e.g., to physiotherapy, a specialist, or further care), does that align with guideline recommendations?
3. Is there evidence of at least SOME appropriate clinical response to the diagnosis?
4. Were any actions taken that the guidelines specifically advise AGAINST?

## Important Rules — READ CAREFULLY

**About the data:** These are SNOMED-coded clinical records, NOT free-text notes. Many GP actions are NOT captured in coded data — verbal advice, over-the-counter recommendations, prescriptions from separate systems, and clinical reasoning are typically absent. The absence of coded treatments does NOT mean no treatment was given.

**Scoring guidance — use this 5-level scale:**

- Score **+2 (COMPLIANT)**: Documented actions clearly and fully align with NICE guidelines. Multiple recommended steps are present (e.g., appropriate treatment AND referral AND investigation as recommended).
- Score **+1 (PARTIALLY COMPLIANT)**: Some guideline-aligned actions are documented but with minor gaps. For example, a referral was made (which is correct) but a recommended first-line treatment is absent from the coded record. Give the benefit of the doubt — absence of coded data does not mean absence of care.
- Score **0 (NOT RELEVANT)**: The guideline is not meaningfully applicable to this episode, OR the coded data is too sparse to make any judgement. Use this when there is genuinely no basis for comparison.
- Score **-1 (NON-COMPLIANT)**: The diagnosis is documented but there are NO treatments, NO referrals, AND NO investigations at all — a complete absence of any documented management. Or, the documented actions clearly deviate from guidelines without safety risk.
- Score **-2 (RISKY NON-COMPLIANT)**: The documented actions directly CONTRADICT the guidelines in a way that could cause patient harm. For example, prescribing a treatment the guidelines specifically advise against (e.g., opioids for chronic non-specific low back pain), or failing to refer when the guidelines flag a safety-critical red flag.

**General principles:**
- Give the benefit of the doubt — GPs may have good clinical reasons for their approach, and many appropriate actions are not captured in coded records
- A physiotherapy or specialist referral alone warrants at least +1 — this IS first-line management for most MSK conditions per NICE guidelines
- Base your evaluation ONLY on the provided guidelines, not general medical knowledge
- You MUST cite the specific guideline text that informed your judgement

## Output Format

You MUST respond in EXACTLY this format:

Score: -2, -1, 0, +1, or +2
Judgement: COMPLIANT, PARTIALLY COMPLIANT, NOT RELEVANT, NON-COMPLIANT, or RISKY NON-COMPLIANT
Confidence: a number between 0.0 and 1.0 indicating your confidence in this judgement
Cited Guideline: a direct quote from the NICE guideline text above that most informed your judgement, or "None" if score is 0
Explanation: 2-3 sentence explanation of your reasoning
Guidelines Followed: comma-separated list of guideline recommendations that were followed, or "None"
Guidelines Not Followed: comma-separated list of guideline recommendations that were NOT followed, or "None"
Missing Care Opportunities: comma-separated list of specific NICE-recommended actions that SHOULD have been documented but are NOT present in the patient record (e.g., "exercise therapy advice", "weight management referral"), or "None" if no gaps identified

Example:
Score: +1
Judgement: PARTIALLY COMPLIANT
Confidence: 0.75
Cited Guideline: "Consider referral to physiotherapy. Do not offer opioids for chronic low back pain."
Explanation: The GP referred the patient to physiotherapy which is recommended by NICE guidelines. However, there is no coded evidence of exercise therapy advice or NSAID prescription.
Guidelines Followed: Physiotherapy referral
Guidelines Not Followed: Exercise therapy advice, NSAID prescription
Missing Care Opportunities: Exercise therapy advice, NSAID prescription"""


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class DiagnosisScore:
    """The audit score for a single diagnosis."""

    diagnosis_term: str
    concept_id: str
    index_date: str
    score: int  # -2 to +2 (AuditJudgement scale)
    judgement: str  # Human-readable label from JUDGEMENT_LABELS
    explanation: str
    confidence: float = 0.0  # 0.0 to 1.0
    cited_guideline_text: str = ""
    guidelines_followed: list[str] = field(default_factory=list)
    guidelines_not_followed: list[str] = field(default_factory=list)
    missing_care_opportunities: list[str] = field(default_factory=list)
    guideline_titles_used: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ScoringResult:
    """The output of the Compliance Auditor Agent for one patient."""

    pat_id: str
    diagnosis_scores: list[DiagnosisScore] = field(default_factory=list)
    total_diagnoses: int = 0
    compliant_count: int = 0
    partial_count: int = 0
    not_relevant_count: int = 0
    non_compliant_count: int = 0
    risky_count: int = 0
    error_count: int = 0

    @property
    def adherent_count(self) -> int:
        """Compliant + partially compliant."""
        return self.compliant_count + self.partial_count

    @property
    def non_adherent_count(self) -> int:
        """Non-compliant + risky non-compliant."""
        return self.non_compliant_count + self.risky_count

    @property
    def aggregate_score(self) -> float:
        """
        Normalized aggregate score (0.0 to 1.0).

        Maps each score from [-2, +2] to [0.0, 1.0] then averages:
        -2 -> 0.0, -1 -> 0.25, 0 -> 0.5, 1 -> 0.75, 2 -> 1.0

        Only counts successfully scored diagnoses (excludes errors).
        """
        scored = [ds for ds in self.diagnosis_scores if ds.error is None]
        if not scored:
            return 0.0
        normalized = [(ds.score + 2) / 4 for ds in scored]
        return sum(normalized) / len(normalized)

    def summary(self) -> dict:
        return {
            "pat_id": self.pat_id,
            "total_diagnoses": self.total_diagnoses,
            "compliant": self.compliant_count,
            "partial": self.partial_count,
            "not_relevant": self.not_relevant_count,
            "non_compliant": self.non_compliant_count,
            "risky": self.risky_count,
            "adherent": self.adherent_count,
            "non_adherent": self.non_adherent_count,
            "errors": self.error_count,
            "aggregate_score": round(self.aggregate_score, 3),
            "scores": [
                {
                    "diagnosis": ds.diagnosis_term,
                    "index_date": ds.index_date,
                    "score": ds.score,
                    "judgement": ds.judgement,
                    "confidence": ds.confidence,
                    "cited_guideline_text": ds.cited_guideline_text,
                    "explanation": ds.explanation,
                    "guidelines_followed": ds.guidelines_followed,
                    "guidelines_not_followed": ds.guidelines_not_followed,
                    "missing_care_opportunities": ds.missing_care_opportunities,
                    "error": ds.error,
                }
                for ds in self.diagnosis_scores
            ],
        }


# ── Response parsing ─────────────────────────────────────────────────

_SCORE_PATTERN = re.compile(r"Score:\s*\[?([+-]?[012])\]?", re.IGNORECASE)
_JUDGEMENT_PATTERN = re.compile(
    r"Judgement:\s*(.+?)(?=\nConfidence:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_CONFIDENCE_PATTERN = re.compile(
    r"Confidence:\s*([01](?:\.\d+)?)",
    re.IGNORECASE,
)
_CITED_GUIDELINE_PATTERN = re.compile(
    r'Cited Guideline:\s*"?(.+?)"?\s*(?=\nExplanation:|\Z)',
    re.IGNORECASE | re.DOTALL,
)
_EXPLANATION_PATTERN = re.compile(
    r"Explanation:\s*(.+?)(?=\nGuidelines Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Followed:\s*(.+?)(?=\nGuidelines Not Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NOT_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Not Followed:\s*(.+?)(?=\nMissing Care Opportunities:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_MISSING_CARE_PATTERN = re.compile(
    r"Missing Care Opportunities:\s*(.+?)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_scoring_response(response_text: str) -> dict:
    """
    Parse the LLM's scoring response into structured fields.

    Returns a dict with: score, judgement, confidence, cited_guideline_text,
    explanation, guidelines_followed, guidelines_not_followed.
    """
    result = {
        "score": -1,
        "judgement": "NON-COMPLIANT",
        "confidence": 0.0,
        "cited_guideline_text": "",
        "explanation": "",
        "guidelines_followed": [],
        "guidelines_not_followed": [],
        "missing_care_opportunities": [],
    }

    # Parse score (-2 to +2)
    score_match = _SCORE_PATTERN.search(response_text)
    if score_match:
        score_val = int(score_match.group(1))
        if -2 <= score_val <= 2:
            result["score"] = score_val
            result["judgement"] = JUDGEMENT_LABELS.get(score_val, "NON-COMPLIANT")

    # Parse judgement (overrides the label derived from score if present)
    judgement_match = _JUDGEMENT_PATTERN.search(response_text)
    if judgement_match:
        result["judgement"] = judgement_match.group(1).strip()

    # Parse confidence
    conf_match = _CONFIDENCE_PATTERN.search(response_text)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))

    # Parse cited guideline
    cited_match = _CITED_GUIDELINE_PATTERN.search(response_text)
    if cited_match:
        raw = cited_match.group(1).strip().strip('"')
        if raw.lower() != "none":
            result["cited_guideline_text"] = raw

    # Parse explanation
    expl_match = _EXPLANATION_PATTERN.search(response_text)
    if expl_match:
        result["explanation"] = expl_match.group(1).strip()

    # Parse guidelines followed
    followed_match = _FOLLOWED_PATTERN.search(response_text)
    if followed_match:
        raw = followed_match.group(1).strip()
        if raw.lower() != "none":
            result["guidelines_followed"] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]

    # Parse guidelines not followed
    not_followed_match = _NOT_FOLLOWED_PATTERN.search(response_text)
    if not_followed_match:
        raw = not_followed_match.group(1).strip()
        if raw.lower() != "none":
            result["guidelines_not_followed"] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]

    # Parse missing care opportunities
    missing_match = _MISSING_CARE_PATTERN.search(response_text)
    if missing_match:
        raw = missing_match.group(1).strip()
        if raw.lower() != "none":
            result["missing_care_opportunities"] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]

    return result


# ── Compliance Auditor Agent ─────────────────────────────────────────


class ComplianceAuditorAgent:
    """
    Evaluates guideline adherence for each diagnosis using an LLM.

    Usage:
        agent = ComplianceAuditorAgent(ai_provider=provider)
        result = await agent.score(extraction_result, retrieval_result)
    """

    def __init__(self, ai_provider: AIProvider) -> None:
        self._ai_provider = ai_provider
        settings = get_settings()
        self._max_guideline_chars = settings.scorer_max_guideline_chars

    async def score(
        self,
        extraction: ExtractionResult,
        retrieval: RetrievalResult,
    ) -> ScoringResult:
        """
        Score guideline adherence for all diagnoses.

        Args:
            extraction: The ExtractionResult from the Consultation Insight Agent.
            retrieval: The RetrievalResult from the Guideline Evidence Finder.

        Returns:
            ScoringResult with per-diagnosis scores and aggregate.
        """
        # Build a lookup: index_date → PatientEpisode
        episode_map: dict[str, PatientEpisode] = {}
        for ep in extraction.episodes:
            episode_map[str(ep.index_date)] = ep

        all_scores: list[DiagnosisScore] = []
        compliant = 0
        partial = 0
        not_relevant = 0
        non_compliant = 0
        risky = 0
        errors = 0

        # Track unique (term, index_date) to avoid duplicate score entries.
        seen_pairs: set[tuple[str, str]] = set()

        for dg in retrieval.diagnosis_guidelines:
            cache_key = (dg.diagnosis_term, dg.index_date)

            if cache_key in seen_pairs:
                logger.debug(
                    "Skipping duplicate score entry for %r (index_date=%s)",
                    dg.diagnosis_term, dg.index_date,
                )
                continue
            seen_pairs.add(cache_key)

            episode = episode_map.get(dg.index_date)
            ds = await self._score_diagnosis(
                diagnosis_term=dg.diagnosis_term,
                concept_id=dg.concept_id,
                index_date=dg.index_date,
                episode=episode,
                guidelines=dg,
            )

            all_scores.append(ds)

            if ds.error:
                errors += 1
            elif ds.score == 2:
                compliant += 1
            elif ds.score == 1:
                partial += 1
            elif ds.score == 0:
                not_relevant += 1
            elif ds.score == -1:
                non_compliant += 1
            elif ds.score == -2:
                risky += 1

        result = ScoringResult(
            pat_id=extraction.pat_id,
            diagnosis_scores=all_scores,
            total_diagnoses=len(all_scores),
            compliant_count=compliant,
            partial_count=partial,
            not_relevant_count=not_relevant,
            non_compliant_count=non_compliant,
            risky_count=risky,
            error_count=errors,
        )

        logger.info(
            "Scored patient %s: %d diagnoses, %d compliant, %d partial, "
            "%d not_relevant, %d non_compliant, %d risky, %d errors, aggregate=%.2f",
            extraction.pat_id,
            len(all_scores),
            compliant, partial, not_relevant, non_compliant, risky, errors,
            result.aggregate_score,
        )

        return result

    async def _score_diagnosis(
        self,
        diagnosis_term: str,
        concept_id: str,
        index_date: str,
        episode: PatientEpisode | None,
        guidelines: DiagnosisGuidelines,
    ) -> DiagnosisScore:
        """Score a single diagnosis against retrieved guidelines."""
        treatments = "None documented"
        referrals = "None documented"
        investigations = "None documented"
        procedures = "None documented"

        if episode:
            if episode.treatments:
                treatments = ", ".join(t.term for t in episode.treatments)
            if episode.referrals:
                referrals = ", ".join(r.term for r in episode.referrals)
            if episode.investigations:
                investigations = ", ".join(i.term for i in episode.investigations)
            if episode.procedures:
                procedures = ", ".join(p.term for p in episode.procedures)

        guidelines_text = self._format_guidelines(guidelines)
        guideline_titles = guidelines.guideline_titles

        prompt = SCORING_PROMPT.format(
            diagnosis=diagnosis_term,
            index_date=index_date,
            treatments=treatments,
            referrals=referrals,
            investigations=investigations,
            procedures=procedures,
            guidelines=guidelines_text,
        )

        try:
            response = await self._ai_provider.chat_simple(
                prompt,
                temperature=0.0,
            )
            parsed = parse_scoring_response(response)

            return DiagnosisScore(
                diagnosis_term=diagnosis_term,
                concept_id=concept_id,
                index_date=index_date,
                score=parsed["score"],
                judgement=parsed["judgement"],
                explanation=parsed["explanation"],
                confidence=parsed["confidence"],
                cited_guideline_text=parsed["cited_guideline_text"],
                guidelines_followed=parsed["guidelines_followed"],
                guidelines_not_followed=parsed["guidelines_not_followed"],
                missing_care_opportunities=parsed["missing_care_opportunities"],
                guideline_titles_used=guideline_titles,
            )

        except Exception as e:
            logger.error(
                "Scoring failed for %r (patient episode %s): %s",
                diagnosis_term,
                index_date,
                e,
            )
            return DiagnosisScore(
                diagnosis_term=diagnosis_term,
                concept_id=concept_id,
                index_date=index_date,
                score=-1,
                judgement="NON-COMPLIANT",
                explanation="Scoring failed due to an error.",
                confidence=0.0,
                guideline_titles_used=guideline_titles,
                error=str(e),
            )

    def _format_guidelines(self, dg: DiagnosisGuidelines) -> str:
        """Format guideline texts for the prompt, respecting max chars."""
        if not dg.guidelines:
            return "No relevant guidelines found."

        parts = []
        total_chars = 0

        for match in sorted(dg.guidelines, key=lambda g: g.rank):
            header = f"### {match.title}\n"
            text = match.clean_text

            addition = header + text + "\n\n"
            if total_chars + len(addition) > self._max_guideline_chars:
                remaining = self._max_guideline_chars - total_chars
                if remaining > len(header) + 50:
                    parts.append(header + text[: remaining - len(header) - 5] + "...")
                break

            parts.append(addition)
            total_chars += len(addition)

        return "\n".join(parts) if parts else "No relevant guidelines found."
