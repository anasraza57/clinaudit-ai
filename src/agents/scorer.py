"""
Scorer Agent — Stage 4 (final) of the audit pipeline.

Takes the ExtractionResult (what the GP did) and the RetrievalResult
(what NICE guidelines recommend) and evaluates whether the documented
clinical care adheres to the guidelines.

The Scorer's job:
1. For each diagnosis, combine the patient's actions (treatments,
   referrals, investigations) with the retrieved guideline texts
2. Ask the LLM to evaluate adherence
3. Parse the response into a structured score (+1 adherent / -1 non-adherent)
4. Produce an aggregate score for the entire patient

This agent requires an LLM provider — it cannot function without one.
"""

import logging
import re
from dataclasses import dataclass, field

from src.agents.extractor import ExtractionResult, PatientEpisode
from src.agents.retriever import DiagnosisGuidelines, RetrievalResult
from src.ai.base import AIProvider
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

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

## Important Rules — READ CAREFULLY

**About the data:** These are SNOMED-coded clinical records, NOT free-text notes. Many GP actions are NOT captured in coded data — verbal advice, over-the-counter recommendations, prescriptions from separate systems, and clinical reasoning are typically absent. The absence of coded treatments does NOT mean no treatment was given.

**Scoring guidance:**
- Score +1 (ADHERENT) if ANY of the following are true:
  - A relevant referral was made (e.g., physiotherapy, specialist, further care) — referrals ARE a form of management
  - Appropriate treatments were prescribed
  - Relevant investigations were ordered
  - The documented actions show reasonable clinical engagement with the diagnosis
- Score -1 (NON-ADHERENT) only if:
  - The diagnosis is documented but there are NO treatments, NO referrals, AND NO investigations at all
  - The documented actions clearly CONTRADICT the guidelines (e.g., a treatment the guidelines specifically advise against)
- **Give the benefit of the doubt** — GPs may have good clinical reasons for their approach, and many appropriate actions are not captured in coded records
- A physiotherapy or specialist referral alone is sufficient for +1 — this IS first-line management for most MSK conditions per NICE guidelines
- Base your evaluation ONLY on the provided guidelines, not general medical knowledge

## Output Format

You MUST respond in EXACTLY this format:

Score: +1 or -1
Explanation: 2-3 sentence explanation of your reasoning
Guidelines Followed: comma-separated list of guideline recommendations that were followed, or "None"
Guidelines Not Followed: comma-separated list of guideline recommendations that were NOT followed, or "None"

Example:
Score: +1
Explanation: The GP referred the patient to physiotherapy which is recommended by NICE guidelines for this condition.
Guidelines Followed: Physiotherapy referral, exercise advice
Guidelines Not Followed: None"""


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class DiagnosisScore:
    """The audit score for a single diagnosis."""

    diagnosis_term: str
    concept_id: str
    index_date: str
    score: int  # +1 (adherent) or -1 (non-adherent)
    explanation: str
    guidelines_followed: list[str] = field(default_factory=list)
    guidelines_not_followed: list[str] = field(default_factory=list)
    guideline_titles_used: list[str] = field(default_factory=list)
    error: str | None = None  # Set if scoring failed for this diagnosis


@dataclass
class ScoringResult:
    """The output of the Scorer Agent for one patient."""

    pat_id: str
    diagnosis_scores: list[DiagnosisScore] = field(default_factory=list)
    total_diagnoses: int = 0
    adherent_count: int = 0
    non_adherent_count: int = 0
    error_count: int = 0

    @property
    def aggregate_score(self) -> float:
        """
        Proportion of adherent diagnoses (0.0 to 1.0).

        Only counts successfully scored diagnoses (excludes errors).
        """
        scored = self.adherent_count + self.non_adherent_count
        if scored == 0:
            return 0.0
        return self.adherent_count / scored

    def summary(self) -> dict:
        return {
            "pat_id": self.pat_id,
            "total_diagnoses": self.total_diagnoses,
            "adherent": self.adherent_count,
            "non_adherent": self.non_adherent_count,
            "errors": self.error_count,
            "aggregate_score": round(self.aggregate_score, 3),
            "scores": [
                {
                    "diagnosis": ds.diagnosis_term,
                    "index_date": ds.index_date,
                    "score": ds.score,
                    "explanation": ds.explanation,
                    "guidelines_followed": ds.guidelines_followed,
                    "guidelines_not_followed": ds.guidelines_not_followed,
                    "error": ds.error,
                }
                for ds in self.diagnosis_scores
            ],
        }


# ── Response parsing ─────────────────────────────────────────────────

_SCORE_PATTERN = re.compile(r"Score:\s*\[?([+-]?1)\]?", re.IGNORECASE)
_EXPLANATION_PATTERN = re.compile(
    r"Explanation:\s*(.+?)(?=\nGuidelines Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Followed:\s*(.+?)(?=\nGuidelines Not Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NOT_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Not Followed:\s*(.+?)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_scoring_response(response_text: str) -> dict:
    """
    Parse the LLM's scoring response into structured fields.

    Returns a dict with: score, explanation, guidelines_followed, guidelines_not_followed.
    """
    result = {
        "score": -1,  # Default to non-adherent if parsing fails
        "explanation": "",
        "guidelines_followed": [],
        "guidelines_not_followed": [],
    }

    # Parse score
    score_match = _SCORE_PATTERN.search(response_text)
    if score_match:
        score_val = score_match.group(1)
        result["score"] = 1 if score_val in ("+1", "1") else -1

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

    return result


# ── Scorer Agent ─────────────────────────────────────────────────────


class ScorerAgent:
    """
    Evaluates guideline adherence for each diagnosis using an LLM.

    Usage:
        agent = ScorerAgent(ai_provider=provider)
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
            extraction: The ExtractionResult from the Extractor Agent.
            retrieval: The RetrievalResult from the Retriever Agent.

        Returns:
            ScoringResult with per-diagnosis scores and aggregate.
        """
        # Build a lookup: (diagnosis_term, index_date) → DiagnosisGuidelines
        guidelines_map: dict[tuple[str, str], DiagnosisGuidelines] = {}
        for dg in retrieval.diagnosis_guidelines:
            key = (dg.diagnosis_term, dg.index_date)
            guidelines_map[key] = dg

        # Build a lookup: index_date → PatientEpisode
        episode_map: dict[str, PatientEpisode] = {}
        for ep in extraction.episodes:
            episode_map[str(ep.index_date)] = ep

        all_scores: list[DiagnosisScore] = []
        adherent = 0
        non_adherent = 0
        errors = 0

        # Track unique (term, index_date) to avoid duplicate score entries.
        # Upstream agents should already deduplicate, but this is a safety net.
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
            elif ds.score == 1:
                adherent += 1
            else:
                non_adherent += 1

        result = ScoringResult(
            pat_id=extraction.pat_id,
            diagnosis_scores=all_scores,
            total_diagnoses=len(all_scores),
            adherent_count=adherent,
            non_adherent_count=non_adherent,
            error_count=errors,
        )

        logger.info(
            "Scored patient %s: %d diagnoses, %d adherent, %d non-adherent, %d errors, aggregate=%.2f",
            extraction.pat_id,
            len(all_scores),
            adherent,
            non_adherent,
            errors,
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
        # Build context from the episode
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

        # Build guidelines text (truncate to max chars)
        guidelines_text = self._format_guidelines(guidelines)
        guideline_titles = guidelines.guideline_titles

        # Build the prompt
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
                temperature=0.0,  # Deterministic scoring
            )
            parsed = parse_scoring_response(response)

            return DiagnosisScore(
                diagnosis_term=diagnosis_term,
                concept_id=concept_id,
                index_date=index_date,
                score=parsed["score"],
                explanation=parsed["explanation"],
                guidelines_followed=parsed["guidelines_followed"],
                guidelines_not_followed=parsed["guidelines_not_followed"],
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
                explanation="Scoring failed due to an error.",
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

            # Check if adding this guideline would exceed the limit
            addition = header + text + "\n\n"
            if total_chars + len(addition) > self._max_guideline_chars:
                # Add as much as we can fit
                remaining = self._max_guideline_chars - total_chars
                if remaining > len(header) + 50:  # Only add if meaningful
                    parts.append(header + text[: remaining - len(header) - 5] + "...")
                break

            parts.append(addition)
            total_chars += len(addition)

        return "\n".join(parts) if parts else "No relevant guidelines found."
