"""
Consultation Insight Agent — Stage 1 of the audit pipeline.

Takes a patient's clinical entries, categorises each using
SNOMED concept classification, and outputs a structured
extraction that downstream agents can work with.

The Consultation Insight Agent's job:
1. Group clinical entries by index_date (each date = one MSK episode)
2. Categorise each entry (diagnosis, treatment, referral, etc.)
3. Identify the primary diagnoses to audit
4. Return structured data for the Audit Query Generator
"""

import logging
from dataclasses import dataclass, field
from datetime import date

from src.ai.base import AIProvider
from src.services.snomed_categoriser import categorise_concepts

logger = logging.getLogger(__name__)


@dataclass
class CategorisedEntry:
    """A single clinical entry with its assigned category."""

    concept_id: str
    term: str
    concept_display: str
    cons_date: date
    category: str
    notes: str | None = None


@dataclass
class PatientEpisode:
    """A patient's MSK episode grouped by index date."""

    index_date: date
    entries: list[CategorisedEntry] = field(default_factory=list)

    @property
    def diagnoses(self) -> list[CategorisedEntry]:
        return [e for e in self.entries if e.category == "diagnosis"]

    @property
    def treatments(self) -> list[CategorisedEntry]:
        return [e for e in self.entries if e.category == "treatment"]

    @property
    def referrals(self) -> list[CategorisedEntry]:
        return [e for e in self.entries if e.category == "referral"]

    @property
    def investigations(self) -> list[CategorisedEntry]:
        return [e for e in self.entries if e.category == "investigation"]

    @property
    def procedures(self) -> list[CategorisedEntry]:
        return [e for e in self.entries if e.category == "procedure"]


@dataclass
class ExtractionResult:
    """The output of the Consultation Insight Agent for one patient."""

    pat_id: str
    episodes: list[PatientEpisode] = field(default_factory=list)
    total_entries: int = 0
    total_diagnoses: int = 0

    def summary(self) -> dict:
        return {
            "pat_id": self.pat_id,
            "episodes": len(self.episodes),
            "total_entries": self.total_entries,
            "total_diagnoses": self.total_diagnoses,
            "diagnoses": [
                {
                    "index_date": str(ep.index_date),
                    "diagnosis": d.term,
                    "concept_id": d.concept_id,
                }
                for ep in self.episodes
                for d in ep.diagnoses
            ],
        }


class ConsultationInsightAgent:
    """
    Extracts and categorises clinical concepts from patient records.

    Usage:
        agent = ConsultationInsightAgent(ai_provider=provider)
        # Pre-load category mappings for all concepts in the dataset
        await agent.load_categories(all_unique_concepts)
        # Extract for a specific patient
        result = agent.extract(pat_id, clinical_entries)
    """

    def __init__(self, ai_provider: AIProvider | None = None) -> None:
        self._ai_provider = ai_provider
        self._category_cache: dict[str, str] = {}

    @property
    def cache_size(self) -> int:
        return len(self._category_cache)

    async def load_categories(self, concept_displays: list[str]) -> dict[str, str]:
        """
        Pre-categorise a list of unique concept display names.

        Call this once with all unique concepts from the dataset
        before running extract() on individual patients. This is
        much more efficient than classifying per-patient.

        Returns the full mapping (also cached internally).
        """
        self._category_cache = await categorise_concepts(
            concept_displays,
            ai_provider=self._ai_provider,
        )
        logger.info("Loaded %d concept categories into cache", len(self._category_cache))
        return self._category_cache

    def set_category_cache(self, mapping: dict[str, str]) -> None:
        """Replace the category cache with pre-loaded data."""
        self._category_cache = mapping

    def get_category(self, concept_display: str) -> str:
        """Look up a concept's category from the cache."""
        return self._category_cache.get(concept_display, "administrative")

    def extract(
        self,
        pat_id: str,
        entries: list[dict],
    ) -> ExtractionResult:
        """
        Extract and categorise clinical entries for a single patient.

        Args:
            pat_id: The patient identifier.
            entries: List of dicts with keys: concept_id, term,
                     concept_display, index_date, cons_date, notes.

        Returns:
            ExtractionResult with categorised episodes.
        """
        # Group entries by index_date
        episodes_map: dict[date, list[CategorisedEntry]] = {}

        for entry in entries:
            idx_date = entry["index_date"]
            if isinstance(idx_date, str):
                idx_date = date.fromisoformat(idx_date)

            cons_dt = entry["cons_date"]
            if isinstance(cons_dt, str):
                cons_dt = date.fromisoformat(cons_dt)

            concept_display = entry["concept_display"]
            category = self.get_category(concept_display)

            categorised = CategorisedEntry(
                concept_id=entry["concept_id"],
                term=entry["term"],
                concept_display=concept_display,
                cons_date=cons_dt,
                category=category,
                notes=entry.get("notes"),
            )

            if idx_date not in episodes_map:
                episodes_map[idx_date] = []
            episodes_map[idx_date].append(categorised)

        # Build episodes sorted by date
        episodes = [
            PatientEpisode(index_date=idx_date, entries=entries_list)
            for idx_date, entries_list in sorted(episodes_map.items())
        ]

        total_diagnoses = sum(len(ep.diagnoses) for ep in episodes)

        result = ExtractionResult(
            pat_id=pat_id,
            episodes=episodes,
            total_entries=len(entries),
            total_diagnoses=total_diagnoses,
        )

        logger.debug(
            "Extracted patient %s: %d episodes, %d entries, %d diagnoses",
            pat_id, len(episodes), len(entries), total_diagnoses,
        )

        return result
