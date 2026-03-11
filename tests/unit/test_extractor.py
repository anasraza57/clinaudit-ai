"""
Tests for the Extractor Agent.

Verifies that the agent correctly groups entries by index date,
applies categories, and produces structured extraction results.
"""

from datetime import date

import pytest

from src.agents.extractor import ConsultationInsightAgent, ExtractionResult


@pytest.fixture()
def agent():
    """Create an ConsultationInsightAgent with a pre-loaded category cache."""
    a = ConsultationInsightAgent(ai_provider=None)
    a._category_cache = {
        "Low back pain": "diagnosis",
        "Fracture of phalanx of finger": "diagnosis",
        "Referral to orthopaedic service": "referral",
        "Injection of steroid into knee joint": "treatment",
        "Blood test requested": "investigation",
        "Telephone consultation": "administrative",
    }
    return a


@pytest.fixture()
def sample_entries():
    """Sample clinical entries for a patient."""
    return [
        {
            "concept_id": "279039007",
            "term": "Low back pain",
            "concept_display": "Low back pain",
            "index_date": date(2024, 1, 15),
            "cons_date": date(2024, 1, 15),
            "notes": None,
        },
        {
            "concept_id": "183545006",
            "term": "Orthopaedic referral",
            "concept_display": "Referral to orthopaedic service",
            "index_date": date(2024, 1, 15),
            "cons_date": date(2024, 2, 1),
            "notes": None,
        },
        {
            "concept_id": "12345",
            "term": "Blood test",
            "concept_display": "Blood test requested",
            "index_date": date(2024, 1, 15),
            "cons_date": date(2024, 1, 20),
            "notes": None,
        },
        # Different index date = different episode
        {
            "concept_id": "18171007",
            "term": "Finger fracture",
            "concept_display": "Fracture of phalanx of finger",
            "index_date": date(2024, 6, 1),
            "cons_date": date(2024, 6, 1),
            "notes": "Left index finger",
        },
    ]


class TestConsultationInsightAgent:
    def test_extract_groups_by_index_date(self, agent, sample_entries):
        result = agent.extract("pat-001", sample_entries)
        assert len(result.episodes) == 2
        assert result.episodes[0].index_date == date(2024, 1, 15)
        assert result.episodes[1].index_date == date(2024, 6, 1)

    def test_extract_categorises_entries(self, agent, sample_entries):
        result = agent.extract("pat-001", sample_entries)
        ep1 = result.episodes[0]

        assert len(ep1.diagnoses) == 1
        assert ep1.diagnoses[0].term == "Low back pain"

        assert len(ep1.referrals) == 1
        assert ep1.referrals[0].concept_display == "Referral to orthopaedic service"

        assert len(ep1.investigations) == 1

    def test_extract_counts(self, agent, sample_entries):
        result = agent.extract("pat-001", sample_entries)
        assert result.total_entries == 4
        assert result.total_diagnoses == 2  # Low back pain + Finger fracture
        assert result.pat_id == "pat-001"

    def test_extract_summary(self, agent, sample_entries):
        result = agent.extract("pat-001", sample_entries)
        summary = result.summary()

        assert summary["pat_id"] == "pat-001"
        assert summary["episodes"] == 2
        assert summary["total_diagnoses"] == 2
        assert len(summary["diagnoses"]) == 2
        assert summary["diagnoses"][0]["diagnosis"] == "Low back pain"

    def test_extract_with_string_dates(self, agent):
        entries = [
            {
                "concept_id": "279039007",
                "term": "Low back pain",
                "concept_display": "Low back pain",
                "index_date": "2024-01-15",
                "cons_date": "2024-01-15",
                "notes": None,
            },
        ]
        result = agent.extract("pat-002", entries)
        assert result.episodes[0].index_date == date(2024, 1, 15)

    def test_extract_unknown_concept_gets_other(self, agent):
        entries = [
            {
                "concept_id": "999999",
                "term": "Unknown thing",
                "concept_display": "Something not in cache",
                "index_date": date(2024, 1, 1),
                "cons_date": date(2024, 1, 1),
                "notes": None,
            },
        ]
        result = agent.extract("pat-003", entries)
        assert result.episodes[0].entries[0].category == "administrative"

    def test_episode_properties(self, agent, sample_entries):
        result = agent.extract("pat-001", sample_entries)
        ep1 = result.episodes[0]

        assert len(ep1.treatments) == 0
        assert len(ep1.procedures) == 0

    def test_cache_size(self, agent):
        assert agent.cache_size == 6

    @pytest.mark.asyncio
    async def test_load_categories(self):
        agent = ConsultationInsightAgent(ai_provider=None)
        concepts = ["Low back pain", "Referral to physiotherapist"]
        mapping = await agent.load_categories(concepts)

        assert mapping["Low back pain"] == "diagnosis"
        assert mapping["Referral to physiotherapist"] == "referral"
        assert agent.cache_size == 2
