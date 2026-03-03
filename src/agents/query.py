"""
Query Agent — Stage 2 of the audit pipeline.

Takes the ExtractionResult from the Extractor and generates targeted
search queries for each diagnosis. These queries are later embedded
and used to search the FAISS guideline index (by the Retriever Agent).

The Query Agent's job:
1. Receive structured extraction (diagnoses per episode)
2. Generate 1-3 search queries per diagnosis, optimised for
   PubMedBERT/FAISS similarity search against NICE guidelines
3. Return structured QueryResult for the Retriever Agent

Two-tier approach (matching the Extractor pattern):
- Rule-based templates for straightforward MSK diagnoses
- LLM generation for complex or unusual diagnoses
"""

import logging
from dataclasses import dataclass, field

from src.agents.extractor import CategorisedEntry, ExtractionResult
from src.ai.base import AIProvider
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Rule-based query templates ──────────────────────────────────────
# For common MSK diagnoses, we can generate effective search queries
# using templates. These produce text that matches how NICE guidelines
# are written, which gives high cosine similarity with PubMedBERT.
#
# Template variables:
#   {dx} = the diagnosis term (e.g. "Low back pain")

_QUERY_TEMPLATES: dict[str, list[str]] = {
    # Back conditions
    "low back pain": [
        "NICE guidelines for assessment and management of low back pain and sciatica",
        "non-specific low back pain pharmacological and non-pharmacological treatment",
        "low back pain referral criteria and imaging recommendations",
    ],
    "sciatica": [
        "NICE guidelines for sciatica assessment and management",
        "sciatica treatment pathway including physiotherapy and surgery referral",
        "lumbar radiculopathy clinical management and investigation",
    ],
    "back pain": [
        "NICE guidelines for assessment and management of back pain",
        "back pain treatment options and referral criteria",
        "non-specific back pain clinical management recommendations",
    ],
    # Knee conditions
    "osteoarthritis of knee": [
        "NICE guidelines for osteoarthritis management and treatment",
        "knee osteoarthritis exercise therapy and weight management recommendations",
        "osteoarthritis referral for joint replacement criteria",
    ],
    "osteoarthritis": [
        "NICE guidelines for osteoarthritis assessment and management",
        "osteoarthritis pharmacological and non-pharmacological treatment options",
        "osteoarthritis referral criteria and surgical management",
    ],
    # Shoulder conditions
    "shoulder pain": [
        "NICE guidelines for shoulder pain assessment and management",
        "shoulder pain differential diagnosis and treatment pathway",
        "rotator cuff disorder management and referral criteria",
    ],
    "rotator cuff": [
        "rotator cuff disorder assessment and management guidelines",
        "rotator cuff tear treatment options including physiotherapy and surgery",
        "shoulder impingement syndrome clinical management",
    ],
    # Hip conditions
    "hip pain": [
        "NICE guidelines for hip pain assessment and management",
        "hip osteoarthritis treatment and referral for hip replacement",
        "hip pain investigation and differential diagnosis",
    ],
    # Hand/wrist conditions
    "carpal tunnel syndrome": [
        "NICE guidelines for carpal tunnel syndrome management",
        "carpal tunnel syndrome treatment including splinting and surgery referral",
        "nerve conduction studies and carpal tunnel diagnosis criteria",
    ],
    # Foot conditions
    "plantar fasciitis": [
        "plantar fasciitis assessment and management guidelines",
        "plantar heel pain treatment options and self-management",
        "plantar fasciitis referral criteria and orthotics",
    ],
    # Inflammatory conditions
    "gout": [
        "NICE guidelines for gout management and treatment",
        "acute gout treatment and long-term urate lowering therapy",
        "gout lifestyle advice and prophylaxis recommendations",
    ],
    "rheumatoid arthritis": [
        "NICE guidelines for rheumatoid arthritis management",
        "rheumatoid arthritis early referral and DMARD treatment",
        "inflammatory arthritis assessment and monitoring",
    ],
    # Bone conditions
    "osteoporosis": [
        "NICE guidelines for osteoporosis assessment and management",
        "osteoporosis fracture risk assessment and bisphosphonate treatment",
        "fragility fracture prevention and bone density screening",
    ],
    "fracture": [
        "fracture management guidelines and referral criteria",
        "fracture assessment initial treatment and follow-up",
        "fracture rehabilitation and return to activity recommendations",
    ],
    # Widespread pain
    "fibromyalgia": [
        "NICE guidelines for chronic pain and fibromyalgia management",
        "fibromyalgia assessment diagnosis and treatment options",
        "chronic widespread pain management exercise and psychological therapy",
    ],
}


def _normalise_diagnosis(term: str) -> str:
    """Normalise a diagnosis term for template lookup."""
    return term.strip().lower()


def _find_template(diagnosis_term: str) -> list[str] | None:
    """
    Try to find a matching query template for a diagnosis.

    Checks for exact match first, then substring matches.
    Returns the template queries or None if no match.
    """
    normalised = _normalise_diagnosis(diagnosis_term)

    # Exact match
    if normalised in _QUERY_TEMPLATES:
        return _QUERY_TEMPLATES[normalised]

    # Substring match — check if any template key appears in the diagnosis
    for key, templates in _QUERY_TEMPLATES.items():
        if key in normalised:
            return templates

    return None


def generate_default_queries(diagnosis_term: str, max_queries: int = 3) -> list[str]:
    """
    Generate generic queries when no template or LLM is available.

    These are reasonable defaults that work with PubMedBERT similarity:
    1. Direct NICE guideline query
    2. Treatment and management query
    3. Referral and investigation query
    """
    queries = [
        f"NICE clinical guidelines for {diagnosis_term} management",
        f"{diagnosis_term} treatment options and recommendations",
        f"{diagnosis_term} referral criteria and investigations",
    ]
    return queries[:max_queries]


# ── LLM query generation ────────────────────────────────────────────

QUERY_GENERATION_PROMPT = """You are a clinical guidelines search expert. Your task is to generate search queries that will find relevant NICE (National Institute for Health and Care Excellence) clinical guidelines for a specific musculoskeletal diagnosis.

The queries will be used for semantic similarity search against a database of NICE guideline documents using medical embeddings (PubMedBERT). Generate queries that use clinical terminology matching how NICE guidelines are written.

Rules:
- Generate exactly {max_queries} search queries
- Each query should cover a different aspect of care (e.g., assessment, treatment, referral)
- Use clinical terminology (not patient-facing language)
- Include "NICE" or "guidelines" in at least one query
- Keep each query concise (10-20 words)
- Output ONLY the queries, one per line, with no numbering or bullet points

Diagnosis: {diagnosis}
Context: {context}

Queries:"""


@dataclass
class DiagnosisQueries:
    """Search queries generated for a single diagnosis."""

    diagnosis_term: str
    concept_id: str
    index_date: str
    queries: list[str] = field(default_factory=list)
    source: str = ""  # "template", "llm", or "default"


@dataclass
class QueryResult:
    """The output of the Query Agent for one patient."""

    pat_id: str
    diagnosis_queries: list[DiagnosisQueries] = field(default_factory=list)
    total_diagnoses: int = 0
    total_queries: int = 0

    def summary(self) -> dict:
        return {
            "pat_id": self.pat_id,
            "total_diagnoses": self.total_diagnoses,
            "total_queries": self.total_queries,
            "diagnoses": [
                {
                    "diagnosis": dq.diagnosis_term,
                    "index_date": dq.index_date,
                    "num_queries": len(dq.queries),
                    "source": dq.source,
                    "queries": dq.queries,
                }
                for dq in self.diagnosis_queries
            ],
        }

    def all_queries(self) -> list[tuple[str, str]]:
        """Return all queries as (diagnosis_term, query) pairs."""
        return [
            (dq.diagnosis_term, q)
            for dq in self.diagnosis_queries
            for q in dq.queries
        ]


class QueryAgent:
    """
    Generates guideline search queries from extracted diagnoses.

    Usage:
        agent = QueryAgent(ai_provider=provider)
        query_result = await agent.generate_queries(extraction_result)
    """

    def __init__(self, ai_provider: AIProvider | None = None) -> None:
        self._ai_provider = ai_provider
        settings = get_settings()
        self._max_queries = settings.max_queries_per_diagnosis

    async def generate_queries(
        self,
        extraction: ExtractionResult,
    ) -> QueryResult:
        """
        Generate search queries for all diagnoses in the extraction.

        For each diagnosis found by the Extractor, generates 1-3 targeted
        queries optimised for FAISS similarity search against NICE guidelines.

        Args:
            extraction: The ExtractionResult from the Extractor Agent.

        Returns:
            QueryResult with queries for each diagnosis.
        """
        all_diagnosis_queries: list[DiagnosisQueries] = []

        # Cache queries by diagnosis term — same term always produces identical
        # queries regardless of which episode it appears in.
        query_cache: dict[str, list[str]] = {}
        source_cache: dict[str, str] = {}

        # Track unique (term, index_date) pairs to avoid duplicate entries
        # (e.g. "Finger pain" coded twice in the same episode).
        seen_pairs: set[tuple[str, str]] = set()

        for episode in extraction.episodes:
            for diagnosis in episode.diagnoses:
                term = diagnosis.term
                pair_key = (term, str(episode.index_date))

                # Skip if this exact (diagnosis, date) is already in the output
                if pair_key in seen_pairs:
                    logger.debug(
                        "Skipping duplicate entry for %r (index_date=%s)",
                        term, episode.index_date,
                    )
                    continue
                seen_pairs.add(pair_key)

                if term in query_cache:
                    logger.debug(
                        "Reusing cached queries for %r (index_date=%s)",
                        term, episode.index_date,
                    )
                    dq = DiagnosisQueries(
                        diagnosis_term=term,
                        concept_id=diagnosis.concept_id,
                        index_date=str(episode.index_date),
                        queries=query_cache[term],
                        source=source_cache[term],
                    )
                else:
                    dq = await self._generate_for_diagnosis(
                        diagnosis=diagnosis,
                        episode_context=self._build_context(episode),
                        index_date=str(episode.index_date),
                    )
                    query_cache[term] = dq.queries
                    source_cache[term] = dq.source

                all_diagnosis_queries.append(dq)

        total_queries = sum(len(dq.queries) for dq in all_diagnosis_queries)

        result = QueryResult(
            pat_id=extraction.pat_id,
            diagnosis_queries=all_diagnosis_queries,
            total_diagnoses=len(all_diagnosis_queries),
            total_queries=total_queries,
        )

        logger.info(
            "Generated %d queries for %d diagnoses (patient %s)",
            total_queries,
            len(all_diagnosis_queries),
            extraction.pat_id,
        )

        return result

    async def _generate_for_diagnosis(
        self,
        diagnosis: CategorisedEntry,
        episode_context: str,
        index_date: str,
    ) -> DiagnosisQueries:
        """Generate queries for a single diagnosis using template or LLM."""
        # Tier 1: Try rule-based templates
        template_queries = _find_template(diagnosis.term)
        if template_queries:
            queries = template_queries[: self._max_queries]
            logger.debug(
                "Template queries for %r: %d queries",
                diagnosis.term,
                len(queries),
            )
            return DiagnosisQueries(
                diagnosis_term=diagnosis.term,
                concept_id=diagnosis.concept_id,
                index_date=index_date,
                queries=queries,
                source="template",
            )

        # Also check concept_display for template match
        template_queries = _find_template(diagnosis.concept_display)
        if template_queries:
            queries = template_queries[: self._max_queries]
            logger.debug(
                "Template queries for %r (via display): %d queries",
                diagnosis.concept_display,
                len(queries),
            )
            return DiagnosisQueries(
                diagnosis_term=diagnosis.term,
                concept_id=diagnosis.concept_id,
                index_date=index_date,
                queries=queries,
                source="template",
            )

        # Tier 2: Try LLM generation
        if self._ai_provider:
            try:
                queries = await self._generate_via_llm(
                    diagnosis.term,
                    episode_context,
                )
                if queries:
                    logger.debug(
                        "LLM queries for %r: %d queries",
                        diagnosis.term,
                        len(queries),
                    )
                    return DiagnosisQueries(
                        diagnosis_term=diagnosis.term,
                        concept_id=diagnosis.concept_id,
                        index_date=index_date,
                        queries=queries,
                        source="llm",
                    )
            except Exception as e:
                logger.warning(
                    "LLM query generation failed for %r: %s, using defaults",
                    diagnosis.term,
                    e,
                )

        # Tier 3: Default generic queries
        queries = generate_default_queries(diagnosis.term, self._max_queries)
        logger.debug(
            "Default queries for %r: %d queries",
            diagnosis.term,
            len(queries),
        )
        return DiagnosisQueries(
            diagnosis_term=diagnosis.term,
            concept_id=diagnosis.concept_id,
            index_date=index_date,
            queries=queries,
            source="default",
        )

    async def _generate_via_llm(
        self,
        diagnosis_term: str,
        context: str,
    ) -> list[str]:
        """Use the LLM to generate search queries for a diagnosis."""
        prompt = QUERY_GENERATION_PROMPT.format(
            max_queries=self._max_queries,
            diagnosis=diagnosis_term,
            context=context,
        )
        response = await self._ai_provider.chat_simple(
            prompt,
            temperature=0.3,
        )

        # Parse response: one query per line, skip empty lines
        queries = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]

        # Limit to max_queries
        return queries[: self._max_queries]

    def _build_context(self, episode) -> str:
        """Build context string from an episode's treatments and referrals."""
        parts = []
        if episode.treatments:
            treatments = ", ".join(t.term for t in episode.treatments)
            parts.append(f"Treatments: {treatments}")
        if episode.referrals:
            referrals = ", ".join(r.term for r in episode.referrals)
            parts.append(f"Referrals: {referrals}")
        if episode.investigations:
            investigations = ", ".join(i.term for i in episode.investigations)
            parts.append(f"Investigations: {investigations}")
        return "; ".join(parts) if parts else "No additional context"
