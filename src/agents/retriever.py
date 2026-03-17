"""
Guideline Evidence Finder — Stage 3 of the audit pipeline.

Takes the QueryResult from the Audit Query Generator, embeds each
search query using PubMedBERT, searches the FAISS guideline index,
and returns the most relevant guideline passages for each diagnosis.

The Guideline Evidence Finder's job:
1. Receive queries (1-3 per diagnosis) from the Audit Query Generator
2. Encode each query into a 768-dim vector via PubMedBERT
3. Search the FAISS index for top-K most similar guidelines
4. Aggregate and deduplicate results across queries for each diagnosis
5. Return structured RetrievalResult for the Compliance Auditor Agent
"""

import logging
from dataclasses import dataclass, field

from src.agents.query import DiagnosisQueries, QueryResult
from src.config.settings import get_settings
from src.services.embedder import Embedder
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Relevance filtering ─────────────────────────────────────────────
# Body-region and condition keyword groups used to check whether a
# retrieved guideline title is topically relevant to a diagnosis.
# Each key is a "topic tag" and the value is a set of keywords that
# appear in diagnosis terms OR guideline titles for that topic.

_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "spine": {"back", "lumbar", "spinal", "spine", "sciatica", "radiculopathy", "stenosis", "spondylosis", "lumbago", "thoracic", "cervical", "disc"},
    "knee": {"knee", "patella", "patellar", "meniscal", "meniscus", "cruciate", "acl"},
    "hip": {"hip", "femoral", "acetabular", "groin"},
    "shoulder": {"shoulder", "rotator", "supraspinatus", "subacromial", "impingement"},
    "hand_wrist": {"hand", "wrist", "carpal", "tunnel", "finger", "thumb", "dupuytren"},
    "foot_ankle": {"foot", "ankle", "plantar", "heel", "toe", "metatarsal", "bunion", "hallux"},
    "elbow": {"elbow", "epicondylitis", "epicondyle", "olecranon", "tennis", "golfer"},
    "neck": {"neck", "cervical", "whiplash"},
    "osteoarthritis": {"osteoarthritis", "arthritis", "arthrosis", "joint replacement"},
    "fracture": {"fracture", "broken", "fractures"},
    "inflammatory": {"rheumatoid", "gout", "inflammatory", "arthritis", "crystal"},
    "pain": {"pain", "chronic pain", "fibromyalgia", "widespread"},
    "osteoporosis": {"osteoporosis", "bone density", "fragility", "bisphosphonate"},
}

# Guideline titles containing these terms are almost certainly irrelevant
# to MSK audit diagnoses (oncology drugs, specialist procedures, etc.).
_EXCLUDE_TITLE_TERMS: set[str] = {
    "cancer", "carcinoma", "lymphoma", "leukaemia", "leukemia",
    "melanoma", "tumour", "tumor", "chemotherapy", "radiotherapy",
    "hepatitis", "hiv", "tuberculosis",
    "diabetes", "diabetic",  # e.g. "diabetic foot" ≠ generic foot pain
    "cardiac", "heart failure", "atrial fibrillation",
    "chest pain",
    "dementia", "alzheimer",
    "pregnancy", "antenatal", "postnatal",
    "neonatal", "infant",
}


def _tokenise(text: str) -> set[str]:
    """Lowercase split into word tokens."""
    return set(text.lower().split())


def _diagnosis_topics(diagnosis_term: str) -> set[str]:
    """Return the set of topic tags that match the diagnosis term."""
    tokens = _tokenise(diagnosis_term)
    topics: set[str] = set()
    for tag, keywords in _TOPIC_KEYWORDS.items():
        if tokens & keywords:
            topics.add(tag)
    return topics


def _title_topics(title: str) -> set[str]:
    """Return the set of topic tags that match a guideline title."""
    tokens = _tokenise(title)
    topics: set[str] = set()
    for tag, keywords in _TOPIC_KEYWORDS.items():
        if tokens & keywords:
            topics.add(tag)
    return topics


def _title_is_excluded(title: str) -> bool:
    """Check if a guideline title contains obviously irrelevant terms."""
    lower = title.lower()
    return any(term in lower for term in _EXCLUDE_TITLE_TERMS)


@dataclass
class GuidelineMatch:
    """A single guideline matched for a diagnosis."""

    guideline_id: str
    title: str
    source: str
    url: str
    clean_text: str
    score: float  # similarity score (higher = more similar after L2 norm)
    rank: int  # rank within this diagnosis's results
    matched_query: str  # the query that found this guideline


@dataclass
class DiagnosisGuidelines:
    """All retrieved guidelines for a single diagnosis."""

    diagnosis_term: str
    concept_id: str
    index_date: str
    guidelines: list[GuidelineMatch] = field(default_factory=list)

    @property
    def guideline_texts(self) -> list[str]:
        """Return just the guideline texts, ordered by rank."""
        return [g.clean_text for g in sorted(self.guidelines, key=lambda g: g.rank)]

    @property
    def guideline_titles(self) -> list[str]:
        """Return just the guideline titles, ordered by rank."""
        return [g.title for g in sorted(self.guidelines, key=lambda g: g.rank)]


@dataclass
class RetrievalResult:
    """The output of the Guideline Evidence Finder for one patient."""

    pat_id: str
    diagnosis_guidelines: list[DiagnosisGuidelines] = field(default_factory=list)
    total_diagnoses: int = 0
    total_guidelines: int = 0

    def summary(self) -> dict:
        return {
            "pat_id": self.pat_id,
            "total_diagnoses": self.total_diagnoses,
            "total_guidelines": self.total_guidelines,
            "diagnoses": [
                {
                    "diagnosis": dg.diagnosis_term,
                    "index_date": dg.index_date,
                    "num_guidelines": len(dg.guidelines),
                    "titles": dg.guideline_titles,
                }
                for dg in self.diagnosis_guidelines
            ],
        }


class GuidelineEvidenceFinder:
    """
    Retrieves relevant NICE guidelines for each diagnosis using
    PubMedBERT embeddings and FAISS similarity search.

    Usage:
        agent = GuidelineEvidenceFinder(embedder=embedder, vector_store=store)
        retrieval = agent.retrieve(query_result)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        settings = get_settings()
        self._min_similarity = settings.retriever_min_similarity

    def retrieve(self, query_result: QueryResult) -> RetrievalResult:
        """
        Retrieve guidelines for all diagnoses in the QueryResult.

        For each diagnosis:
        1. Embed all queries (1-3) for that diagnosis
        2. Search FAISS for each query embedding
        3. Merge and deduplicate results (same guideline from different queries)
        4. Keep top-K unique guidelines, ranked by best score

        Args:
            query_result: The QueryResult from the Audit Query Generator.

        Returns:
            RetrievalResult with matched guidelines per diagnosis.
        """
        all_dg: list[DiagnosisGuidelines] = []

        # Cache retrieval results by diagnosis term — same term has identical
        # queries, embeddings, and FAISS results regardless of episode.
        retrieval_cache: dict[str, list[GuidelineMatch]] = {}

        # Track unique (term, index_date) to avoid duplicate output entries.
        seen_pairs: set[tuple[str, str]] = set()

        for dq in query_result.diagnosis_queries:
            pair_key = (dq.diagnosis_term, dq.index_date)
            if pair_key in seen_pairs:
                logger.debug(
                    "Skipping duplicate retrieval entry for %r (index_date=%s)",
                    dq.diagnosis_term, dq.index_date,
                )
                continue
            seen_pairs.add(pair_key)

            if dq.diagnosis_term in retrieval_cache:
                logger.debug(
                    "Reusing cached retrieval for %r (index_date=%s)",
                    dq.diagnosis_term, dq.index_date,
                )
                dg = DiagnosisGuidelines(
                    diagnosis_term=dq.diagnosis_term,
                    concept_id=dq.concept_id,
                    index_date=dq.index_date,
                    guidelines=retrieval_cache[dq.diagnosis_term],
                )
            else:
                dg = self._retrieve_for_diagnosis(dq)
                retrieval_cache[dq.diagnosis_term] = dg.guidelines

            all_dg.append(dg)

        total_guidelines = sum(len(dg.guidelines) for dg in all_dg)

        result = RetrievalResult(
            pat_id=query_result.pat_id,
            diagnosis_guidelines=all_dg,
            total_diagnoses=len(all_dg),
            total_guidelines=total_guidelines,
        )

        logger.info(
            "Retrieved %d guidelines for %d diagnoses (patient %s)",
            total_guidelines,
            len(all_dg),
            query_result.pat_id,
        )

        return result

    def _retrieve_for_diagnosis(
        self,
        dq: DiagnosisQueries,
    ) -> DiagnosisGuidelines:
        """
        Retrieve and merge guidelines for a single diagnosis.

        Multiple queries may return the same guideline — we keep the
        best (lowest distance / highest similarity) score for each.
        """
        # Collect all results across queries, keyed by guideline_id
        seen: dict[str, GuidelineMatch] = {}

        # Batch-encode all queries in a single forward pass (instead of N individual calls)
        logger.info(
            "Encoding %d queries for %r via PubMedBERT...",
            len(dq.queries), dq.diagnosis_term,
        )
        query_embeddings = self._embedder.encode_batch(dq.queries)
        logger.info(
            "Encoding complete for %r — shape %s, searching FAISS...",
            dq.diagnosis_term, query_embeddings.shape,
        )

        for i, query_text in enumerate(dq.queries):
            # Search FAISS with pre-computed embedding
            raw_results = self._vector_store.search(
                query_embeddings[i],
                top_k=self._top_k,
            )

            for result in raw_results:
                gid = result.get("id", "")
                score = result.get("score", float("inf"))

                # Keep the result with the best score for each guideline
                if gid not in seen or score < seen[gid].score:
                    seen[gid] = GuidelineMatch(
                        guideline_id=gid,
                        title=result.get("title", ""),
                        source=result.get("source", ""),
                        url=result.get("url", ""),
                        clean_text=result.get("clean_text", ""),
                        score=score,
                        rank=0,  # Will be set after sorting
                        matched_query=query_text,
                    )

        # Sort by score (lower = more similar for L2 distance) and assign ranks
        sorted_matches = sorted(seen.values(), key=lambda m: m.score)

        # Apply relevance filtering before truncating to top-K
        filtered = self._filter_irrelevant(dq.diagnosis_term, sorted_matches)
        top_matches = filtered[: self._top_k]

        for rank, match in enumerate(top_matches, start=1):
            match.rank = rank

        logger.debug(
            "Retrieved %d unique guidelines for %r (from %d queries, "
            "%d raw results, %d after filtering)",
            len(top_matches),
            dq.diagnosis_term,
            len(dq.queries),
            len(seen),
            len(filtered),
        )

        return DiagnosisGuidelines(
            diagnosis_term=dq.diagnosis_term,
            concept_id=dq.concept_id,
            index_date=dq.index_date,
            guidelines=top_matches,
        )

    def _filter_irrelevant(
        self,
        diagnosis_term: str,
        matches: list[GuidelineMatch],
    ) -> list[GuidelineMatch]:
        """
        Remove guidelines that are clearly irrelevant to the diagnosis.

        Two-layer filter:
        1. Title relevance — exclude guidelines from unrelated specialties
           or with no topical overlap with the diagnosis.
        2. Similarity threshold — exclude results whose L2 distance exceeds
           the configured maximum.

        If all guidelines are filtered out, returns the single best match
        as a fallback to avoid passing zero context to the scorer.
        """
        if not matches:
            return matches

        diag_topics = _diagnosis_topics(diagnosis_term)
        kept: list[GuidelineMatch] = []

        for match in matches:
            # Layer B: similarity threshold
            if match.score > self._min_similarity:
                logger.debug(
                    "Filtered (distance %.3f > %.1f): %r for diagnosis %r",
                    match.score, self._min_similarity,
                    match.title[:60], diagnosis_term,
                )
                continue

            # Layer A: title exclusion (obviously wrong specialty)
            if _title_is_excluded(match.title):
                logger.debug(
                    "Filtered (excluded title term): %r for diagnosis %r",
                    match.title[:60], diagnosis_term,
                )
                continue

            # Layer A: topic overlap check — only apply when the diagnosis
            # matches at least one known topic, so we don't accidentally
            # filter novel/rare conditions.
            if diag_topics:
                title_topics = _title_topics(match.title)
                # If the title also matches known topics but NONE overlap
                # with the diagnosis, it's likely a mismatch.
                if title_topics and not (diag_topics & title_topics):
                    logger.debug(
                        "Filtered (topic mismatch %s vs %s): %r for diagnosis %r",
                        diag_topics, title_topics,
                        match.title[:60], diagnosis_term,
                    )
                    continue

            kept.append(match)

        if not kept and matches:
            # Fallback: return single best match to avoid empty context
            logger.warning(
                "All guidelines filtered for %r — keeping best match as fallback: %r",
                diagnosis_term, matches[0].title[:60],
            )
            return [matches[0]]

        return kept
