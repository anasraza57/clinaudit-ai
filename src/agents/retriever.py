"""
Retriever Agent — Stage 3 of the audit pipeline.

Takes the QueryResult from the Query Agent, embeds each search query
using PubMedBERT, searches the FAISS guideline index, and returns
the most relevant guideline passages for each diagnosis.

The Retriever's job:
1. Receive queries (1-3 per diagnosis) from the Query Agent
2. Encode each query into a 768-dim vector via PubMedBERT
3. Search the FAISS index for top-K most similar guidelines
4. Aggregate and deduplicate results across queries for each diagnosis
5. Return structured RetrievalResult for the Scorer Agent
"""

import logging
from dataclasses import dataclass, field

from src.agents.query import DiagnosisQueries, QueryResult
from src.services.embedder import Embedder
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


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
    """The output of the Retriever Agent for one patient."""

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


class RetrieverAgent:
    """
    Retrieves relevant NICE guidelines for each diagnosis using
    PubMedBERT embeddings and FAISS similarity search.

    Usage:
        agent = RetrieverAgent(embedder=embedder, vector_store=store)
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

    def retrieve(self, query_result: QueryResult) -> RetrievalResult:
        """
        Retrieve guidelines for all diagnoses in the QueryResult.

        For each diagnosis:
        1. Embed all queries (1-3) for that diagnosis
        2. Search FAISS for each query embedding
        3. Merge and deduplicate results (same guideline from different queries)
        4. Keep top-K unique guidelines, ranked by best score

        Args:
            query_result: The QueryResult from the Query Agent.

        Returns:
            RetrievalResult with matched guidelines per diagnosis.
        """
        all_dg: list[DiagnosisGuidelines] = []

        for dq in query_result.diagnosis_queries:
            dg = self._retrieve_for_diagnosis(dq)
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
        top_matches = sorted_matches[: self._top_k]

        for rank, match in enumerate(top_matches, start=1):
            match.rank = rank

        logger.debug(
            "Retrieved %d unique guidelines for %r (from %d queries, %d raw results)",
            len(top_matches),
            dq.diagnosis_term,
            len(dq.queries),
            sum(1 for _ in seen.values()),
        )

        return DiagnosisGuidelines(
            diagnosis_term=dq.diagnosis_term,
            concept_id=dq.concept_id,
            index_date=dq.index_date,
            guidelines=top_matches,
        )
