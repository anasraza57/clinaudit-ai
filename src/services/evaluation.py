"""
LLM-as-Judge evaluation service.

Evaluates each pipeline agent's output quality using either
weak supervision (extractor) or a separate LLM judge call
(query generator, retriever, scorer).

Evaluation methods:
- Extractor: SNOMED rules as pseudo-ground-truth (no LLM needed)
- Query Generator: LLM rates relevance (1-5) and coverage (1-5)
- Retriever: LLM rates guideline relevance (1-5)
- Scorer: LLM rates reasoning quality, citation accuracy,
          and score calibration (all 1-5)
"""

import logging
import math
import re
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.extractor import ExtractionResult
from src.agents.query import QueryResult
from src.agents.retriever import RetrievalResult
from src.agents.scorer import DiagnosisScore, ScoringResult
from src.ai.base import AIProvider
from src.models.patient import ClinicalEntry
from src.services.pipeline import PipelineResult
from src.services.snomed_categoriser import categorise_by_rules

logger = logging.getLogger(__name__)


# ── Rating parser ────────────────────────────────────────────────────

_RATING_PATTERN = re.compile(r":\s*(\d+)")


def _parse_rating(text: str, field_name: str) -> int:
    """Extract a 1-5 integer rating for a named field from LLM response.

    Scans each line of *text* for one containing *field_name*
    (case-insensitive), then pulls the first integer after a colon.
    Clamps to [1, 5]. Returns 3 if nothing matches.
    """
    for line in text.splitlines():
        if field_name.lower() in line.lower():
            match = _RATING_PATTERN.search(line)
            if match:
                val = int(match.group(1))
                return max(1, min(5, val))
    return 3


# ── Evaluation dataclasses ───────────────────────────────────────────


@dataclass
class ExtractorMetrics:
    """Weak-supervision metrics for the extractor agent."""

    total_entries: int = 0
    rule_matched: int = 0
    rule_match_rate: float = 0.0
    per_category: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "rule_matched": self.rule_matched,
            "rule_match_rate": round(self.rule_match_rate, 4),
            "per_category": self.per_category,
        }


@dataclass
class QueryMetrics:
    """LLM-as-Judge metrics for the query generator."""

    total_diagnoses: int = 0
    mean_relevance: float = 0.0
    mean_coverage: float = 0.0
    per_diagnosis: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_diagnoses": self.total_diagnoses,
            "mean_relevance": round(self.mean_relevance, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "per_diagnosis": self.per_diagnosis,
        }


@dataclass
class RetrieverMetrics:
    """LLM-as-Judge metrics for the retriever agent."""

    total_diagnoses: int = 0
    total_guidelines: int = 0
    mean_relevance: float = 0.0
    per_diagnosis: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_diagnoses": self.total_diagnoses,
            "total_guidelines": self.total_guidelines,
            "mean_relevance": round(self.mean_relevance, 4),
            "per_diagnosis": self.per_diagnosis,
        }


@dataclass
class ScorerMetrics:
    """LLM-as-Judge metrics for the scorer agent."""

    total_diagnoses: int = 0
    mean_reasoning_quality: float = 0.0
    mean_citation_accuracy: float = 0.0
    mean_score_calibration: float = 0.0
    per_diagnosis: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_diagnoses": self.total_diagnoses,
            "mean_reasoning_quality": round(self.mean_reasoning_quality, 4),
            "mean_citation_accuracy": round(self.mean_citation_accuracy, 4),
            "mean_score_calibration": round(self.mean_score_calibration, 4),
            "per_diagnosis": self.per_diagnosis,
        }


@dataclass
class PipelineEvaluation:
    """Complete evaluation metrics for one patient."""

    pat_id: str
    extractor: ExtractorMetrics | None = None
    query: QueryMetrics | None = None
    retriever: RetrieverMetrics | None = None
    scorer: ScorerMetrics | None = None

    def summary(self) -> dict:
        result: dict = {"pat_id": self.pat_id}
        if self.extractor:
            result["extractor"] = self.extractor.summary()
        if self.query:
            result["query"] = self.query.summary()
        if self.retriever:
            result["retriever"] = self.retriever.summary()
        if self.scorer:
            result["scorer"] = self.scorer.summary()
        return result


@dataclass
class AggregateEvaluation:
    """Aggregated evaluation across multiple patients."""

    total_patients: int = 0
    extractor: ExtractorMetrics | None = None
    query: QueryMetrics | None = None
    retriever: RetrieverMetrics | None = None
    scorer: ScorerMetrics | None = None
    per_patient: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        result: dict = {"total_patients": self.total_patients}
        if self.extractor:
            result["extractor"] = self.extractor.summary()
        if self.query:
            result["query"] = self.query.summary()
        if self.retriever:
            result["retriever"] = self.retriever.summary()
        if self.scorer:
            result["scorer"] = self.scorer.summary()
        result["per_patient"] = self.per_patient
        return result


# ── Judge prompts ────────────────────────────────────────────────────

QUERY_JUDGE_PROMPT = """\
You are evaluating search queries generated for a clinical guideline audit.

Diagnosis: {diagnosis}
Index Date: {index_date}

Generated queries:
{queries}

Rate the following on a scale of 1-5:
1. Relevance: How relevant are these queries to finding NICE clinical \
guidelines for managing this diagnosis? (1=irrelevant, 5=highly targeted)
2. Coverage: How well do the queries cover key clinical management aspects \
(treatment, referral, investigation, red flags)? (1=major gaps, 5=comprehensive)

Format your response EXACTLY as:
Relevance: [1-5]
Coverage: [1-5]"""

RETRIEVER_JUDGE_PROMPT = """\
You are evaluating clinical guidelines retrieved for a specific diagnosis.

Diagnosis: {diagnosis}

Retrieved guidelines:
{guidelines}

Rate the overall relevance of these guidelines to managing this diagnosis \
on a scale of 1-5:
(1=completely irrelevant, 2=mostly irrelevant, 3=somewhat relevant, \
4=mostly relevant, 5=highly relevant and specific)

Format your response EXACTLY as:
Relevance: [1-5]"""

SCORER_JUDGE_PROMPT = """\
You are evaluating the quality of a clinical audit scoring decision.

Diagnosis: {diagnosis}
Score: {score} ({judgement})
Explanation: {explanation}
Cited Guideline: {cited_guideline}
Guidelines Followed: {guidelines_followed}
Guidelines Not Followed: {guidelines_not_followed}

Rate the following on a scale of 1-5:
1. Reasoning Quality: Is the explanation logical, clear, and does it justify \
the assigned score? (1=incoherent, 5=excellent reasoning)
2. Citation Accuracy: Does the cited guideline text support the conclusion? \
Is it relevant? (1=irrelevant/missing, 5=precise)
3. Score Calibration: Is the assigned score appropriate given the documented \
evidence? (1=clearly wrong, 5=perfectly calibrated)

Format your response EXACTLY as:
Reasoning Quality: [1-5]
Citation Accuracy: [1-5]
Score Calibration: [1-5]"""


# ── Evaluation functions ─────────────────────────────────────────────


def evaluate_extractor(
    extraction: ExtractionResult,
    raw_entries: list[dict],
) -> ExtractorMetrics:
    """
    Evaluate extractor using SNOMED rules as pseudo-ground-truth.

    Compares rule-based categories against the extractor's assigned
    categories. Computes per-category precision/recall/F1 and overall
    rule_match_rate. No LLM call needed.
    """
    if not raw_entries:
        return ExtractorMetrics()

    # Build rule-based ground truth
    rule_labels: dict[str, str] = {}
    for entry in raw_entries:
        concept = entry.get("concept_display", "")
        if concept:
            rule_cat = categorise_by_rules(concept)
            if rule_cat:
                rule_labels[concept] = rule_cat

    # Build extractor labels from extraction result
    extractor_labels: dict[str, str] = {}
    for episode in extraction.episodes:
        for entry in episode.entries:
            extractor_labels[entry.concept_display] = entry.category

    # Compare only entries where rules produced a label
    total = 0
    matched = 0
    cat_tp: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}

    for concept, rule_cat in rule_labels.items():
        ext_cat = extractor_labels.get(concept)
        if ext_cat is None:
            continue
        total += 1
        if ext_cat == rule_cat:
            matched += 1
            cat_tp[rule_cat] = cat_tp.get(rule_cat, 0) + 1
        else:
            cat_fp[ext_cat] = cat_fp.get(ext_cat, 0) + 1
            cat_fn[rule_cat] = cat_fn.get(rule_cat, 0) + 1

    # Compute per-category P/R/F1
    per_category: dict[str, dict] = {}
    all_cats = set(
        list(cat_tp.keys()) + list(cat_fp.keys()) + list(cat_fn.keys()),
    )
    for cat in sorted(all_cats):
        tp = cat_tp.get(cat, 0)
        fp = cat_fp.get(cat, 0)
        fn = cat_fn.get(cat, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return ExtractorMetrics(
        total_entries=total,
        rule_matched=matched,
        rule_match_rate=round(matched / total, 4) if total > 0 else 0.0,
        per_category=per_category,
    )


async def evaluate_queries(
    query_result: QueryResult,
    ai_provider: AIProvider,
) -> QueryMetrics:
    """
    Evaluate query generator using LLM-as-Judge.

    For each diagnosis, sends the generated queries to a separate LLM
    call that rates relevance (1-5) and coverage (1-5).
    """
    if not query_result.diagnosis_queries:
        return QueryMetrics()

    per_diagnosis: list[dict] = []
    all_relevance: list[int] = []
    all_coverage: list[int] = []

    for dq in query_result.diagnosis_queries:
        if not dq.queries:
            continue

        prompt = QUERY_JUDGE_PROMPT.format(
            diagnosis=dq.diagnosis_term,
            index_date=dq.index_date,
            queries="\n".join(f"- {q}" for q in dq.queries),
        )

        try:
            response = await ai_provider.chat_simple(prompt, temperature=0.0)
            relevance = _parse_rating(response, "Relevance")
            coverage = _parse_rating(response, "Coverage")
        except Exception:
            logger.warning(
                "Judge call failed for queries on %s", dq.diagnosis_term,
            )
            relevance = 3
            coverage = 3

        all_relevance.append(relevance)
        all_coverage.append(coverage)
        per_diagnosis.append({
            "diagnosis": dq.diagnosis_term,
            "num_queries": len(dq.queries),
            "source": dq.source,
            "relevance": relevance,
            "coverage": coverage,
        })

    n = len(all_relevance)
    return QueryMetrics(
        total_diagnoses=n,
        mean_relevance=round(sum(all_relevance) / n, 4) if n > 0 else 0.0,
        mean_coverage=round(sum(all_coverage) / n, 4) if n > 0 else 0.0,
        per_diagnosis=per_diagnosis,
    )


async def evaluate_retrieval(
    retrieval: RetrievalResult,
    ai_provider: AIProvider,
) -> RetrieverMetrics:
    """
    Evaluate retriever using LLM-as-Judge.

    For each diagnosis, sends the retrieved guidelines to a separate
    LLM call that rates overall relevance (1-5).
    """
    if not retrieval.diagnosis_guidelines:
        return RetrieverMetrics()

    per_diagnosis: list[dict] = []
    all_relevance: list[int] = []
    total_guidelines = 0

    for dg in retrieval.diagnosis_guidelines:
        if not dg.guidelines:
            continue

        total_guidelines += len(dg.guidelines)
        guideline_text = "\n".join(
            f"{i + 1}. {g.title} — {g.clean_text[:200]}..."
            for i, g in enumerate(dg.guidelines)
        )

        prompt = RETRIEVER_JUDGE_PROMPT.format(
            diagnosis=dg.diagnosis_term,
            guidelines=guideline_text,
        )

        try:
            response = await ai_provider.chat_simple(prompt, temperature=0.0)
            relevance = _parse_rating(response, "Relevance")
        except Exception:
            logger.warning(
                "Judge call failed for retrieval on %s", dg.diagnosis_term,
            )
            relevance = 3

        all_relevance.append(relevance)
        per_diagnosis.append({
            "diagnosis": dg.diagnosis_term,
            "num_guidelines": len(dg.guidelines),
            "relevance": relevance,
        })

    n = len(all_relevance)
    return RetrieverMetrics(
        total_diagnoses=n,
        total_guidelines=total_guidelines,
        mean_relevance=round(sum(all_relevance) / n, 4) if n > 0 else 0.0,
        per_diagnosis=per_diagnosis,
    )


async def evaluate_scoring(
    scoring: ScoringResult,
    ai_provider: AIProvider,
) -> ScorerMetrics:
    """
    Evaluate scorer using LLM-as-Judge.

    For each diagnosis (skipping errors), sends the scoring decision
    to a separate LLM call that rates reasoning quality (1-5),
    citation accuracy (1-5), and score calibration (1-5).
    """
    if not scoring.diagnosis_scores:
        return ScorerMetrics()

    per_diagnosis: list[dict] = []
    all_reasoning: list[int] = []
    all_citation: list[int] = []
    all_calibration: list[int] = []

    for ds in scoring.diagnosis_scores:
        if ds.error:
            continue

        prompt = SCORER_JUDGE_PROMPT.format(
            diagnosis=ds.diagnosis_term,
            score=ds.score,
            judgement=ds.judgement,
            explanation=ds.explanation,
            cited_guideline=ds.cited_guideline_text or "None",
            guidelines_followed=", ".join(ds.guidelines_followed) or "None",
            guidelines_not_followed=(
                ", ".join(ds.guidelines_not_followed) or "None"
            ),
        )

        try:
            response = await ai_provider.chat_simple(prompt, temperature=0.0)
            reasoning = _parse_rating(response, "Reasoning Quality")
            citation = _parse_rating(response, "Citation Accuracy")
            calibration = _parse_rating(response, "Score Calibration")
        except Exception:
            logger.warning(
                "Judge call failed for scoring on %s", ds.diagnosis_term,
            )
            reasoning = 3
            citation = 3
            calibration = 3

        all_reasoning.append(reasoning)
        all_citation.append(citation)
        all_calibration.append(calibration)
        per_diagnosis.append({
            "diagnosis": ds.diagnosis_term,
            "score": ds.score,
            "judgement": ds.judgement,
            "reasoning_quality": reasoning,
            "citation_accuracy": citation,
            "score_calibration": calibration,
        })

    n = len(all_reasoning)
    return ScorerMetrics(
        total_diagnoses=n,
        mean_reasoning_quality=(
            round(sum(all_reasoning) / n, 4) if n > 0 else 0.0
        ),
        mean_citation_accuracy=(
            round(sum(all_citation) / n, 4) if n > 0 else 0.0
        ),
        mean_score_calibration=(
            round(sum(all_calibration) / n, 4) if n > 0 else 0.0
        ),
        per_diagnosis=per_diagnosis,
    )


# ── Orchestrators ────────────────────────────────────────────────────


async def evaluate_patient(
    pipeline_result: PipelineResult,
    raw_entries: list[dict],
    ai_provider: AIProvider,
    agents: list[str] | None = None,
) -> PipelineEvaluation:
    """
    Evaluate all (or selected) pipeline agents for a single patient.

    Requires a ``PipelineResult`` with intermediate results and the
    original raw clinical entries (for extractor weak supervision).

    Args:
        pipeline_result: Complete pipeline run with intermediate results.
        raw_entries: Original clinical entries for weak supervision.
        ai_provider: AI provider for LLM-as-Judge calls.
        agents: Which agents to evaluate.
            Default ``["extractor", "query", "retriever", "scorer"]``.
    """
    if agents is None:
        agents = ["extractor", "query", "retriever", "scorer"]

    evaluation = PipelineEvaluation(pat_id=pipeline_result.pat_id)

    if "extractor" in agents and pipeline_result.extraction:
        evaluation.extractor = evaluate_extractor(
            pipeline_result.extraction, raw_entries,
        )

    if "query" in agents and pipeline_result.query_result:
        evaluation.query = await evaluate_queries(
            pipeline_result.query_result, ai_provider,
        )

    if "retriever" in agents and pipeline_result.retrieval:
        evaluation.retriever = await evaluate_retrieval(
            pipeline_result.retrieval, ai_provider,
        )

    if "scorer" in agents and pipeline_result.scoring:
        evaluation.scorer = await evaluate_scoring(
            pipeline_result.scoring, ai_provider,
        )

    return evaluation


def aggregate_evaluations(
    evaluations: list[PipelineEvaluation],
) -> AggregateEvaluation:
    """Aggregate per-patient evaluations into overall metrics."""
    if not evaluations:
        return AggregateEvaluation()

    ext_entries, ext_matched = 0, 0
    ext_cat_tp: dict[str, int] = {}
    ext_cat_fp: dict[str, int] = {}
    ext_cat_fn: dict[str, int] = {}

    query_relevance: list[int] = []
    query_coverage: list[int] = []

    ret_relevance: list[int] = []
    ret_guidelines = 0

    scorer_reasoning: list[int] = []
    scorer_citation: list[int] = []
    scorer_calibration: list[int] = []

    for ev in evaluations:
        if ev.extractor:
            ext_entries += ev.extractor.total_entries
            ext_matched += ev.extractor.rule_matched
            for cat, metrics in ev.extractor.per_category.items():
                ext_cat_tp[cat] = (
                    ext_cat_tp.get(cat, 0) + metrics.get("tp", 0)
                )
                ext_cat_fp[cat] = (
                    ext_cat_fp.get(cat, 0) + metrics.get("fp", 0)
                )
                ext_cat_fn[cat] = (
                    ext_cat_fn.get(cat, 0) + metrics.get("fn", 0)
                )

        if ev.query:
            for d in ev.query.per_diagnosis:
                query_relevance.append(d["relevance"])
                query_coverage.append(d["coverage"])

        if ev.retriever:
            ret_guidelines += ev.retriever.total_guidelines
            for d in ev.retriever.per_diagnosis:
                ret_relevance.append(d["relevance"])

        if ev.scorer:
            for d in ev.scorer.per_diagnosis:
                scorer_reasoning.append(d["reasoning_quality"])
                scorer_citation.append(d["citation_accuracy"])
                scorer_calibration.append(d["score_calibration"])

    agg = AggregateEvaluation(
        total_patients=len(evaluations),
        per_patient=[ev.summary() for ev in evaluations],
    )

    # Extractor aggregate
    if ext_entries > 0:
        agg_per_cat: dict[str, dict] = {}
        all_cats = set(
            list(ext_cat_tp.keys())
            + list(ext_cat_fp.keys())
            + list(ext_cat_fn.keys()),
        )
        for cat in sorted(all_cats):
            tp = ext_cat_tp.get(cat, 0)
            fp = ext_cat_fp.get(cat, 0)
            fn = ext_cat_fn.get(cat, 0)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if tp + fp + fn > 0:
                agg_per_cat[cat] = {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
        agg.extractor = ExtractorMetrics(
            total_entries=ext_entries,
            rule_matched=ext_matched,
            rule_match_rate=round(ext_matched / ext_entries, 4),
            per_category=agg_per_cat,
        )

    # Query aggregate
    n_q = len(query_relevance)
    if n_q > 0:
        agg.query = QueryMetrics(
            total_diagnoses=n_q,
            mean_relevance=round(sum(query_relevance) / n_q, 4),
            mean_coverage=round(sum(query_coverage) / n_q, 4),
        )

    # Retriever aggregate
    n_r = len(ret_relevance)
    if n_r > 0:
        agg.retriever = RetrieverMetrics(
            total_diagnoses=n_r,
            total_guidelines=ret_guidelines,
            mean_relevance=round(sum(ret_relevance) / n_r, 4),
        )

    # Scorer aggregate
    n_s = len(scorer_reasoning)
    if n_s > 0:
        agg.scorer = ScorerMetrics(
            total_diagnoses=n_s,
            mean_reasoning_quality=round(
                sum(scorer_reasoning) / n_s, 4,
            ),
            mean_citation_accuracy=round(
                sum(scorer_citation) / n_s, 4,
            ),
            mean_score_calibration=round(
                sum(scorer_calibration) / n_s, 4,
            ),
        )

    return agg


# ── Stored-data helpers ──────────────────────────────────────────────


def scoring_from_stored(details: dict) -> ScoringResult:
    """Reconstruct a ``ScoringResult`` from stored ``details_json`` dict.

    Useful for evaluating scorer quality from previously completed
    audit results without re-running the pipeline.
    """
    scores: list[DiagnosisScore] = []
    for ds in details.get("scores", []):
        scores.append(
            DiagnosisScore(
                diagnosis_term=ds.get("diagnosis", "Unknown"),
                concept_id=ds.get("concept_id", ""),
                index_date=ds.get("index_date", ""),
                score=ds.get("score", 0),
                judgement=ds.get("judgement", ""),
                explanation=ds.get("explanation", ""),
                confidence=ds.get("confidence", 0.0),
                cited_guideline_text=ds.get("cited_guideline_text", ""),
                guidelines_followed=ds.get("guidelines_followed", []),
                guidelines_not_followed=ds.get("guidelines_not_followed", []),
                missing_care_opportunities=ds.get(
                    "missing_care_opportunities", [],
                ),
                error=ds.get("error"),
            ),
        )
    return ScoringResult(
        pat_id=details.get("pat_id", "Unknown"),
        diagnosis_scores=scores,
        total_diagnoses=details.get("total_diagnoses", len(scores)),
    )


# ── DB-based extractor evaluation ────────────────────────────────────


async def evaluate_extractor_from_db(
    session: AsyncSession,
    sample_size: int | None = None,
) -> dict:
    """
    Evaluate extractor quality using stored categories vs rule-based ground truth.

    Loads clinical entries with their LLM-assigned ``category`` column,
    runs ``categorise_by_rules()`` on each concept as pseudo-ground-truth,
    and computes per-category precision/recall/F1. No LLM calls needed.
    """
    query = (
        select(ClinicalEntry.concept_display, ClinicalEntry.category)
        .where(ClinicalEntry.category.isnot(None))
        .distinct()
    )
    if sample_size is not None:
        query = query.limit(sample_size)

    result = await session.execute(query)
    rows = list(result.all())

    total_concepts = len(rows)
    total_with_rules = 0
    matched = 0
    cat_tp: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}
    category_distribution: dict[str, int] = {}

    for concept_display, stored_category in rows:
        # Track distribution of stored categories
        category_distribution[stored_category] = (
            category_distribution.get(stored_category, 0) + 1
        )

        # Get rule-based ground truth
        rule_cat = categorise_by_rules(concept_display)
        if rule_cat is None:
            continue

        total_with_rules += 1
        if stored_category == rule_cat:
            matched += 1
            cat_tp[rule_cat] = cat_tp.get(rule_cat, 0) + 1
        else:
            cat_fp[stored_category] = cat_fp.get(stored_category, 0) + 1
            cat_fn[rule_cat] = cat_fn.get(rule_cat, 0) + 1

    # Per-category P/R/F1
    per_category: dict[str, dict] = {}
    all_cats = sorted(set(list(cat_tp) + list(cat_fp) + list(cat_fn)))
    for cat in all_cats:
        tp = cat_tp.get(cat, 0)
        fp = cat_fp.get(cat, 0)
        fn = cat_fn.get(cat, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    rule_match_rate = matched / total_with_rules if total_with_rules > 0 else 0.0

    return {
        "total_concepts": total_concepts,
        "total_with_rules": total_with_rules,
        "rule_match_rate": round(rule_match_rate, 4),
        "per_category": per_category,
        "category_distribution": category_distribution,
    }


# ── Retriever IR metrics ─────────────────────────────────────────────


RETRIEVER_PER_GUIDELINE_JUDGE_PROMPT = """\
You are evaluating whether a specific clinical guideline is relevant \
to managing a diagnosis.

Diagnosis: {diagnosis}

Guideline title: {title}
Guideline text (excerpt): {text}

Rate the relevance of this guideline to managing this diagnosis \
on a scale of 1-5:
(1=completely irrelevant, 2=mostly irrelevant, 3=somewhat relevant, \
4=mostly relevant, 5=highly relevant and specific)

Format your response EXACTLY as:
Relevance: [1-5]"""


async def evaluate_retrieval_ir(
    retrieval: RetrievalResult,
    ai_provider: AIProvider,
    relevance_threshold: int = 3,
) -> dict:
    """
    Evaluate retriever using per-guideline LLM-as-Judge ratings.

    Rates each retrieved guideline individually and computes standard
    IR metrics: Precision@k, nDCG@k, MRR. Uses the LLM-as-Judge
    relevance rating as proxy ground truth.
    """
    if not retrieval.diagnosis_guidelines:
        return {
            "total_diagnoses": 0, "total_guidelines_rated": 0,
            "mean_precision_at_k": 0.0, "mean_recall_at_k": 0.0,
            "mean_ndcg": 0.0, "mean_mrr": 0.0, "mean_relevance": 0.0,
            "per_diagnosis": [],
        }

    per_diagnosis: list[dict] = []
    all_precision: list[float] = []
    all_recall: list[float] = []
    all_ndcg: list[float] = []
    all_mrr: list[float] = []
    all_relevance: list[int] = []
    total_guidelines_rated = 0

    for dg in retrieval.diagnosis_guidelines:
        if not dg.guidelines:
            continue

        guideline_ratings: list[dict] = []
        ratings: list[int] = []

        for g in sorted(dg.guidelines, key=lambda x: x.rank):
            prompt = RETRIEVER_PER_GUIDELINE_JUDGE_PROMPT.format(
                diagnosis=dg.diagnosis_term,
                title=g.title,
                text=g.clean_text[:300],
            )
            try:
                response = await ai_provider.chat_simple(prompt, temperature=0.0)
                rating = _parse_rating(response, "Relevance")
            except Exception:
                logger.warning(
                    "Judge call failed for guideline %s on %s",
                    g.title, dg.diagnosis_term,
                )
                rating = 3

            ratings.append(rating)
            all_relevance.append(rating)
            total_guidelines_rated += 1
            guideline_ratings.append({
                "title": g.title,
                "relevance": rating,
                "relevant": rating >= relevance_threshold,
            })

        k = len(ratings)
        relevant_binary = [1 if r >= relevance_threshold else 0 for r in ratings]
        num_relevant = sum(relevant_binary)

        # Precision@k
        precision_at_k = num_relevant / k if k > 0 else 0.0

        # Recall@k (all retrieved, so recall = 1.0 if any relevant exist)
        recall_at_k = 1.0 if num_relevant > 0 else 0.0

        # MRR — 1/rank of first relevant result
        mrr = 0.0
        for i, rel in enumerate(relevant_binary):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break

        # nDCG — using binary relevance
        dcg = sum(
            rel / math.log2(i + 2)  # i+2 because log2(1) = 0
            for i, rel in enumerate(relevant_binary)
        )
        ideal = sorted(relevant_binary, reverse=True)
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        all_precision.append(precision_at_k)
        all_recall.append(recall_at_k)
        all_ndcg.append(ndcg)
        all_mrr.append(mrr)

        per_diagnosis.append({
            "diagnosis": dg.diagnosis_term,
            "num_guidelines": k,
            "precision_at_k": round(precision_at_k, 4),
            "recall_at_k": round(recall_at_k, 4),
            "ndcg": round(ndcg, 4),
            "mrr": round(mrr, 4),
            "per_guideline": guideline_ratings,
        })

    n = len(all_precision)
    return {
        "total_diagnoses": n,
        "total_guidelines_rated": total_guidelines_rated,
        "mean_precision_at_k": round(sum(all_precision) / n, 4) if n else 0.0,
        "mean_recall_at_k": round(sum(all_recall) / n, 4) if n else 0.0,
        "mean_ndcg": round(sum(all_ndcg) / n, 4) if n else 0.0,
        "mean_mrr": round(sum(all_mrr) / n, 4) if n else 0.0,
        "mean_relevance": (
            round(sum(all_relevance) / len(all_relevance), 4)
            if all_relevance else 0.0
        ),
        "per_diagnosis": per_diagnosis,
    }


# ── Full agent evaluation orchestrator ───────────────────────────────


async def run_agent_evaluation(
    session: AsyncSession,
    ai_provider: AIProvider,
    limit: int = 5,
) -> dict:
    """
    Run the full pipeline for a sample of patients and evaluate all 4 agents.

    Expensive: each patient runs the complete pipeline plus LLM-as-Judge
    calls for query, retriever (per-guideline), and scorer evaluation.
    """
    from src.models.patient import Patient
    from src.services.pipeline import AuditPipeline

    # Pick random patients
    q = select(Patient.pat_id).order_by(func.random()).limit(limit)
    result = await session.execute(q)
    pat_ids = [row[0] for row in result.all()]

    if not pat_ids:
        return {"total_patients": 0, "per_patient": []}

    from src.services.embedder import get_embedder
    from src.services.vector_store import get_vector_store

    pipeline = AuditPipeline(
        ai_provider=ai_provider,
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )
    await pipeline.load_categories_from_db(session)

    evaluations: list[PipelineEvaluation] = []
    retriever_ir_metrics: list[dict] = []

    for pat_id in pat_ids:
        try:
            pipeline_result = await pipeline.run_single(session, pat_id)
        except Exception:
            logger.warning("Pipeline failed for %s during evaluation", pat_id)
            continue

        if not pipeline_result.success:
            continue

        # Load raw entries for extractor evaluation
        raw_entries = await pipeline._load_patient_entries(session, pat_id)

        # Standard agent evaluations
        evaluation = await evaluate_patient(
            pipeline_result, raw_entries, ai_provider,
        )
        evaluations.append(evaluation)

        # Enhanced retriever IR metrics
        if pipeline_result.retrieval:
            ir = await evaluate_retrieval_ir(
                pipeline_result.retrieval, ai_provider,
            )
            retriever_ir_metrics.append(ir)

    # Aggregate standard metrics
    agg = aggregate_evaluations(evaluations)
    result_dict = agg.summary()

    # Aggregate retriever IR metrics
    if retriever_ir_metrics:
        n = len(retriever_ir_metrics)
        result_dict["retriever_ir"] = {
            "mean_precision_at_k": round(
                sum(m["mean_precision_at_k"] for m in retriever_ir_metrics) / n, 4,
            ),
            "mean_recall_at_k": round(
                sum(m["mean_recall_at_k"] for m in retriever_ir_metrics) / n, 4,
            ),
            "mean_ndcg": round(
                sum(m["mean_ndcg"] for m in retriever_ir_metrics) / n, 4,
            ),
            "mean_mrr": round(
                sum(m["mean_mrr"] for m in retriever_ir_metrics) / n, 4,
            ),
            "mean_relevance": round(
                sum(m["mean_relevance"] for m in retriever_ir_metrics) / n, 4,
            ),
        }

    return result_dict
