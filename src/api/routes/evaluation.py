"""
Evaluation and comparison API routes.

Endpoints for comparing models across batch jobs, retrieving missing
care opportunities, gold-standard evaluation, and LLM-as-Judge runs.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditResult
from src.models.database import get_session
from src.services.comparison import compare_jobs, compute_cross_model_classification
from src.services.evaluation import (
    evaluate_extractor_from_db,
    evaluate_scoring,
    run_agent_evaluation,
    scoring_from_stored,
)
from src.services.reporting import compute_system_metrics, get_missing_care_summary

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


# ── Response schemas ──────────────────────────────────────────────────


class DiagnosisComparisonItem(BaseModel):
    diagnosis: str
    index_date: str | None
    score_a: int | None
    score_b: int | None
    judgement_a: str | None
    judgement_b: str | None
    agreement: bool


class PatientComparisonItem(BaseModel):
    pat_id: str
    score_a: float | None = Field(description="Overall score from job A")
    score_b: float | None = Field(description="Overall score from job B")
    score_diff: float | None = Field(description="score_a - score_b")
    diagnoses_a: int
    diagnoses_b: int
    agreement: bool = Field(description="All diagnoses agree in direction")
    per_diagnosis: list[DiagnosisComparisonItem] = []


class ConditionComparisonItem(BaseModel):
    condition: str
    count: int
    adherence_rate_a: float
    adherence_rate_b: float
    diff: float = Field(description="adherence_rate_a - adherence_rate_b")


class ComparisonResponse(BaseModel):
    job_a_id: int
    job_b_id: int
    job_a_provider: str | None = Field(description="AI provider used for job A")
    job_b_provider: str | None = Field(description="AI provider used for job B")
    total_patients_compared: int
    mean_score_a: float
    mean_score_b: float
    mean_abs_diff: float = Field(description="Mean absolute score difference")
    score_correlation: float = Field(description="Pearson correlation of patient scores")
    agreement_rate: float = Field(description="Fraction of diagnoses with same direction")
    cohen_kappa: float = Field(description="Inter-rater agreement (Cohen's kappa)")
    patients: list[PatientComparisonItem]
    per_condition: list[ConditionComparisonItem]


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare two batch jobs",
)
async def compare_models(
    job_a: int = Query(..., description="First job ID"),
    job_b: int = Query(..., description="Second job ID"),
    session: AsyncSession = Depends(get_session),
):
    """
    Compare audit results from two different batch jobs side-by-side.

    Typically used to compare different AI providers (e.g., OpenAI vs Ollama)
    on the same patient set. Returns per-patient score differences,
    agreement metrics (Cohen's kappa), and per-condition breakdown.
    """
    try:
        result = await compare_jobs(session, job_a, job_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return result.summary()


# ── Missing Care Opportunities schemas ────────────────────────────────


class MissingCareAction(BaseModel):
    action: str = Field(description="NICE-recommended action not documented")
    count: int = Field(description="How many times this gap was identified")


class MissingCareCondition(BaseModel):
    condition: str = Field(description="The diagnosis/condition")
    total_opportunities: int
    opportunities: list[MissingCareAction]


class MissingCareCase(BaseModel):
    pat_id: str
    diagnosis: str
    index_date: str | None
    score: int | None
    missing_care_opportunities: list[str]


class MissingCareResponse(BaseModel):
    total_patients: int
    total_opportunities: int = Field(description="Total missing care gaps identified")
    opportunities_by_condition: list[MissingCareCondition]
    cases: list[MissingCareCase]


@router.get(
    "/missing-care",
    response_model=MissingCareResponse,
    summary="Missing care opportunities",
)
async def missing_care_opportunities(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    min_count: int = Query(1, ge=1, description="Minimum occurrences to include"),
    session: AsyncSession = Depends(get_session),
):
    """
    Identify NICE-recommended actions NOT documented in patient records.

    Surfaces care gaps — things guidelines recommend but the GP did not
    document. Grouped by condition for pattern identification. Useful for
    identifying systematic quality improvement opportunities.
    """
    return await get_missing_care_summary(session, job_id, min_count)


# ── LLM-as-Judge Scorer Evaluation schemas ───────────────────────────


class ScorerDiagnosisEvalItem(BaseModel):
    diagnosis: str
    score: int
    judgement: str
    reasoning_quality: int = Field(ge=1, le=5)
    citation_accuracy: int = Field(ge=1, le=5)
    score_calibration: int = Field(ge=1, le=5)


class ScorerPatientEvalItem(BaseModel):
    pat_id: str
    total_diagnoses: int
    mean_reasoning_quality: float
    mean_citation_accuracy: float
    mean_score_calibration: float
    per_diagnosis: list[ScorerDiagnosisEvalItem]


class ScorerEvaluationResponse(BaseModel):
    job_id: int
    total_patients: int
    total_diagnoses: int
    mean_reasoning_quality: float = Field(
        description="Average reasoning quality (1-5)",
    )
    mean_citation_accuracy: float = Field(
        description="Average citation accuracy (1-5)",
    )
    mean_score_calibration: float = Field(
        description="Average score calibration (1-5)",
    )
    per_patient: list[ScorerPatientEvalItem]


@router.post(
    "/evaluate/scorer/{job_id}",
    response_model=ScorerEvaluationResponse,
    summary="Evaluate scorer using LLM-as-Judge",
)
async def evaluate_scorer_endpoint(
    job_id: int,
    limit: int = Query(
        5, ge=1, le=50,
        description="Max patients to evaluate (each costs LLM calls)",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate the scorer agent's output quality using LLM-as-Judge.

    Loads stored scoring results from a completed batch job and sends
    each diagnosis to a separate LLM call for quality assessment.
    Rates reasoning quality, citation accuracy, and score calibration
    (all 1-5). Does NOT re-run the pipeline — works from stored data.
    """
    from src.ai.factory import get_ai_provider

    query = (
        select(AuditResult)
        .where(
            AuditResult.status == "completed",
            AuditResult.job_id == job_id,
        )
        .options(selectinload(AuditResult.patient))
    )
    result = await session.execute(query)
    results = list(result.scalars().all())

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No completed results for job {job_id}",
        )

    results = results[:limit]
    ai_provider = get_ai_provider()

    per_patient: list[dict] = []
    all_reasoning: list[int] = []
    all_citation: list[int] = []
    all_calibration: list[int] = []
    total_diagnoses = 0

    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        pat_id = (
            r.patient.pat_id
            if r.patient
            else details.get("pat_id", "Unknown")
        )
        scoring = scoring_from_stored(details)
        metrics = await evaluate_scoring(scoring, ai_provider)

        total_diagnoses += metrics.total_diagnoses
        for d in metrics.per_diagnosis:
            all_reasoning.append(d["reasoning_quality"])
            all_citation.append(d["citation_accuracy"])
            all_calibration.append(d["score_calibration"])

        per_patient.append({
            "pat_id": pat_id,
            "total_diagnoses": metrics.total_diagnoses,
            "mean_reasoning_quality": metrics.mean_reasoning_quality,
            "mean_citation_accuracy": metrics.mean_citation_accuracy,
            "mean_score_calibration": metrics.mean_score_calibration,
            "per_diagnosis": metrics.per_diagnosis,
        })

    n = len(all_reasoning)
    return {
        "job_id": job_id,
        "total_patients": len(per_patient),
        "total_diagnoses": total_diagnoses,
        "mean_reasoning_quality": (
            round(sum(all_reasoning) / n, 4) if n > 0 else 0.0
        ),
        "mean_citation_accuracy": (
            round(sum(all_citation) / n, 4) if n > 0 else 0.0
        ),
        "mean_score_calibration": (
            round(sum(all_calibration) / n, 4) if n > 0 else 0.0
        ),
        "per_patient": per_patient,
    }


# ── System-level Metrics ─────────────────────────────────────────────


class ConfidenceStatsSchema(BaseModel):
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    std: float | None


class SystemMetricsResponse(BaseModel):
    job_id: int
    total_patients: int
    total_diagnoses: int
    score_class_distribution: dict[str, int] = Field(
        description="Count per score level: +2, +1, 0, -1, -2",
    )
    adherence_rate: float = Field(
        description="(compliant + partial) / total_scored",
    )
    confidence_stats: ConfidenceStatsSchema
    error_rate: float
    per_class_counts: dict[str, int]


@router.get(
    "/system-metrics",
    response_model=SystemMetricsResponse,
    summary="System-level metrics for a job",
)
async def system_metrics(
    job_id: int = Query(..., description="Batch job ID"),
    session: AsyncSession = Depends(get_session),
):
    """
    Comprehensive system-level evaluation metrics for a batch job.

    Returns score class distribution (+2 to -2), adherence rate,
    confidence statistics, and per-class counts. Computed entirely
    from stored data — no LLM calls needed.
    """
    return await compute_system_metrics(session, job_id)


# ── Cross-Model Classification Metrics ───────────────────────────────


class ConfusionMatrixSchema(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ClassMetricsSchema(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class CrossModelMetricsResponse(BaseModel):
    job_a_id: int
    job_b_id: int
    job_a_provider: str | None
    job_b_provider: str | None
    total_diagnoses_compared: int
    confusion_matrix: ConfusionMatrixSchema
    per_class_metrics: dict[str, ClassMetricsSchema]
    cohen_kappa_5class: float = Field(description="5-class Cohen's kappa")
    cohen_kappa_3class: float = Field(description="3-class Cohen's kappa")
    exact_match_accuracy: float = Field(
        description="Fraction of diagnoses with identical scores",
    )
    auroc: float | None = Field(
        description="AUROC (adherent vs non-adherent, using confidence)",
    )
    agreement_rate: float
    pearson_correlation: float


@router.get(
    "/cross-model-metrics",
    response_model=CrossModelMetricsResponse,
    summary="Cross-model classification metrics",
)
async def cross_model_metrics(
    job_a: int = Query(..., description="First job ID (Model A)"),
    job_b: int = Query(..., description="Second job ID (Model B)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Enhanced cross-model comparison with confusion matrix and classification metrics.

    Returns a 5×5 confusion matrix, per-class precision/recall/F1,
    Cohen's kappa at both 5-class and 3-class levels, exact-match
    accuracy, and AUROC. Computed from stored data — no LLM calls.
    """
    try:
        return await compute_cross_model_classification(session, job_a, job_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Extractor Metrics ────────────────────────────────────────────────


class CategoryMetricsSchema(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class ExtractorMetricsResponse(BaseModel):
    total_concepts: int
    total_with_rules: int = Field(
        description="Concepts where rule-based ground truth was available",
    )
    rule_match_rate: float = Field(
        description="Fraction of rule-matched concepts where extractor agreed",
    )
    per_category: dict[str, CategoryMetricsSchema]
    category_distribution: dict[str, int] = Field(
        description="Count of entries per stored category",
    )


@router.get(
    "/extractor-metrics",
    response_model=ExtractorMetricsResponse,
    summary="Extractor SNOMED categorisation quality",
)
async def extractor_metrics(
    sample_size: int | None = Query(
        None, ge=10, le=10000,
        description="Limit to N concepts (default: all)",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate extractor quality using SNOMED rules as pseudo-ground-truth.

    Compares the extractor's stored category assignments against
    rule-based categorisation. Returns per-category precision, recall,
    and F1-score. No LLM calls needed — works from stored DB data.
    """
    return await evaluate_extractor_from_db(session, sample_size)


# ── Full Agent Evaluation ────────────────────────────────────────────


class RetrieverIRMetrics(BaseModel):
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_ndcg: float
    mean_mrr: float
    mean_relevance: float


class AgentEvaluationResponse(BaseModel):
    total_patients: int
    extractor: dict | None = None
    query: dict | None = None
    retriever: dict | None = None
    retriever_ir: RetrieverIRMetrics | None = None
    scorer: dict | None = None
    per_patient: list[dict] = []


@router.post(
    "/evaluate/agents",
    response_model=AgentEvaluationResponse,
    summary="Full agent evaluation (runs pipeline)",
)
async def evaluate_agents(
    limit: int = Query(
        5, ge=1, le=20,
        description="Max patients to evaluate (each runs full pipeline + LLM judge)",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate all 4 pipeline agents by running the pipeline on a sample.

    Picks random patients, runs the complete pipeline, then evaluates:
    - Extractor: precision/recall/F1 per SNOMED category
    - Query Generator: relevance and coverage (1-5)
    - Retriever: per-guideline relevance + Precision@k, nDCG, MRR
    - Scorer: reasoning quality, citation accuracy, calibration (1-5)

    **Expensive**: each patient requires multiple LLM calls.
    """
    from src.ai.factory import get_ai_provider

    ai_provider = get_ai_provider()
    return await run_agent_evaluation(session, ai_provider, limit)
