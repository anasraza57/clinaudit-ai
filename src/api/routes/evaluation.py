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

from src.models.audit import AuditJob, AuditResult
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
    job_a_id: int | None = None
    job_b_id: int | None = None
    job_a_model: str | None = Field(description="Model used for job A (e.g. gpt-4o-mini)")
    job_b_model: str | None = Field(description="Model used for job B (e.g. mistral-small)")
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
    summary="Compare two batch jobs or models",
)
async def compare_models(
    job_a: int | None = Query(None, description="First job ID"),
    job_b: int | None = Query(None, description="Second job ID"),
    model_a: str | None = Query(None, description="First model name (e.g. gpt-4.1-mini)"),
    model_b: str | None = Query(None, description="Second model name (e.g. mistral-small)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Compare audit results from two different batch jobs or models side-by-side.

    **Usage options:**
    - By job: `?job_a=1&job_b=2`
    - By model: `?model_a=gpt-4.1-mini&model_b=mistral-small`
    - Mixed: `?job_a=1&model_b=mistral-small`

    When using model names, results are aggregated across all jobs for that model.
    Returns per-patient score differences, agreement metrics (Cohen's kappa),
    and per-condition breakdown.
    """
    if job_a is None and model_a is None:
        raise HTTPException(status_code=400, detail="Provide either job_a or model_a")
    if job_b is None and model_b is None:
        raise HTTPException(status_code=400, detail="Provide either job_b or model_b")

    try:
        result = await compare_jobs(
            session, job_a_id=job_a, job_b_id=job_b,
            model_a=model_a, model_b=model_b,
        )
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
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    min_count: int = Query(1, ge=1, description="Minimum occurrences to include"),
    session: AsyncSession = Depends(get_session),
):
    """
    Identify NICE-recommended actions NOT documented in patient records.

    Surfaces care gaps -- things guidelines recommend but the GP did not
    document. Grouped by condition for pattern identification. Useful for
    identifying systematic quality improvement opportunities.
    """
    return await get_missing_care_summary(session, job_id, min_count, model=model)


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
    job_id: int | None = None
    model: str | None = None
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
    "/evaluate/scorer",
    response_model=ScorerEvaluationResponse,
    summary="Evaluate scorer using LLM-as-Judge",
)
async def evaluate_scorer_endpoint(
    job_id: int | None = Query(None, description="Batch job ID"),
    model: str | None = Query(None, description="Model name (e.g. gpt-4.1-mini)"),
    limit: int = Query(
        5, ge=1, le=50,
        description="Max patients to evaluate (each costs LLM calls)",
    ),
    offset: int = Query(
        0, ge=0,
        description=(
            "Skip first N patients (sorted by pat_id). "
            "Use offset=0&limit=20, then offset=20&limit=10 for next batch."
        ),
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate the scorer agent's output quality using LLM-as-Judge.

    **Usage options:**
    - By job: `?job_id=1`
    - By model: `?model=gpt-4.1-mini`
    - Page through: `?model=gpt-4.1-mini&offset=20&limit=10`

    Results are **sorted by pat_id** so the same limit/offset gives the
    same patients across different models — enabling fair cross-model comparison.

    Loads stored scoring results and sends each diagnosis to a separate
    LLM call for quality assessment. Rates reasoning quality, citation
    accuracy, and score calibration (all 1-5).
    """
    if job_id is None and model is None:
        raise HTTPException(status_code=400, detail="Provide either job_id or model")
    return await _evaluate_scorer_impl(
        session, limit, offset=offset, job_id=job_id, model=model,
    )


async def _evaluate_scorer_impl(
    session: AsyncSession,
    limit: int,
    offset: int = 0,
    job_id: int | None = None,
    model: str | None = None,
) -> dict:
    """Shared implementation for scorer evaluation by job or model.

    Results are sorted deterministically by pat_id so the same
    offset/limit yields the same patients across different models.
    """
    from src.ai.factory import get_ai_provider
    from src.models.patient import Patient

    query = (
        select(AuditResult)
        .join(Patient, AuditResult.patient_id == Patient.id)
        .where(AuditResult.status == "completed")
        .options(selectinload(AuditResult.patient))
    )
    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)
    elif model is not None:
        job_ids_q = select(AuditJob.id).where(AuditJob.provider == model)
        query = query.where(AuditResult.job_id.in_(job_ids_q))

    # Deterministic ordering by pat_id for cross-model consistency
    query = query.order_by(Patient.pat_id).offset(offset).limit(limit)

    result = await session.execute(query)
    results = list(result.scalars().all())

    if not results:
        detail = f"job {job_id}" if job_id else f"model '{model}'"
        raise HTTPException(
            status_code=404,
            detail=f"No completed results for {detail} (offset={offset})",
        )
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
        "model": model,
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
    job_id: int | None = None
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
    job_id: int | None = Query(None, description="Batch job ID"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Comprehensive system-level evaluation metrics for a batch job or model.

    Returns score class distribution (+2 to -2), adherence rate,
    confidence statistics, and per-class counts. Computed entirely
    from stored data -- no LLM calls needed.
    """
    return await compute_system_metrics(session, job_id, model=model)


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
    job_a_id: int | None = None
    job_b_id: int | None = None
    job_a_model: str | None = Field(description="Model used for job A")
    job_b_model: str | None = Field(description="Model used for job B")
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
    job_a: int | None = Query(None, description="First job ID (Model A)"),
    job_b: int | None = Query(None, description="Second job ID (Model B)"),
    model_a: str | None = Query(None, description="First model name (e.g. gpt-4.1-mini)"),
    model_b: str | None = Query(None, description="Second model name (e.g. mistral-small)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Enhanced cross-model comparison with confusion matrix and classification metrics.

    **Usage options:**
    - By job: `?job_a=1&job_b=2`
    - By model: `?model_a=gpt-4.1-mini&model_b=mistral-small`

    Returns a 5×5 confusion matrix, per-class precision/recall/F1,
    Cohen's kappa at both 5-class and 3-class levels, exact-match
    accuracy, and AUROC. Computed from stored data — no LLM calls.
    """
    if job_a is None and model_a is None:
        raise HTTPException(status_code=400, detail="Provide either job_a or model_a")
    if job_b is None and model_b is None:
        raise HTTPException(status_code=400, detail="Provide either job_b or model_b")

    try:
        return await compute_cross_model_classification(
            session, job_a_id=job_a, job_b_id=job_b,
            model_a=model_a, model_b=model_b,
        )
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
    model: str | None = Field(None, description="Model used for pipeline + judge")
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
    model: str | None = Query(
        None,
        description=(
            "Model name to run the pipeline with (e.g. gpt-4.1-mini, mistral-small). "
            "Defaults to the AI_PROVIDER configured in .env."
        ),
    ),
    limit: int = Query(
        5, ge=1, le=50,
        description="Max patients to evaluate (each runs full pipeline + LLM judge)",
    ),
    offset: int = Query(
        0, ge=0,
        description=(
            "Skip first N patients (sorted by pat_id). "
            "Use offset=0&limit=5, then offset=5&limit=5 for next batch."
        ),
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate all 4 pipeline agents by running the pipeline on a sample.

    **Usage options:**
    - Default provider: `?limit=5`
    - Specific model: `?model=gpt-4.1-mini&limit=5`
    - Page through: `?model=mistral-small&offset=5&limit=5`

    Patients are **sorted by pat_id** so the same offset/limit gives
    deterministic, resumable evaluation across calls. Pass the same
    `model` parameter to compare agents across different models.

    Evaluates:
    - Extractor: precision/recall/F1 per SNOMED category
    - Query Generator: relevance and coverage (1-5)
    - Retriever: per-guideline relevance + Precision@k, nDCG, MRR
    - Scorer: reasoning quality, citation accuracy, calibration (1-5)

    **Expensive**: each patient requires multiple LLM calls.
    """
    if model:
        from src.ai.factory import get_ai_provider_for_model
        ai_provider = get_ai_provider_for_model(model)
    else:
        from src.ai.factory import get_ai_provider
        ai_provider = get_ai_provider()
    return await run_agent_evaluation(
        session, ai_provider, limit, offset=offset, model=model,
    )
