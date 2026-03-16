"""
Report API routes.

Read-only analytics endpoints for reviewing audit results:
dashboard stats, condition breakdowns, non-adherent cases,
score distributions, and downloadable exports (CSV/HTML).
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditResult
from src.models.database import get_session
from src.services.export import generate_comparison_html, generate_csv, generate_html_report
from src.services.reporting import (
    get_condition_breakdown,
    get_dashboard_stats,
    get_non_adherent_cases,
    get_score_distribution,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


# ── Response schemas ──────────────────────────────────────────────────


class ScoreStatsSchema(BaseModel):
    mean: float | None = Field(description="Mean adherence score across all patients")
    median: float | None = Field(description="Median adherence score")
    min: float | None = Field(description="Lowest adherence score")
    max: float | None = Field(description="Highest adherence score")


class DashboardResponse(BaseModel):
    total_audited: int = Field(description="Number of patients with completed audits")
    total_failed: int = Field(description="Number of patients whose audit failed")
    failure_rate: float = Field(description="Proportion of audits that failed (0.0–1.0)")
    score_stats: ScoreStatsSchema

    model_config = {"json_schema_extra": {"examples": [
        {"total_audited": 50, "total_failed": 2, "failure_rate": 0.038, "score_stats": {"mean": 0.42, "median": 0.33, "min": 0.0, "max": 1.0}}
    ]}}


class ConditionBreakdownItem(BaseModel):
    diagnosis: str = Field(description="The diagnosis term (e.g. 'Low back pain')")
    total_cases: int = Field(description="Total scored cases for this diagnosis")
    compliant: int = Field(description="Score +2: fully guideline-compliant")
    partial: int = Field(description="Score +1: partially compliant")
    not_relevant: int = Field(description="Score 0: guideline not applicable")
    non_compliant: int = Field(description="Score -1: non-compliant")
    risky: int = Field(description="Score -2: risky non-compliant")
    adherent: int = Field(description="Compliant + partially compliant")
    non_adherent: int = Field(description="Non-compliant + risky")
    errors: int = Field(description="Number with scoring errors")
    adherence_rate: float = Field(description="Proportion adherent (0.0–1.0)")


class NonAdherentCase(BaseModel):
    pat_id: str = Field(description="Patient identifier")
    diagnosis: str = Field(description="The non-adherent diagnosis")
    index_date: str | None = Field(description="Date of the clinical episode")
    score: int | None = Field(None, description="Score value (-1 or -2)")
    judgement: str = Field("NON-COMPLIANT", description="Judgement label")
    confidence: float | None = Field(None, description="Confidence score (0.0–1.0)")
    explanation: str = Field(description="LLM explanation of why non-adherent")
    cited_guideline_text: str = Field("", description="Cited NICE guideline text")
    guidelines_not_followed: list[str] = Field(description="Specific guidelines not followed")


class NonAdherentResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    cases: list[NonAdherentCase]


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class ScoreDistributionResponse(BaseModel):
    bins: list[HistogramBin]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=DashboardResponse, summary="Dashboard summary")
async def dashboard(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    session: AsyncSession = Depends(get_session),
):
    """
    High-level audit summary: total patients audited, failure rate,
    and adherence score statistics (mean, median, min, max).

    Uses only SQL aggregation on stored scores -- fast even with
    thousands of results.
    """
    return await get_dashboard_stats(session, job_id, model=model)


@router.get("/conditions", response_model=list[ConditionBreakdownItem], summary="Per-condition breakdown")
async def conditions(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    min_count: int = Query(1, ge=1, description="Minimum cases to include a diagnosis"),
    sort_by: str = Query("count", description="Sort by 'count' (descending) or 'adherence_rate' (ascending, worst-first)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Adherence breakdown grouped by diagnosis. Shows how many patients
    were adherent vs non-adherent for each condition.

    Use `sort_by=adherence_rate` to find conditions with the worst
    guideline adherence. Use `min_count` to filter out rare diagnoses.
    """
    return await get_condition_breakdown(session, job_id, min_count, sort_by, model=model)


@router.get("/non-adherent", response_model=NonAdherentResponse, summary="Non-adherent cases")
async def non_adherent(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    session: AsyncSession = Depends(get_session),
):
    """
    Paginated list of every diagnosis scored as non-adherent (-1 or -2).

    Each case includes the patient ID, diagnosis, judgement, confidence,
    the LLM's explanation and cited guideline text. Intended for clinical review.
    """
    return await get_non_adherent_cases(session, job_id, page, page_size, model=model)


@router.get("/score-distribution", response_model=ScoreDistributionResponse, summary="Score histogram")
async def score_distribution(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    bins: int = Query(10, ge=2, le=100, description="Number of histogram bins"),
    session: AsyncSession = Depends(get_session),
):
    """
    Histogram of patient-level overall adherence scores.

    Divides the 0.0-1.0 range into equal bins and counts how many
    patients fall in each bin. Useful for visualising the distribution
    of guideline adherence across the patient population.
    """
    return await get_score_distribution(session, job_id, bins, model=model)


# ── Export endpoints ─────────────────────────────────────────────────


@router.get("/export/csv", summary="Download CSV export", response_class=Response)
async def export_csv(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Download audit results as a CSV file.

    One row per diagnosis per patient, with columns: pat_id, overall_score,
    diagnosis, index_date, score, explanation, guidelines_followed,
    guidelines_not_followed. Open in Excel/Google Sheets or share directly.
    """
    csv_content = await generate_csv(session, job_id, model=model)
    label = model or (f"job_{job_id}" if job_id else "all")
    filename = f"clinaudit_ai_audit_{label}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/html", summary="Download HTML report", response_class=HTMLResponse)
async def export_html(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    model: str | None = Query(None, description="Scope by model name (e.g. gpt-4.1-mini)"),
    use_saved_evals: bool = Query(
        False, description="Load pre-computed scorer and agent evaluation results from data/eval_results/",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Generate a self-contained HTML audit report.

    Opens directly in any browser. Includes dashboard summary, per-condition
    adherence breakdown, system metrics, extractor quality, missing care
    opportunities, and detailed per-patient results with LLM explanations.

    Set use_saved_evals=true to include pre-computed scorer and agent
    evaluation results (fast, no LLM calls).
    """
    html_content = await generate_html_report(
        session, job_id, model=model, use_saved_evals=use_saved_evals,
    )
    return HTMLResponse(content=html_content)


@router.get(
    "/export/comparison-html",
    summary="Download comparison HTML report",
    response_class=HTMLResponse,
)
async def export_comparison_html(
    job_a: int | None = Query(None, description="First job ID (e.g. OpenAI)"),
    job_b: int | None = Query(None, description="Second job ID (e.g. Ollama)"),
    model_a: str | None = Query(None, description="First model name (e.g. gpt-4.1-mini)"),
    model_b: str | None = Query(None, description="Second model name (e.g. mistral-small)"),
    include_scorer_eval: bool = Query(
        False, description="Run LLM-as-Judge scorer evaluation live (slow, prefer use_saved_evals)",
    ),
    scorer_eval_limit: int = Query(
        10, ge=1, le=50, description="Max patients for live scorer eval (if include_scorer_eval=true)",
    ),
    use_saved_evals: bool = Query(
        False, description="Load pre-computed evaluation results from data/eval_results/",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Generate a self-contained HTML comparison report for two batch jobs or models.

    Provide either job IDs or model names (or a mix) for each side.
    Shows side-by-side system metrics, confusion matrix, per-class P/R/F1,
    Cohen's kappa, AUROC, extractor quality, missing care comparison,
    per-condition adherence, and per-patient results — all with inline
    SVG charts. Share via email or open in any browser.

    Set use_saved_evals=true to include pre-computed scorer and agent
    evaluation results from data/eval_results/ (fast, no LLM calls).

    Set include_scorer_eval=true to run LLM-as-Judge scorer evaluation
    live for both jobs (slow, adds ~30s per job).
    """
    if job_a is None and model_a is None:
        raise HTTPException(status_code=400, detail="Provide either job_a or model_a")
    if job_b is None and model_b is None:
        raise HTTPException(status_code=400, detail="Provide either job_b or model_b")

    scorer_evals = None
    agent_evals = None

    if use_saved_evals:
        scorer_evals, agent_evals = _load_saved_evals(
            model_a or "gpt-4.1-mini",
            model_b or "mistral-small",
        )

    if include_scorer_eval and not scorer_evals:
        from src.services.evaluation import evaluate_scoring, scoring_from_stored
        from src.ai.factory import get_ai_provider
        import json as _json

        ai_provider = get_ai_provider()
        provider_name = getattr(ai_provider, "model", "unknown")

        scorer_evals = {}
        for side_job, side_model, key_prefix in [
            (job_a, model_a, "job_a"),
            (job_b, model_b, "job_b"),
        ]:
            results = await _load_results_for_scorer(
                session, side_job, side_model, scorer_eval_limit,
            )

            all_rq, all_ca, all_sc = [], [], []
            for r in results:
                if not r.details_json:
                    continue
                try:
                    details = _json.loads(r.details_json)
                except _json.JSONDecodeError:
                    continue
                scoring = scoring_from_stored(details)
                metrics = await evaluate_scoring(scoring, ai_provider)
                for d in metrics.per_diagnosis:
                    all_rq.append(d["reasoning_quality"])
                    all_ca.append(d["citation_accuracy"])
                    all_sc.append(d["score_calibration"])

            n = len(all_rq)
            judge_key = f"{key_prefix}_openai_judge" if "gpt" in str(provider_name).lower() else f"{key_prefix}_ollama_judge"
            scorer_evals[judge_key] = {
                "mean_reasoning_quality": round(sum(all_rq) / n, 4) if n else 0,
                "mean_citation_accuracy": round(sum(all_ca) / n, 4) if n else 0,
                "mean_score_calibration": round(sum(all_sc) / n, 4) if n else 0,
            }

    try:
        html_content = await generate_comparison_html(
            session, job_a, job_b,
            model_a=model_a, model_b=model_b,
            scorer_evals=scorer_evals,
            agent_eval=agent_evals,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return HTMLResponse(content=html_content)


def _load_saved_evals(
    model_a: str, model_b: str,
) -> tuple[dict | None, dict | None]:
    """Load pre-computed scorer and agent eval results from data/eval_results/.

    Returns (scorer_evals, agent_evals) dicts ready to pass to
    generate_comparison_html(). Returns (None, None) if files not found.
    """
    import json
    from pathlib import Path

    eval_dir = Path("data/eval_results")

    # Known model name → file slug mappings
    _SLUG_MAP = {
        "gpt-4.1-mini": "gpt4_mini",
        "gpt-4o-mini": "gpt4_mini",
        "gpt-4.1": "gpt4",
        "mistral-small": "mistral_small",
    }

    def _model_slug(model: str) -> str:
        return _SLUG_MAP.get(model, model.replace(".", "").replace("-", "_").replace(" ", "_"))

    slug_a = _model_slug(model_a)
    slug_b = _model_slug(model_b)

    # ── Scorer evals: 4 files (each model × each judge) ──────────
    scorer_evals = {}
    scorer_file_map = {
        f"job_a_ollama_judge": f"scorer_eval_{slug_a}_ollama_judge_100.json",
        f"job_a_openai_judge": f"scorer_eval_{slug_a}_openai_judge_100.json",
        f"job_b_ollama_judge": f"scorer_eval_{slug_b}_ollama_judge_100.json",
        f"job_b_openai_judge": f"scorer_eval_{slug_b}_openai_judge_100.json",
    }
    for key, filename in scorer_file_map.items():
        path = eval_dir / filename
        if not path.exists():
            # Try without _100 suffix
            alt = filename.replace("_100.json", ".json")
            path = eval_dir / alt
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            scorer_evals[key] = {
                "mean_reasoning_quality": data.get("mean_reasoning_quality", 0),
                "mean_citation_accuracy": data.get("mean_citation_accuracy", 0),
                "mean_score_calibration": data.get("mean_score_calibration", 0),
                "total_patients": data.get("total_patients", 0),
                "total_diagnoses": data.get("total_diagnoses", 0),
            }

    # ── Agent evals: 2 files (one per model) ─────────────────────
    agent_evals = {}
    for label, slug in [("model_a", slug_a), ("model_b", slug_b)]:
        path = eval_dir / f"agents_eval_{slug}_50.json"
        if not path.exists():
            # Try glob pattern
            matches = list(eval_dir.glob(f"agents_eval_{slug}*.json"))
            if matches:
                path = matches[0]
        if path.exists():
            with open(path) as f:
                agent_evals[label] = json.load(f)

    return (
        scorer_evals if scorer_evals else None,
        agent_evals if agent_evals else None,
    )


async def _load_results_for_scorer(
    session: AsyncSession,
    job_id: int | None,
    model: str | None,
    limit: int,
    offset: int = 0,
) -> list:
    """Load results for scorer evaluation by job_id or model name.

    Sorted deterministically by pat_id so the same offset/limit yields
    the same patients across different models.
    """
    from src.models.audit import AuditJob
    from src.models.patient import Patient

    if job_id is not None:
        query = (
            select(AuditResult)
            .join(Patient, AuditResult.patient_id == Patient.id)
            .where(AuditResult.status == "completed", AuditResult.job_id == job_id)
            .options(selectinload(AuditResult.patient))
        )
    elif model is not None:
        job_ids_q = select(AuditJob.id).where(AuditJob.provider == model)
        query = (
            select(AuditResult)
            .join(Patient, AuditResult.patient_id == Patient.id)
            .where(AuditResult.status == "completed", AuditResult.job_id.in_(job_ids_q))
            .options(selectinload(AuditResult.patient))
        )
    else:
        return []

    query = query.order_by(Patient.pat_id).offset(offset).limit(limit)
    result = await session.execute(query)
    return list(result.scalars().all())
