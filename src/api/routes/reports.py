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
    session: AsyncSession = Depends(get_session),
):
    """
    High-level audit summary: total patients audited, failure rate,
    and adherence score statistics (mean, median, min, max).

    Uses only SQL aggregation on stored scores — fast even with
    thousands of results.
    """
    return await get_dashboard_stats(session, job_id)


@router.get("/conditions", response_model=list[ConditionBreakdownItem], summary="Per-condition breakdown")
async def conditions(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
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
    return await get_condition_breakdown(session, job_id, min_count, sort_by)


@router.get("/non-adherent", response_model=NonAdherentResponse, summary="Non-adherent cases")
async def non_adherent(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    session: AsyncSession = Depends(get_session),
):
    """
    Paginated list of every diagnosis scored as non-adherent (-1 or -2).

    Each case includes the patient ID, diagnosis, judgement, confidence,
    the LLM's explanation and cited guideline text. Intended for clinical review.
    """
    return await get_non_adherent_cases(session, job_id, page, page_size)


@router.get("/score-distribution", response_model=ScoreDistributionResponse, summary="Score histogram")
async def score_distribution(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    bins: int = Query(10, ge=2, le=100, description="Number of histogram bins"),
    session: AsyncSession = Depends(get_session),
):
    """
    Histogram of patient-level overall adherence scores.

    Divides the 0.0–1.0 range into equal bins and counts how many
    patients fall in each bin. Useful for visualising the distribution
    of guideline adherence across the patient population.
    """
    return await get_score_distribution(session, job_id, bins)


# ── Export endpoints ─────────────────────────────────────────────────


@router.get("/export/csv", summary="Download CSV export", response_class=Response)
async def export_csv(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    session: AsyncSession = Depends(get_session),
):
    """
    Download audit results as a CSV file.

    One row per diagnosis per patient, with columns: pat_id, overall_score,
    diagnosis, index_date, score, explanation, guidelines_followed,
    guidelines_not_followed. Open in Excel/Google Sheets or share directly.
    """
    csv_content = await generate_csv(session, job_id)
    filename = f"guideline_guard_audit{'_job_' + str(job_id) if job_id else ''}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/html", summary="Download HTML report", response_class=HTMLResponse)
async def export_html(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    session: AsyncSession = Depends(get_session),
):
    """
    Generate a self-contained HTML audit report.

    Opens directly in any browser. Includes dashboard summary, per-condition
    adherence breakdown, and detailed per-patient results with LLM explanations.
    Suitable for sharing via email, printing, or presenting.
    """
    html_content = await generate_html_report(session, job_id)
    return HTMLResponse(content=html_content)


@router.get(
    "/export/comparison-html",
    summary="Download comparison HTML report",
    response_class=HTMLResponse,
)
async def export_comparison_html(
    job_a: int = Query(..., description="First job ID (e.g. OpenAI)"),
    job_b: int = Query(..., description="Second job ID (e.g. Ollama)"),
    include_scorer_eval: bool = Query(
        False, description="Run LLM-as-Judge scorer evaluation and include results",
    ),
    scorer_eval_limit: int = Query(
        10, ge=1, le=50, description="Max patients for scorer eval (if included)",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Generate a self-contained HTML comparison report for two batch jobs.

    Shows side-by-side system metrics, confusion matrix, per-class P/R/F1,
    Cohen's kappa, AUROC, extractor quality, missing care comparison,
    per-condition adherence, and per-patient results — all with inline
    SVG charts. Share via email or open in any browser.

    Set include_scorer_eval=true to run LLM-as-Judge scorer evaluation
    for both jobs and include the results (adds ~30s per job).
    """
    scorer_evals = None
    if include_scorer_eval:
        from src.services.evaluation import evaluate_scoring, scoring_from_stored
        from src.ai.factory import get_ai_provider
        import json as _json

        ai_provider = get_ai_provider()
        provider_name = getattr(ai_provider, "model", "unknown")

        scorer_evals = {}
        for job_id, key_prefix in [(job_a, "job_a"), (job_b, "job_b")]:
            query = (
                select(AuditResult)
                .where(AuditResult.status == "completed", AuditResult.job_id == job_id)
                .options(selectinload(AuditResult.patient))
            )
            result = await session.execute(query)
            results = list(result.scalars().all())[:scorer_eval_limit]

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
            scorer_evals=scorer_evals,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return HTMLResponse(content=html_content)
