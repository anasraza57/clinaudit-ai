"""
Report API routes.

Read-only analytics endpoints for reviewing audit results:
dashboard stats, condition breakdowns, non-adherent cases,
score distributions, and downloadable exports (CSV/HTML).
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_session
from src.services.export import generate_csv, generate_html_report
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
    total_cases: int = Field(description="Total patients with this diagnosis")
    adherent: int = Field(description="Number scored +1 (adherent)")
    non_adherent: int = Field(description="Number scored -1 (non-adherent)")
    errors: int = Field(description="Number with scoring errors")
    adherence_rate: float = Field(description="Proportion adherent (0.0–1.0)")


class NonAdherentCase(BaseModel):
    pat_id: str = Field(description="Patient identifier")
    diagnosis: str = Field(description="The non-adherent diagnosis")
    index_date: str | None = Field(description="Date of the clinical episode")
    explanation: str = Field(description="LLM explanation of why non-adherent")
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
    Paginated list of every diagnosis scored as non-adherent (-1).

    Each case includes the patient ID, diagnosis, the LLM's explanation
    of why the treatment was non-adherent, and the specific guidelines
    that were not followed. Intended for clinical review.
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
