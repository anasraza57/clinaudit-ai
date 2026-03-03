"""
Export service — CSV and HTML report generation for audit results.

Generates downloadable reports that can be shared with colleagues,
supervisors, or used for presentations.
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditJob, AuditResult

logger = logging.getLogger(__name__)


async def _load_results_with_patients(
    session: AsyncSession,
    job_id: int | None = None,
) -> list[AuditResult]:
    """Load completed audit results with patient data eager-loaded."""
    query = (
        select(AuditResult)
        .where(AuditResult.status == "completed")
        .options(selectinload(AuditResult.patient))
        .order_by(AuditResult.id)
    )
    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)

    result = await session.execute(query)
    return list(result.scalars().all())


async def _get_job_info(
    session: AsyncSession,
    job_id: int,
) -> AuditJob | None:
    """Load a batch job by ID."""
    result = await session.execute(
        select(AuditJob).where(AuditJob.id == job_id)
    )
    return result.scalar_one_or_none()


def _parse_details(details_json: str | None) -> list[dict]:
    """Parse the per-diagnosis scores from details_json."""
    if not details_json:
        return []
    try:
        details = json.loads(details_json)
        return details.get("scores", [])
    except (json.JSONDecodeError, AttributeError):
        return []


# ── CSV Export ───────────────────────────────────────────────────────


async def generate_csv(
    session: AsyncSession,
    job_id: int | None = None,
) -> str:
    """
    Generate a CSV string with one row per diagnosis per patient.

    Columns: pat_id, overall_score, diagnosis, index_date, score,
    explanation, guidelines_followed, guidelines_not_followed
    """
    results = await _load_results_with_patients(session, job_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "pat_id",
        "overall_score",
        "diagnosis",
        "index_date",
        "score",
        "explanation",
        "guidelines_followed",
        "guidelines_not_followed",
    ])

    for r in results:
        pat_id = r.patient.pat_id if r.patient else "Unknown"
        scores = _parse_details(r.details_json)

        if not scores:
            # Patient with no diagnosis detail — write summary row
            writer.writerow([
                pat_id,
                r.overall_score,
                "",
                r.index_date or "",
                "",
                "",
                "",
                "",
            ])
            continue

        for ds in scores:
            writer.writerow([
                pat_id,
                r.overall_score,
                ds.get("diagnosis", ""),
                ds.get("index_date", ""),
                ds.get("score", ""),
                ds.get("explanation", ""),
                "; ".join(ds.get("guidelines_followed", [])),
                "; ".join(ds.get("guidelines_not_followed", [])),
            ])

    return output.getvalue()


# ── HTML Report ──────────────────────────────────────────────────────


async def generate_html_report(
    session: AsyncSession,
    job_id: int | None = None,
) -> str:
    """
    Generate a self-contained HTML report with dashboard stats,
    per-condition breakdown, and per-patient detail tables.
    """
    results = await _load_results_with_patients(session, job_id)

    # Compute stats
    scores = [r.overall_score for r in results if r.overall_score is not None]
    total_patients = len(results)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    sorted_scores = sorted(scores)
    if sorted_scores:
        n = len(sorted_scores)
        median_score = (
            (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            if n % 2 == 0
            else sorted_scores[n // 2]
        )
    else:
        median_score = 0.0

    # Per-condition breakdown
    conditions: dict[str, dict[str, int]] = {}
    for r in results:
        for ds in _parse_details(r.details_json):
            term = ds.get("diagnosis", "Unknown")
            if term not in conditions:
                conditions[term] = {"adherent": 0, "non_adherent": 0}
            if ds.get("score") == 1:
                conditions[term]["adherent"] += 1
            elif ds.get("score") == -1:
                conditions[term]["non_adherent"] += 1

    condition_rows = []
    for term, counts in sorted(conditions.items(), key=lambda x: x[1]["adherent"] + x[1]["non_adherent"], reverse=True):
        total = counts["adherent"] + counts["non_adherent"]
        rate = counts["adherent"] / total if total > 0 else 0.0
        condition_rows.append((term, total, counts["adherent"], counts["non_adherent"], rate))

    # Per-patient detail
    patient_rows = []
    for r in results:
        pat_id = r.patient.pat_id if r.patient else "Unknown"
        diagnosis_details = []
        for ds in _parse_details(r.details_json):
            diagnosis_details.append({
                "diagnosis": ds.get("diagnosis", "Unknown"),
                "score": ds.get("score", ""),
                "explanation": ds.get("explanation", ""),
                "followed": ds.get("guidelines_followed", []),
                "not_followed": ds.get("guidelines_not_followed", []),
            })
        patient_rows.append({
            "pat_id": pat_id,
            "overall_score": r.overall_score,
            "diagnoses_found": r.diagnoses_found,
            "adherent": r.guidelines_followed,
            "non_adherent": r.guidelines_not_followed,
            "details": diagnosis_details,
        })

    # Job info
    job_info = ""
    if job_id:
        job = await _get_job_info(session, job_id)
        if job:
            job_info = f"Batch Job #{job.id} &mdash; {job.status}"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build HTML
    html = _build_html(
        generated_at=generated_at,
        job_info=job_info,
        total_patients=total_patients,
        mean_score=mean_score,
        median_score=median_score,
        min_score=min_score,
        max_score=max_score,
        condition_rows=condition_rows,
        patient_rows=patient_rows,
    )

    return html


def _score_class(score: float | None) -> str:
    """Return a CSS class name based on score value."""
    if score is None:
        return "score-na"
    if score >= 0.7:
        return "score-good"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _score_badge(score) -> str:
    """Return a score badge HTML for +1/-1 values."""
    if score == 1:
        return '<span class="badge badge-good">+1 Adherent</span>'
    if score == -1:
        return '<span class="badge badge-bad">-1 Non-adherent</span>'
    return '<span class="badge">N/A</span>'


def _build_html(
    *,
    generated_at: str,
    job_info: str,
    total_patients: int,
    mean_score: float,
    median_score: float,
    min_score: float,
    max_score: float,
    condition_rows: list[tuple],
    patient_rows: list[dict],
) -> str:
    """Build the complete HTML report string."""

    # Condition breakdown rows
    condition_html = ""
    for term, total, adherent, non_adherent, rate in condition_rows:
        bar_width = int(rate * 100)
        condition_html += f"""
        <tr>
            <td>{term}</td>
            <td>{total}</td>
            <td>{adherent}</td>
            <td>{non_adherent}</td>
            <td>
                <div class="bar-container">
                    <div class="bar {_score_class(rate)}" style="width:{bar_width}%"></div>
                </div>
                {rate:.0%}
            </td>
        </tr>"""

    # Patient detail cards
    patient_html = ""
    for p in patient_rows:
        score_pct = f"{p['overall_score']:.0%}" if p["overall_score"] is not None else "N/A"
        css_class = _score_class(p["overall_score"])

        diagnosis_rows = ""
        for d in p["details"]:
            followed = ", ".join(d["followed"]) if d["followed"] else "None"
            not_followed = ", ".join(d["not_followed"]) if d["not_followed"] else "None"
            diagnosis_rows += f"""
            <div class="diagnosis-card">
                <div class="diagnosis-header">
                    <strong>{d['diagnosis']}</strong>
                    {_score_badge(d['score'])}
                </div>
                <p class="explanation">{d['explanation']}</p>
                <div class="guideline-tags">
                    <span class="tag tag-followed">Followed: {followed}</span>
                    <span class="tag tag-not-followed">Not followed: {not_followed}</span>
                </div>
            </div>"""

        patient_html += f"""
        <div class="patient-card">
            <div class="patient-header">
                <div>
                    <h3>{p['pat_id'][:12]}...</h3>
                    <span class="meta">{p['diagnoses_found']} diagnoses &middot; {p['adherent']} adherent &middot; {p['non_adherent']} non-adherent</span>
                </div>
                <div class="score-circle {css_class}">{score_pct}</div>
            </div>
            {diagnosis_rows}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GuidelineGuard Audit Report</title>
<style>
    :root {{
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --text-light: #64748b;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        padding: 2rem;
        max-width: 1100px;
        margin: 0 auto;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1rem; margin: 0; }}
    .subtitle {{ color: var(--text-light); font-size: 0.9rem; margin-bottom: 2rem; }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    .stat-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }}
    .stat-value {{ font-size: 2rem; font-weight: 700; }}
    .stat-label {{ color: var(--text-light); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .score-good {{ color: var(--green); }}
    .score-mid {{ color: var(--amber); }}
    .score-low {{ color: var(--red); }}
    .score-na {{ color: var(--text-light); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: #f1f5f9; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    tr:last-child td {{ border-bottom: none; }}
    .bar-container {{ display: inline-block; width: 80px; height: 8px; background: #e2e8f0; border-radius: 4px; vertical-align: middle; margin-right: 0.5rem; }}
    .bar {{ height: 100%; border-radius: 4px; }}
    .bar.score-good {{ background: var(--green); }}
    .bar.score-mid {{ background: var(--amber); }}
    .bar.score-low {{ background: var(--red); }}
    .patient-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}
    .patient-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}
    .meta {{ color: var(--text-light); font-size: 0.85rem; }}
    .score-circle {{
        width: 56px; height: 56px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.95rem;
        border: 3px solid currentColor;
    }}
    .diagnosis-card {{
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }}
    .diagnosis-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-good {{ background: #dcfce7; color: var(--green); }}
    .badge-bad {{ background: #fef2f2; color: var(--red); }}
    .explanation {{ font-size: 0.9rem; color: var(--text-light); margin-bottom: 0.5rem; }}
    .guideline-tags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .tag {{
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }}
    .tag-followed {{ background: #f0fdf4; color: var(--green); }}
    .tag-not-followed {{ background: #fef2f2; color: var(--red); }}
    .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-light); font-size: 0.8rem; text-align: center; }}
    @media print {{
        body {{ padding: 1rem; }}
        .patient-card {{ break-inside: avoid; }}
    }}
</style>
</head>
<body>

<h1>GuidelineGuard Audit Report</h1>
<p class="subtitle">Generated {generated_at} {('&mdash; ' + job_info) if job_info else ''}</p>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_patients}</div>
        <div class="stat-label">Patients Audited</div>
    </div>
    <div class="stat-card">
        <div class="stat-value {_score_class(mean_score)}">{mean_score:.0%}</div>
        <div class="stat-label">Mean Adherence</div>
    </div>
    <div class="stat-card">
        <div class="stat-value {_score_class(median_score)}">{median_score:.0%}</div>
        <div class="stat-label">Median Adherence</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{min_score:.0%} – {max_score:.0%}</div>
        <div class="stat-label">Score Range</div>
    </div>
</div>

<h2>Adherence by Condition</h2>
<table>
    <thead>
        <tr>
            <th>Diagnosis</th>
            <th>Cases</th>
            <th>Adherent</th>
            <th>Non-adherent</th>
            <th>Adherence Rate</th>
        </tr>
    </thead>
    <tbody>
        {condition_html if condition_html else '<tr><td colspan="5" style="text-align:center;color:var(--text-light)">No condition data available</td></tr>'}
    </tbody>
</table>

<h2>Patient Results</h2>
{patient_html if patient_html else '<p style="color:var(--text-light)">No patient results available</p>'}

<div class="footer">
    GuidelineGuard &mdash; MSK Clinical Guideline Adherence Audit System
</div>

</body>
</html>"""
