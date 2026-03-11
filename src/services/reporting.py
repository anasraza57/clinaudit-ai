"""
Reporting service — aggregate analytics over audit results.

Provides functions for dashboard statistics, condition breakdowns,
non-adherent case listing, and score distributions. Designed to
be extended with gold-standard validation metrics later.
"""

import json
import logging
import math
import statistics as stats

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditResult

logger = logging.getLogger(__name__)


# ── Private helpers ───────────────────────────────────────────────────


async def _load_completed_results(
    session: AsyncSession,
    job_id: int | None = None,
    include_details: bool = False,
) -> list[AuditResult]:
    """
    Shared query helper for loading completed AuditResults.

    Args:
        session: Async database session.
        job_id: Optional job ID to scope results to a batch run.
        include_details: If True, eager-loads the Patient relationship
            (needed when the response requires pat_id).

    Returns:
        List of AuditResult objects with status='completed'.
    """
    query = select(AuditResult).where(AuditResult.status == "completed")

    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)

    if include_details:
        query = query.options(selectinload(AuditResult.patient))

    result = await session.execute(query)
    return list(result.scalars().all())


# ── Public functions ──────────────────────────────────────────────────


async def get_dashboard_stats(
    session: AsyncSession,
    job_id: int | None = None,
) -> dict:
    """
    High-level summary statistics.

    Returns total audited/failed counts, mean/median/min/max adherence
    score, and failure rate. Uses SQL columns only (no JSON parsing).
    """
    base_filters: list = []
    if job_id is not None:
        base_filters.append(AuditResult.job_id == job_id)

    # Count completed
    q_completed = select(func.count(AuditResult.id)).where(
        AuditResult.status == "completed", *base_filters,
    )
    total_completed = (await session.execute(q_completed)).scalar() or 0

    # Count failed
    q_failed = select(func.count(AuditResult.id)).where(
        AuditResult.status == "failed", *base_filters,
    )
    total_failed = (await session.execute(q_failed)).scalar() or 0

    # Score aggregates (completed results with non-null scores)
    score_filters = [
        AuditResult.status == "completed",
        AuditResult.overall_score.isnot(None),
        *base_filters,
    ]

    q_stats = select(
        func.avg(AuditResult.overall_score),
        func.min(AuditResult.overall_score),
        func.max(AuditResult.overall_score),
    ).where(*score_filters)

    stats_row = (await session.execute(q_stats)).one()
    avg_score, min_score, max_score = stats_row

    # Median (no SQL standard — compute in Python)
    q_scores = (
        select(AuditResult.overall_score)
        .where(*score_filters)
        .order_by(AuditResult.overall_score)
    )
    scores = [row[0] for row in (await session.execute(q_scores)).all()]

    median_score = None
    if scores:
        n = len(scores)
        if n % 2 == 0:
            median_score = (scores[n // 2 - 1] + scores[n // 2]) / 2
        else:
            median_score = scores[n // 2]

    total = total_completed + total_failed
    failure_rate = total_failed / total if total > 0 else 0.0

    return {
        "total_audited": total_completed,
        "total_failed": total_failed,
        "failure_rate": round(failure_rate, 4),
        "score_stats": {
            "mean": round(avg_score, 4) if avg_score is not None else None,
            "median": round(median_score, 4) if median_score is not None else None,
            "min": round(min_score, 4) if min_score is not None else None,
            "max": round(max_score, 4) if max_score is not None else None,
        },
    }


async def get_condition_breakdown(
    session: AsyncSession,
    job_id: int | None = None,
    min_count: int = 1,
    sort_by: str = "count",
) -> list[dict]:
    """
    Adherence rates grouped by diagnosis term.

    Parses details_json for each completed result, extracts the
    per-diagnosis scores, and groups them by diagnosis term.

    Args:
        session: Async database session.
        job_id: Optional job ID to scope results.
        min_count: Minimum number of cases to include a diagnosis.
        sort_by: "count" (descending) or "adherence_rate" (ascending).

    Returns:
        List of dicts with diagnosis, counts, and adherence_rate.
    """
    results = await _load_completed_results(session, job_id, include_details=False)

    conditions: dict[str, dict[str, int]] = {}
    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        for ds in details.get("scores", []):
            term = ds.get("diagnosis", "Unknown")
            if term not in conditions:
                conditions[term] = {
                    "compliant": 0, "partial": 0, "not_relevant": 0,
                    "non_compliant": 0, "risky": 0, "errors": 0,
                }

            score = ds.get("score")
            is_new_format = "judgement" in ds

            if is_new_format:
                if score == 2:
                    conditions[term]["compliant"] += 1
                elif score == 1:
                    conditions[term]["partial"] += 1
                elif score == 0:
                    conditions[term]["not_relevant"] += 1
                elif score == -1:
                    conditions[term]["non_compliant"] += 1
                elif score == -2:
                    conditions[term]["risky"] += 1
            else:
                # Legacy binary format: +1 = compliant, -1 = non-compliant
                if score == 1:
                    conditions[term]["compliant"] += 1
                elif score == -1:
                    conditions[term]["non_compliant"] += 1

            if ds.get("error"):
                conditions[term]["errors"] += 1

    breakdown = []
    for term, counts in conditions.items():
        scored = (counts["compliant"] + counts["partial"]
                  + counts["non_compliant"] + counts["risky"])
        if scored < min_count:
            continue
        adherent = counts["compliant"] + counts["partial"]
        non_adherent = counts["non_compliant"] + counts["risky"]
        adherence_rate = adherent / scored if scored > 0 else 0.0
        breakdown.append({
            "diagnosis": term,
            "total_cases": scored,
            "compliant": counts["compliant"],
            "partial": counts["partial"],
            "not_relevant": counts["not_relevant"],
            "non_compliant": counts["non_compliant"],
            "risky": counts["risky"],
            "adherent": adherent,
            "non_adherent": non_adherent,
            "errors": counts["errors"],
            "adherence_rate": round(adherence_rate, 4),
        })

    if sort_by == "adherence_rate":
        breakdown.sort(key=lambda x: x["adherence_rate"])
    else:
        breakdown.sort(key=lambda x: x["total_cases"], reverse=True)

    return breakdown


async def get_non_adherent_cases(
    session: AsyncSession,
    job_id: int | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """
    Paginated list of non-adherent diagnoses for clinical review.

    Returns every diagnosis that scored -1, with the patient ID,
    explanation, and list of guidelines not followed.
    """
    results = await _load_completed_results(session, job_id, include_details=True)

    non_adherent = []
    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        pat_id = r.patient.pat_id if r.patient else details.get("pat_id", "Unknown")

        for ds in details.get("scores", []):
            score = ds.get("score")
            is_new_format = "judgement" in ds
            # New format: -1 and -2 are non-adherent; Legacy: only -1
            is_non_adherent = (
                (is_new_format and score is not None and score <= -1)
                or (not is_new_format and score == -1)
            )
            if is_non_adherent:
                non_adherent.append({
                    "pat_id": pat_id,
                    "diagnosis": ds.get("diagnosis", "Unknown"),
                    "index_date": ds.get("index_date"),
                    "score": score,
                    "judgement": ds.get("judgement", "NON-COMPLIANT"),
                    "confidence": ds.get("confidence"),
                    "explanation": ds.get("explanation", ""),
                    "cited_guideline_text": ds.get("cited_guideline_text", ""),
                    "guidelines_not_followed": ds.get("guidelines_not_followed", []),
                })

    total = len(non_adherent)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = non_adherent[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        "cases": page_data,
    }


async def get_score_distribution(
    session: AsyncSession,
    job_id: int | None = None,
    bins: int = 10,
) -> dict:
    """
    Histogram of patient-level overall_score values.

    Divides the 0.0–1.0 range into equal bins and counts how many
    patients fall into each. Uses SQL columns only (no JSON parsing).
    """
    score_filters = [
        AuditResult.status == "completed",
        AuditResult.overall_score.isnot(None),
    ]
    if job_id is not None:
        score_filters.append(AuditResult.job_id == job_id)

    q_scores = select(AuditResult.overall_score).where(*score_filters)
    scores = [row[0] for row in (await session.execute(q_scores)).all()]

    if not scores:
        return {"bins": [], "total": 0}

    bin_width = 1.0 / bins
    histogram = []
    for i in range(bins):
        bin_start = round(i * bin_width, 4)
        bin_end = round((i + 1) * bin_width, 4)
        if i == bins - 1:
            # Last bin includes the right edge (1.0)
            count = sum(1 for s in scores if bin_start <= s <= bin_end)
        else:
            count = sum(1 for s in scores if bin_start <= s < bin_end)
        histogram.append({
            "bin_start": bin_start,
            "bin_end": bin_end,
            "count": count,
        })

    return {"bins": histogram, "total": len(scores)}


async def get_missing_care_summary(
    session: AsyncSession,
    job_id: int | None = None,
    min_count: int = 1,
) -> dict:
    """
    Aggregate missing care opportunities across all results.

    Parses details_json for missing_care_opportunities field, groups by
    condition, and returns frequency counts. Helps identify systematic
    gaps in documented care.
    """
    results = await _load_completed_results(session, job_id, include_details=True)

    # Collect all opportunities grouped by condition
    by_condition: dict[str, dict[str, int]] = {}
    all_cases: list[dict] = []
    total_opportunities = 0

    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        pat_id = r.patient.pat_id if r.patient else details.get("pat_id", "Unknown")

        for ds in details.get("scores", []):
            opportunities = ds.get("missing_care_opportunities", [])
            if not opportunities:
                continue

            diagnosis = ds.get("diagnosis", "Unknown")
            if diagnosis not in by_condition:
                by_condition[diagnosis] = {}

            for opp in opportunities:
                by_condition[diagnosis][opp] = by_condition[diagnosis].get(opp, 0) + 1
                total_opportunities += 1

            all_cases.append({
                "pat_id": pat_id,
                "diagnosis": diagnosis,
                "index_date": ds.get("index_date"),
                "score": ds.get("score"),
                "missing_care_opportunities": opportunities,
            })

    # Build per-condition summary
    opportunities_by_condition = []
    for condition, opps in sorted(by_condition.items(), key=lambda x: sum(x[1].values()), reverse=True):
        total = sum(opps.values())
        if total < min_count:
            continue
        opportunities_by_condition.append({
            "condition": condition,
            "total_opportunities": total,
            "opportunities": [
                {"action": action, "count": count}
                for action, count in sorted(opps.items(), key=lambda x: x[1], reverse=True)
            ],
        })

    return {
        "total_patients": len(results),
        "total_opportunities": total_opportunities,
        "opportunities_by_condition": opportunities_by_condition,
        "cases": all_cases,
    }


async def compute_system_metrics(
    session: AsyncSession,
    job_id: int,
) -> dict:
    """
    Comprehensive system-level metrics for a batch job.

    Computes score class distribution, adherence rate, confidence
    statistics, and per-class counts — all from stored data without
    any LLM calls.
    """
    results = await _load_completed_results(session, job_id, include_details=False)

    # Count failed results for error rate
    q_failed = select(func.count(AuditResult.id)).where(
        AuditResult.status == "failed",
        AuditResult.job_id == job_id,
    )
    total_failed = (await session.execute(q_failed)).scalar() or 0

    class_dist = {"+2": 0, "+1": 0, "0": 0, "-1": 0, "-2": 0}
    per_class = {
        "compliant": 0, "partial": 0, "not_relevant": 0,
        "non_compliant": 0, "risky": 0, "errors": 0,
    }
    confidences: list[float] = []
    total_diagnoses = 0

    score_to_class = {2: "compliant", 1: "partial", 0: "not_relevant",
                      -1: "non_compliant", -2: "risky"}
    score_to_label = {2: "+2", 1: "+1", 0: "0", -1: "-1", -2: "-2"}

    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        for ds in details.get("scores", []):
            total_diagnoses += 1
            score = ds.get("score")

            if ds.get("error"):
                per_class["errors"] += 1

            if score is not None and score in score_to_label:
                class_dist[score_to_label[score]] += 1
                per_class[score_to_class[score]] += 1

            conf = ds.get("confidence")
            if conf is not None:
                confidences.append(float(conf))

    # Adherence rate: (compliant + partial) / total_scored (excl not_relevant, errors)
    total_scored = (per_class["compliant"] + per_class["partial"]
                    + per_class["non_compliant"] + per_class["risky"])
    adherent = per_class["compliant"] + per_class["partial"]
    adherence_rate = adherent / total_scored if total_scored > 0 else 0.0

    # Confidence statistics
    conf_stats: dict[str, float | None] = {
        "mean": None, "median": None, "min": None, "max": None, "std": None,
    }
    if confidences:
        conf_stats["mean"] = round(stats.mean(confidences), 4)
        conf_stats["median"] = round(stats.median(confidences), 4)
        conf_stats["min"] = round(min(confidences), 4)
        conf_stats["max"] = round(max(confidences), 4)
        conf_stats["std"] = (
            round(stats.stdev(confidences), 4) if len(confidences) > 1 else 0.0
        )

    total_all = len(results) + total_failed
    error_rate = total_failed / total_all if total_all > 0 else 0.0

    return {
        "job_id": job_id,
        "total_patients": len(results),
        "total_diagnoses": total_diagnoses,
        "score_class_distribution": class_dist,
        "adherence_rate": round(adherence_rate, 4),
        "confidence_stats": conf_stats,
        "error_rate": round(error_rate, 4),
        "per_class_counts": per_class,
    }
