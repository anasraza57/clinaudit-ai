"""
Export service — CSV and HTML report generation for audit results.

Generates downloadable reports that can be shared with colleagues,
supervisors, or used for presentations.
"""

import csv
import io
import json
import logging
import math
import os
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditJob, AuditResult

try:
    import cairosvg
except OSError:
    cairosvg = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── SVG Chart Helpers ────────────────────────────────────────────────

# Colour palette matching the CSS badge colours
_CHART_COLOURS = {
    "compliant": "#16a34a",
    "partial": "#d97706",
    "not_relevant": "#94a3b8",
    "non_compliant": "#dc2626",
    "risky": "#be185d",
}

_LEVEL_LABELS = {
    "compliant": "+2 Compliant",
    "partial": "+1 Partial",
    "not_relevant": "0 N/R",
    "non_compliant": "-1 Non-compliant",
    "risky": "-2 Risky",
}


def _svg_score_distribution(scores: list[float], width: int = 480, height: int = 200) -> str:
    """
    Generate an inline SVG bar chart showing patient score distribution.

    Bins scores into 5 buckets: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%.
    """
    if not scores:
        return ""

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    counts = [0] * 5
    for s in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break

    max_count = max(counts) if max(counts) > 0 else 1
    bar_w = 60
    gap = 20
    left_margin = 40
    bottom_margin = 40
    chart_h = height - bottom_margin - 20
    total_w = left_margin + len(counts) * (bar_w + gap)
    actual_width = max(width, total_w + 20)

    bars_svg = ""
    colours = ["#dc2626", "#d97706", "#eab308", "#65a30d", "#16a34a"]
    for i, count in enumerate(counts):
        bar_h = int((count / max_count) * chart_h) if max_count > 0 else 0
        x = left_margin + i * (bar_w + gap)
        y = 20 + chart_h - bar_h
        bars_svg += (
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="{colours[i]}" rx="3" opacity="0.85"/>'
        )
        # Count label above bar
        if count > 0:
            bars_svg += (
                f'<text x="{x + bar_w // 2}" y="{y - 5}" '
                f'text-anchor="middle" font-size="12" fill="#1e293b">{count}</text>'
            )
        # Bin label below bar
        bars_svg += (
            f'<text x="{x + bar_w // 2}" y="{height - bottom_margin + 16}" '
            f'text-anchor="middle" font-size="11" fill="#64748b">{bin_labels[i]}</text>'
        )

    # Y-axis ticks
    axis_svg = ""
    for tick_val in range(0, max_count + 1, max(1, max_count // 4)):
        tick_y = 20 + chart_h - int((tick_val / max_count) * chart_h)
        axis_svg += (
            f'<text x="{left_margin - 8}" y="{tick_y + 4}" '
            f'text-anchor="end" font-size="11" fill="#64748b">{tick_val}</text>'
            f'<line x1="{left_margin}" y1="{tick_y}" '
            f'x2="{actual_width - 10}" y2="{tick_y}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {actual_width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{axis_svg}{bars_svg}</svg>'
    )


def _svg_compliance_donut(
    level_counts: dict[str, int],
    size: int = 220,
) -> str:
    """
    Generate an inline SVG donut chart for compliance level distribution.

    level_counts keys: compliant, partial, not_relevant, non_compliant, risky
    """
    total = sum(level_counts.values())
    if total == 0:
        return ""

    cx = cy = size // 2
    r = (size // 2) - 30
    inner_r = r * 0.55
    circumference = 2 * 3.14159 * r

    slices_svg = ""
    legend_svg = ""
    offset = 0

    for key in ["compliant", "partial", "not_relevant", "non_compliant", "risky"]:
        count = level_counts.get(key, 0)
        if count == 0:
            continue
        pct = count / total
        dash = pct * circumference
        gap = circumference - dash
        colour = _CHART_COLOURS[key]

        slices_svg += (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
            f'stroke="{colour}" stroke-width="{r - inner_r}" '
            f'stroke-dasharray="{dash:.2f} {gap:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"/>'
        )
        offset += dash

    # Centre label
    slices_svg += (
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'font-size="22" font-weight="700" fill="#1e293b">{total}</text>'
        f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" '
        f'font-size="11" fill="#64748b">diagnoses</text>'
    )

    # Legend (below the donut)
    legend_y = size + 5
    lx = 10
    for key in ["compliant", "partial", "not_relevant", "non_compliant", "risky"]:
        count = level_counts.get(key, 0)
        if count == 0:
            continue
        colour = _CHART_COLOURS[key]
        label = _LEVEL_LABELS[key]
        legend_svg += (
            f'<rect x="{lx}" y="{legend_y}" width="10" height="10" '
            f'fill="{colour}" rx="2"/>'
            f'<text x="{lx + 14}" y="{legend_y + 9}" '
            f'font-size="11" fill="#1e293b">{label}: {count}</text>'
        )
        legend_y += 18

    total_height = size + 10 + len([k for k in level_counts if level_counts.get(k, 0) > 0]) * 18

    return (
        f'<svg class="chart" viewBox="0 0 {size} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{slices_svg}{legend_svg}</svg>'
    )


def _svg_condition_bars(
    condition_rows: list[tuple],
    width: int = 520,
    bar_height: int = 24,
) -> str:
    """
    Generate an inline SVG horizontal bar chart for per-condition adherence.

    condition_rows: list of (term, total, adherent, non_adherent, rate)
    """
    if not condition_rows:
        return ""

    label_width = 180
    bar_area = width - label_width - 60
    gap = 8
    top_margin = 5
    n = len(condition_rows)
    total_height = top_margin + n * (bar_height + gap) + 10

    bars_svg = ""
    for i, (term, total, adherent, non_adherent, rate) in enumerate(condition_rows):
        y = top_margin + i * (bar_height + gap)
        bar_w = int(rate * bar_area)

        # Truncate long labels
        display_term = term if len(term) <= 25 else term[:22] + "..."

        # Label
        bars_svg += (
            f'<text x="{label_width - 8}" y="{y + bar_height // 2 + 4}" '
            f'text-anchor="end" font-size="12" fill="#1e293b">{display_term}</text>'
        )
        # Background bar
        bars_svg += (
            f'<rect x="{label_width}" y="{y}" width="{bar_area}" '
            f'height="{bar_height}" fill="#f1f5f9" rx="4"/>'
        )
        # Filled bar
        if bar_w > 0:
            colour = "#16a34a" if rate >= 0.7 else "#d97706" if rate >= 0.4 else "#dc2626"
            bars_svg += (
                f'<rect x="{label_width}" y="{y}" width="{bar_w}" '
                f'height="{bar_height}" fill="{colour}" rx="4" opacity="0.85"/>'
            )
        # Rate label
        bars_svg += (
            f'<text x="{label_width + bar_area + 6}" y="{y + bar_height // 2 + 4}" '
            f'font-size="12" fill="#64748b">{rate:.0%}</text>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{bars_svg}</svg>'
    )


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


# ── Chart Data Collection ────────────────────────────────────────────


async def _collect_chart_data(
    session: AsyncSession,
    job_id: int | None = None,
) -> tuple[list[float], dict[str, int], list[tuple]]:
    """
    Collect scores, level_counts, and condition_rows for chart generation.

    Returns (scores, level_counts, condition_rows) where:
    - scores: list of patient-level overall_score floats
    - level_counts: dict with keys compliant/partial/not_relevant/non_compliant/risky
    - condition_rows: list of (term, total, adherent, non_adherent, rate) tuples
    """
    results = await _load_results_with_patients(session, job_id)

    scores = [r.overall_score for r in results if r.overall_score is not None]

    conditions: dict[str, dict[str, int]] = {}
    level_counts: dict[str, int] = {
        "compliant": 0, "partial": 0, "not_relevant": 0,
        "non_compliant": 0, "risky": 0,
    }
    for r in results:
        for ds in _parse_details(r.details_json):
            term = ds.get("diagnosis", "Unknown")
            if term not in conditions:
                conditions[term] = {
                    "compliant": 0, "partial": 0, "not_relevant": 0,
                    "non_compliant": 0, "risky": 0,
                }

            score = ds.get("score")
            is_new_format = "judgement" in ds

            if is_new_format:
                if score == 2:
                    conditions[term]["compliant"] += 1
                    level_counts["compliant"] += 1
                elif score == 1:
                    conditions[term]["partial"] += 1
                    level_counts["partial"] += 1
                elif score == 0:
                    conditions[term]["not_relevant"] += 1
                    level_counts["not_relevant"] += 1
                elif score == -1:
                    conditions[term]["non_compliant"] += 1
                    level_counts["non_compliant"] += 1
                elif score == -2:
                    conditions[term]["risky"] += 1
                    level_counts["risky"] += 1
            else:
                # Legacy binary format
                if score == 1:
                    conditions[term]["compliant"] += 1
                    level_counts["compliant"] += 1
                elif score == -1:
                    conditions[term]["non_compliant"] += 1
                    level_counts["non_compliant"] += 1

    condition_rows = []
    for term, counts in sorted(
        conditions.items(),
        key=lambda x: (
            x[1]["compliant"] + x[1]["partial"]
            + x[1]["non_compliant"] + x[1]["risky"]
        ),
        reverse=True,
    ):
        adherent = counts["compliant"] + counts["partial"]
        non_adherent = counts["non_compliant"] + counts["risky"]
        total = adherent + non_adherent
        rate = adherent / total if total > 0 else 0.0
        condition_rows.append((term, total, adherent, non_adherent, rate))

    return scores, level_counts, condition_rows


# ── Comparison Chart Helpers ──────────────────────────────────────────


def _svg_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    width: int = 400,
    height: int = 400,
    title: str = "Score Agreement Matrix",
) -> str:
    """Generate a heatmap SVG for a confusion matrix."""
    n = len(labels)
    if n == 0:
        return ""

    margin_left = 60
    margin_top = 60
    margin_right = 20
    margin_bottom = 40
    cell_w = (width - margin_left - margin_right) / n
    cell_h = (height - margin_top - margin_bottom) / n

    max_val = max(max(row) for row in matrix) if matrix else 1
    if max_val == 0:
        max_val = 1

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:system-ui,sans-serif;font-size:12px">',
        f'<text x="{width / 2}" y="18" text-anchor="middle" '
        f'font-weight="600" font-size="14">{title}</text>',
        # Axis labels
        f'<text x="{margin_left + (width - margin_left - margin_right) / 2}" '
        f'y="{height - 8}" text-anchor="middle" font-size="11" fill="#64748b">'
        f'Model B Score</text>',
        f'<text x="14" y="{margin_top + (height - margin_top - margin_bottom) / 2}" '
        f'text-anchor="middle" font-size="11" fill="#64748b" '
        f'transform="rotate(-90,14,'
        f'{margin_top + (height - margin_top - margin_bottom) / 2})">'
        f'Model A Score</text>',
    ]

    # Column headers
    for j, label in enumerate(labels):
        cx = margin_left + j * cell_w + cell_w / 2
        svg_parts.append(
            f'<text x="{cx}" y="{margin_top - 8}" text-anchor="middle" '
            f'font-size="11" fill="#64748b">{label}</text>'
        )

    # Row headers + cells
    for i, row_label in enumerate(labels):
        cy = margin_top + i * cell_h + cell_h / 2
        svg_parts.append(
            f'<text x="{margin_left - 8}" y="{cy + 4}" text-anchor="end" '
            f'font-size="11" fill="#64748b">{row_label}</text>'
        )
        for j in range(n):
            val = matrix[i][j]
            intensity = val / max_val
            x = margin_left + j * cell_w
            y = margin_top + i * cell_h

            # Diagonal = green tint, off-diagonal = blue tint
            if i == j:
                r = int(230 - intensity * 180)
                g = int(245 - intensity * 60)
                b = int(230 - intensity * 180)
                fill = f"rgb({r},{g},{b})"
            else:
                r = int(245 - intensity * 50)
                g = int(245 - intensity * 80)
                b = int(255 - intensity * 30)
                fill = f"rgb({r},{g},{b})"

            text_color = "#1e293b" if intensity < 0.7 else "#fff"
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="#e2e8f0" stroke-width="1"/>'
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" '
                f'text-anchor="middle" fill="{text_color}" '
                f'font-weight="600">{val}</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _svg_comparison_scores(
    scores_a: dict[str, int],
    scores_b: dict[str, int],
    label_a: str = "Model A",
    label_b: str = "Model B",
    width: int = 520,
    height: int = 280,
) -> str:
    """Generate a grouped bar chart comparing score class distributions."""
    classes = ["+2", "+1", "0", "-1", "-2"]
    color_a = "#3b82f6"  # blue
    color_b = "#f97316"  # orange

    margin_left = 50
    margin_top = 50
    margin_right = 20
    margin_bottom = 50
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    values_a = [scores_a.get(c, 0) for c in classes]
    values_b = [scores_b.get(c, 0) for c in classes]
    max_val = max(max(values_a), max(values_b), 1)

    group_w = chart_w / len(classes)
    bar_w = group_w * 0.35
    gap = group_w * 0.05

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:system-ui,sans-serif;font-size:12px">',
        f'<text x="{width / 2}" y="16" text-anchor="middle" '
        f'font-weight="600" font-size="14">Score Class Distribution</text>',
        # Legend
        f'<rect x="{width / 2 - 100}" y="26" width="12" height="12" '
        f'fill="{color_a}" rx="2"/>',
        f'<text x="{width / 2 - 84}" y="37" font-size="11">{label_a}</text>',
        f'<rect x="{width / 2 + 10}" y="26" width="12" height="12" '
        f'fill="{color_b}" rx="2"/>',
        f'<text x="{width / 2 + 26}" y="37" font-size="11">{label_b}</text>',
        # Y-axis line
        f'<line x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + chart_h}" '
        f'stroke="#cbd5e1" stroke-width="1"/>',
        # X-axis line
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" '
        f'x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" '
        f'stroke="#cbd5e1" stroke-width="1"/>',
    ]

    # Y-axis ticks
    for i in range(5):
        tick_val = int(max_val * i / 4)
        tick_y = margin_top + chart_h - (chart_h * i / 4)
        svg_parts.append(
            f'<text x="{margin_left - 8}" y="{tick_y + 4}" '
            f'text-anchor="end" font-size="10" fill="#94a3b8">{tick_val}</text>'
            f'<line x1="{margin_left}" y1="{tick_y}" '
            f'x2="{margin_left + chart_w}" y2="{tick_y}" '
            f'stroke="#f1f5f9" stroke-width="1"/>'
        )

    for i, cls in enumerate(classes):
        x_group = margin_left + i * group_w
        va = values_a[i]
        vb = values_b[i]
        h_a = (va / max_val) * chart_h if max_val > 0 else 0
        h_b = (vb / max_val) * chart_h if max_val > 0 else 0

        # Bar A
        x_a = x_group + gap
        y_a = margin_top + chart_h - h_a
        svg_parts.append(
            f'<rect x="{x_a}" y="{y_a}" width="{bar_w}" height="{h_a}" '
            f'fill="{color_a}" rx="2"/>'
        )
        if va > 0:
            svg_parts.append(
                f'<text x="{x_a + bar_w / 2}" y="{y_a - 4}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{va}</text>'
            )

        # Bar B
        x_b = x_a + bar_w + gap
        y_b = margin_top + chart_h - h_b
        svg_parts.append(
            f'<rect x="{x_b}" y="{y_b}" width="{bar_w}" height="{h_b}" '
            f'fill="{color_b}" rx="2"/>'
        )
        if vb > 0:
            svg_parts.append(
                f'<text x="{x_b + bar_w / 2}" y="{y_b - 4}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{vb}</text>'
            )

        # X-axis label
        svg_parts.append(
            f'<text x="{x_group + group_w / 2}" '
            f'y="{margin_top + chart_h + 18}" text-anchor="middle" '
            f'font-size="11">{cls}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _svg_comparison_compliance(
    levels_a: dict[str, int],
    levels_b: dict[str, int],
    label_a: str = "Model A",
    label_b: str = "Model B",
    size: int = 180,
) -> str:
    """Generate side-by-side donut charts comparing compliance breakdowns."""
    width = size * 2 + 80
    height = size + 80

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:system-ui,sans-serif;font-size:12px">',
    ]

    for idx, (levels, label) in enumerate([(levels_a, label_a), (levels_b, label_b)]):
        cx = size / 2 + 20 + idx * (size + 40)
        cy = size / 2 + 30
        r_outer = size / 2 - 10
        r_inner = r_outer * 0.55

        # Title
        svg_parts.append(
            f'<text x="{cx}" y="18" text-anchor="middle" '
            f'font-weight="600" font-size="13">{label}</text>'
        )

        total = sum(levels.values())
        if total == 0:
            svg_parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_outer}" '
                f'fill="#f1f5f9" stroke="#e2e8f0"/>'
            )
            continue

        # Draw arcs
        import math
        angle = -math.pi / 2  # start at top
        order = ["compliant", "partial", "not_relevant", "non_compliant", "risky"]
        for key in order:
            count = levels.get(key, 0)
            if count == 0:
                continue
            sweep = 2 * math.pi * count / total
            end_angle = angle + sweep

            x1_o = cx + r_outer * math.cos(angle)
            y1_o = cy + r_outer * math.sin(angle)
            x2_o = cx + r_outer * math.cos(end_angle)
            y2_o = cy + r_outer * math.sin(end_angle)
            x1_i = cx + r_inner * math.cos(end_angle)
            y1_i = cy + r_inner * math.sin(end_angle)
            x2_i = cx + r_inner * math.cos(angle)
            y2_i = cy + r_inner * math.sin(angle)

            large = 1 if sweep > math.pi else 0
            color = _CHART_COLOURS.get(key, "#94a3b8")

            svg_parts.append(
                f'<path d="M {x1_o} {y1_o} '
                f'A {r_outer} {r_outer} 0 {large} 1 {x2_o} {y2_o} '
                f'L {x1_i} {y1_i} '
                f'A {r_inner} {r_inner} 0 {large} 0 {x2_i} {y2_i} Z" '
                f'fill="{color}"/>'
            )
            angle = end_angle

        # Center text
        svg_parts.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'font-weight="700" font-size="16">{total}</text>'
        )

    # Shared legend at bottom
    legend_y = height - 16
    legend_items = [
        ("compliant", "+2"), ("partial", "+1"), ("not_relevant", "0"),
        ("non_compliant", "-1"), ("risky", "-2"),
    ]
    legend_x = 20
    for key, short_label in legend_items:
        color = _CHART_COLOURS.get(key, "#94a3b8")
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 8}" width="10" height="10" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{legend_x + 14}" y="{legend_y}" '
            f'font-size="10" fill="#64748b">{short_label}</text>'
        )
        legend_x += 55

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# ── PNG Chart Export ─────────────────────────────────────────────────


async def export_charts_to_png(
    session: AsyncSession,
    output_dir: str,
    job_id: int | None = None,
    dpi: int = 200,
) -> list[str]:
    """
    Generate chart PNG files and save to output_dir.

    Produces up to 3 files:
    - score_distribution.png — bar chart of patient score distribution
    - compliance_breakdown.png — donut chart of compliance levels
    - condition_adherence.png — horizontal bar chart of per-condition rates

    Returns list of saved file paths (only for charts with data).

    Requires cairosvg (pip install cairosvg) and the system cairo library.
    """
    if cairosvg is None:
        raise RuntimeError(
            "cairosvg is not available. Install it with: "
            "pip install cairosvg (and brew install cairo on macOS)",
        )

    scores, level_counts, condition_rows = await _collect_chart_data(
        session, job_id,
    )

    charts = [
        ("score_distribution.png", _svg_score_distribution(scores)),
        ("compliance_breakdown.png", _svg_compliance_donut(level_counts)),
        ("condition_adherence.png", _svg_condition_bars(condition_rows)),
    ]

    saved: list[str] = []
    for filename, svg_str in charts:
        if not svg_str:
            continue

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            dpi=dpi,
        )
        with open(filepath, "wb") as f:
            f.write(png_bytes)
        saved.append(filepath)
        logger.info("Saved chart: %s", filepath)

    return saved


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
        "judgement",
        "confidence",
        "explanation",
        "cited_guideline_text",
        "guidelines_followed",
        "guidelines_not_followed",
        "missing_care_opportunities",
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
                ds.get("judgement", ""),
                ds.get("confidence", ""),
                ds.get("explanation", ""),
                ds.get("cited_guideline_text", ""),
                "; ".join(ds.get("guidelines_followed", [])),
                "; ".join(ds.get("guidelines_not_followed", [])),
                "; ".join(ds.get("missing_care_opportunities", [])),
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

    # Collect chart data (scores, level counts, condition breakdown)
    scores, level_counts, condition_rows = await _collect_chart_data(
        session, job_id,
    )

    # Compute summary stats
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

    # Per-patient detail
    patient_rows = []
    for r in results:
        pat_id = r.patient.pat_id if r.patient else "Unknown"
        diagnosis_details = []
        for ds in _parse_details(r.details_json):
            diagnosis_details.append({
                "diagnosis": ds.get("diagnosis", "Unknown"),
                "score": ds.get("score", ""),
                "judgement": ds.get("judgement", ""),
                "confidence": ds.get("confidence"),
                "cited_guideline_text": ds.get("cited_guideline_text", ""),
                "explanation": ds.get("explanation", ""),
                "followed": ds.get("guidelines_followed", []),
                "not_followed": ds.get("guidelines_not_followed", []),
                "missing_care": ds.get("missing_care_opportunities", []),
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
        scores=scores,
        level_counts=level_counts,
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


def _score_badge(score, judgement: str = "") -> str:
    """Return a score badge HTML for the 5-level grading scale."""
    _BADGE_MAP = {
        2: ("badge-compliant", "+2 Compliant"),
        1: ("badge-partial", "+1 Partial"),
        0: ("badge-neutral", "0 N/R"),
        -1: ("badge-bad", "-1 Non-compliant"),
        -2: ("badge-risky", "-2 Risky"),
    }
    if score in _BADGE_MAP:
        css_class, label = _BADGE_MAP[score]
        return f'<span class="badge {css_class}">{label}</span>'
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
    scores: list[float] | None = None,
    level_counts: dict[str, int] | None = None,
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

    # Generate SVG charts
    score_dist_chart = _svg_score_distribution(scores or [])
    donut_chart = _svg_compliance_donut(level_counts or {})
    condition_bar_chart = _svg_condition_bars(condition_rows)

    # Patient detail cards
    patient_html = ""
    for p in patient_rows:
        score_pct = f"{p['overall_score']:.0%}" if p["overall_score"] is not None else "N/A"
        css_class = _score_class(p["overall_score"])

        diagnosis_rows = ""
        for d in p["details"]:
            followed = ", ".join(d["followed"]) if d["followed"] else "None"
            not_followed = ", ".join(d["not_followed"]) if d["not_followed"] else "None"
            missing_care = d.get("missing_care", [])
            confidence_html = ""
            if d.get("confidence") is not None:
                conf_pct = f"{d['confidence']:.0%}"
                confidence_html = f'<span class="confidence">Confidence: {conf_pct}</span>'
            cited_html = ""
            if d.get("cited_guideline_text"):
                cited_html = f'<blockquote class="cited-guideline">{d["cited_guideline_text"]}</blockquote>'
            missing_care_html = ""
            if missing_care:
                missing_care_str = ", ".join(missing_care)
                missing_care_html = f'\n                    <span class="tag tag-missing-care">Missing care: {missing_care_str}</span>'
            diagnosis_rows += f"""
            <div class="diagnosis-card">
                <div class="diagnosis-header">
                    <strong>{d['diagnosis']}</strong>
                    <div>{_score_badge(d['score'], d.get('judgement', ''))} {confidence_html}</div>
                </div>
                <p class="explanation">{d['explanation']}</p>
                {cited_html}
                <div class="guideline-tags">
                    <span class="tag tag-followed">Followed: {followed}</span>
                    <span class="tag tag-not-followed">Not followed: {not_followed}</span>{missing_care_html}
                </div>
            </div>"""

        patient_html += f"""
        <div class="patient-card">
            <div class="patient-header">
                <div>
                    <h3>{p['pat_id']}</h3>
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
    .badge-compliant {{ background: #dcfce7; color: var(--green); }}
    .badge-partial {{ background: #fef9c3; color: #a16207; }}
    .badge-neutral {{ background: #f1f5f9; color: var(--text-light); }}
    .badge-bad {{ background: #fef2f2; color: var(--red); }}
    .badge-risky {{ background: #fce7f3; color: #be185d; }}
    .confidence {{ font-size: 0.75rem; color: var(--text-light); margin-left: 0.5rem; }}
    .cited-guideline {{ font-size: 0.85rem; color: var(--text-light); border-left: 3px solid var(--border); padding: 0.4rem 0.8rem; margin: 0.5rem 0; font-style: italic; }}
    .explanation {{ font-size: 0.9rem; color: var(--text-light); margin-bottom: 0.5rem; }}
    .guideline-tags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .tag {{
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }}
    .tag-followed {{ background: #f0fdf4; color: var(--green); }}
    .tag-not-followed {{ background: #fef2f2; color: var(--red); }}
    .tag-missing-care {{ background: #fefce8; color: #a16207; }}
    .chart-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}
    .chart-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
    }}
    .chart-card h3 {{
        font-size: 0.9rem;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }}
    .chart {{ width: 100%; height: auto; }}
    .chart-full {{
        grid-column: 1 / -1;
    }}
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

{f'''<h2>Charts</h2>
<div class="chart-grid">
    <div class="chart-card">
        <h3>Score Distribution</h3>
        {score_dist_chart if score_dist_chart else '<p style="color:var(--text-light)">No score data</p>'}
    </div>
    <div class="chart-card">
        <h3>Compliance Breakdown</h3>
        {donut_chart if donut_chart else '<p style="color:var(--text-light)">No diagnosis data</p>'}
    </div>
    <div class="chart-card chart-full">
        <h3>Adherence by Condition</h3>
        {condition_bar_chart if condition_bar_chart else '<p style="color:var(--text-light)">No condition data</p>'}
    </div>
</div>
''' if (score_dist_chart or donut_chart or condition_bar_chart) else ''}
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


# ── Comprehensive Comparison Report ─────────────────────────────────


def _build_scorer_eval_section(
    scorer_evals: dict | None,
    label_a: str,
    label_b: str,
) -> str:
    """Build HTML section for LLM-as-Judge scorer evaluation results."""
    if not scorer_evals:
        return ""

    def _scorer_row(label, data):
        if not data:
            return f"<tr><td>{label}</td><td colspan='3' style='color:var(--muted)'>Not available</td></tr>"
        return (
            f"<tr><td>{label}</td>"
            f"<td>{data['mean_reasoning_quality']:.2f}</td>"
            f"<td>{data['mean_citation_accuracy']:.2f}</td>"
            f"<td>{data['mean_score_calibration']:.2f}</td></tr>"
        )

    rows = ""
    rows += _scorer_row(
        f"{label_a} (judged by GPT-4o-mini)",
        scorer_evals.get("job_a_openai_judge"),
    )
    rows += _scorer_row(
        f"{label_b} (judged by GPT-4o-mini)",
        scorer_evals.get("job_b_openai_judge"),
    )
    rows += _scorer_row(
        f"{label_a} (judged by mistral-small)",
        scorer_evals.get("job_a_ollama_judge"),
    )
    rows += _scorer_row(
        f"{label_b} (judged by mistral-small)",
        scorer_evals.get("job_b_ollama_judge"),
    )

    # Note about cross-model judging
    note = (
        "Each model's output is evaluated by both judges (GPT-4o-mini and mistral-small) "
        "for cross-model validation. Scores are rated 1–5, where 5 is best."
    )

    return f"""
<!-- ── LLM-as-Judge Scorer Evaluation ─────────────────────────────── -->

<h2>LLM-as-Judge: Scorer Quality</h2>
<p style="color:var(--muted);margin-bottom:0.75rem;">{note}</p>
<table>
    <thead><tr>
        <th>Scorer &rarr; Judge</th>
        <th>Reasoning Quality</th>
        <th>Citation Accuracy</th>
        <th>Score Calibration</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>
"""


def _build_agent_eval_section(agent_eval: dict | None) -> str:
    """Build HTML section for full pipeline agent evaluation results."""
    if not agent_eval:
        return ""

    # Support both single eval dict and dict-of-dicts keyed by judge name
    evals = agent_eval if "openai_judge" in agent_eval or "ollama_judge" in agent_eval else {"judge": agent_eval}

    sections = []
    for judge_label, eval_data in evals.items():
        if not isinstance(eval_data, dict) or "total_patients" not in eval_data:
            continue

        total = eval_data.get("total_patients", 0)
        judge_display = {
            "openai_judge": "GPT-4o-mini",
            "ollama_judge": "mistral-small",
            "judge": "LLM Judge",
        }.get(judge_label, judge_label)

        rows = []
        # Query Generator
        query = eval_data.get("query")
        if query:
            rows.append(f"<tr><td>Query Relevance (1-5)</td><td>{query.get('mean_relevance', 0):.2f}</td></tr>")
            rows.append(f"<tr><td>Query Coverage (1-5)</td><td>{query.get('mean_coverage', 0):.2f}</td></tr>")

        # Retriever IR
        ir = eval_data.get("retriever_ir")
        if ir:
            rows.append(f"<tr><td>Retriever Precision@k</td><td>{ir.get('mean_precision_at_k', 0):.3f}</td></tr>")
            rows.append(f"<tr><td>Retriever Recall@k</td><td>{ir.get('mean_recall_at_k', 0):.3f}</td></tr>")
            rows.append(f"<tr><td>Retriever nDCG</td><td>{ir.get('mean_ndcg', 0):.3f}</td></tr>")
            rows.append(f"<tr><td>Retriever MRR</td><td>{ir.get('mean_mrr', 0):.3f}</td></tr>")
            rows.append(f"<tr><td>Retriever Mean Relevance (1-5)</td><td>{ir.get('mean_relevance', 0):.2f}</td></tr>")

        # Scorer
        scorer = eval_data.get("scorer")
        if scorer:
            rows.append(f"<tr><td>Scorer Reasoning Quality (1-5)</td><td>{scorer.get('mean_reasoning_quality', 0):.2f}</td></tr>")
            rows.append(f"<tr><td>Scorer Citation Accuracy (1-5)</td><td>{scorer.get('mean_citation_accuracy', 0):.2f}</td></tr>")
            rows.append(f"<tr><td>Scorer Calibration (1-5)</td><td>{scorer.get('mean_score_calibration', 0):.2f}</td></tr>")

        rows_html = "\n".join(rows)
        sections.append(f"""
<div>
    <h3 style="font-size:0.95rem;margin-bottom:0.75rem;">Judge: {judge_display} ({total} patients)</h3>
    <table>
        <thead><tr><th>Agent / Metric</th><th>Score</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>""")

    grid_content = "\n".join(sections)
    return f"""
<!-- ── Full Agent Evaluation ──────────────────────────────────────── -->

<h2>Agent-Level Evaluation (Pipeline Run)</h2>
<p style="color:var(--muted);margin-bottom:0.75rem;">
    Full pipeline executed on random patients with LLM-as-Judge rating each
    agent's output. Both GPT-4o-mini and mistral-small served as independent judges.
</p>
<div class="two-col">
{grid_content}
</div>
"""


async def generate_comparison_html(
    session: AsyncSession,
    job_a_id: int,
    job_b_id: int,
    *,
    scorer_evals: dict | None = None,
    agent_eval: dict | None = None,
) -> str:
    """
    Generate a self-contained HTML report comparing two batch jobs.

    Pulls together system metrics, cross-model classification, extractor
    quality, missing care, per-patient comparison, LLM-as-Judge scorer
    evaluation, and full agent evaluation into one page with inline SVG
    charts. No external dependencies — open in any browser.

    Optional parameters:
        scorer_evals: dict with keys like "job_a_openai_judge", "job_b_openai_judge",
            "job_a_ollama_judge", "job_b_ollama_judge" — each a scorer eval response dict.
        agent_eval: dict from the full agent evaluation endpoint.
    """
    from src.services.comparison import compare_jobs, compute_cross_model_classification
    from src.services.evaluation import evaluate_extractor_from_db
    from src.services.reporting import compute_system_metrics, get_missing_care_summary

    # ── Gather all data in parallel-friendly order ──────────────────
    job_a = await _get_job_info(session, job_a_id)
    job_b = await _get_job_info(session, job_b_id)

    label_a = f"{job_a.provider.capitalize() if job_a and job_a.provider else 'Model A'} (Job {job_a_id})"
    label_b = f"{job_b.provider.capitalize() if job_b and job_b.provider else 'Model B'} (Job {job_b_id})"

    metrics_a = await compute_system_metrics(session, job_a_id)
    metrics_b = await compute_system_metrics(session, job_b_id)

    comparison = (await compare_jobs(session, job_a_id, job_b_id)).summary()

    cross = await compute_cross_model_classification(session, job_a_id, job_b_id)

    scores_a, levels_a, conditions_a = await _collect_chart_data(session, job_a_id)
    scores_b, levels_b, conditions_b = await _collect_chart_data(session, job_b_id)

    extractor = await evaluate_extractor_from_db(session)

    missing_a = await get_missing_care_summary(session, job_a_id, min_count=1)
    missing_b = await get_missing_care_summary(session, job_b_id, min_count=1)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Build charts ────────────────────────────────────────────────
    score_dist_svg = _svg_comparison_scores(
        metrics_a["score_class_distribution"],
        metrics_b["score_class_distribution"],
        label_a, label_b,
    )
    compliance_svg = _svg_comparison_compliance(levels_a, levels_b, label_a, label_b)
    confusion_svg = _svg_confusion_matrix(
        cross["confusion_matrix"]["matrix"],
        cross["confusion_matrix"]["labels"],
        title=f"Score Agreement: {label_a} vs {label_b}",
    )

    # ── Helper: build a metric row ──────────────────────────────────
    def _metric_row(label, val_a, val_b, fmt=".1f", suffix="", highlight_higher=True):
        """Build a table row comparing two values with colour hint."""
        if val_a is None or val_b is None:
            return f"<tr><td>{label}</td><td>—</td><td>—</td><td>—</td></tr>"
        diff = val_a - val_b
        if abs(diff) < 0.001:
            cls = ""
        elif (diff > 0) == highlight_higher:
            cls = ' class="val-better"'
        else:
            cls = ' class="val-worse"'
        return (
            f"<tr><td>{label}</td>"
            f"<td>{val_a:{fmt}}{suffix}</td>"
            f"<td>{val_b:{fmt}}{suffix}</td>"
            f"<td{cls}>{diff:+{fmt}}{suffix}</td></tr>"
        )

    # ── Executive summary rows ──────────────────────────────────────
    exec_rows = "".join([
        _metric_row("Total Diagnoses", metrics_a["total_diagnoses"], metrics_b["total_diagnoses"], fmt="d", suffix=""),
        _metric_row("Adherence Rate", metrics_a["adherence_rate"] * 100, metrics_b["adherence_rate"] * 100, suffix="%"),
        _metric_row("Partial (+1)", metrics_a["score_class_distribution"].get("+1", 0), metrics_b["score_class_distribution"].get("+1", 0), fmt="d"),
        _metric_row("Non-compliant (-1)", metrics_a["score_class_distribution"].get("-1", 0), metrics_b["score_class_distribution"].get("-1", 0), fmt="d", highlight_higher=False),
        _metric_row("Not Relevant (0)", metrics_a["score_class_distribution"].get("0", 0), metrics_b["score_class_distribution"].get("0", 0), fmt="d"),
        _metric_row("Risky (-2)", metrics_a["score_class_distribution"].get("-2", 0), metrics_b["score_class_distribution"].get("-2", 0), fmt="d", highlight_higher=False),
        _metric_row("Mean Confidence", metrics_a["confidence_stats"]["mean"], metrics_b["confidence_stats"]["mean"], fmt=".3f"),
        _metric_row("Error Rate", metrics_a["error_rate"] * 100, metrics_b["error_rate"] * 100, suffix="%", highlight_higher=False),
    ])

    # ── Cross-model metrics ─────────────────────────────────────────
    cross_rows = f"""
    <tr><td>Exact-match Accuracy</td><td>{cross['exact_match_accuracy']:.1%}</td></tr>
    <tr><td>Agreement Rate (direction)</td><td>{cross['agreement_rate']:.1%}</td></tr>
    <tr><td>5-class Cohen's Kappa</td><td>{cross['cohen_kappa_5class']:.4f}</td></tr>
    <tr><td>3-class Cohen's Kappa</td><td>{cross['cohen_kappa_3class']:.4f}</td></tr>
    <tr><td>Pearson Correlation</td><td>{cross['pearson_correlation']:.4f}</td></tr>
    <tr><td>AUROC</td><td>{cross['auroc'] if cross['auroc'] is not None else 'N/A'}</td></tr>
    """

    # ── Per-class P/R/F1 ────────────────────────────────────────────
    prf_rows = ""
    for cls_label in ["-2", "-1", "0", "+1", "+2"]:
        m = cross["per_class_metrics"].get(cls_label, {})
        sup = m.get("support", 0)
        if sup == 0:
            continue
        prf_rows += (
            f"<tr><td>{cls_label}</td>"
            f"<td>{m['precision']:.3f}</td>"
            f"<td>{m['recall']:.3f}</td>"
            f"<td>{m['f1']:.3f}</td>"
            f"<td>{sup}</td></tr>"
        )

    # ── Extractor table ─────────────────────────────────────────────
    ext_rows = ""
    for cat, m in sorted(extractor["per_category"].items()):
        ext_rows += (
            f"<tr><td style='text-transform:capitalize'>{cat}</td>"
            f"<td>{m['precision']:.3f}</td>"
            f"<td>{m['recall']:.3f}</td>"
            f"<td>{m['f1']:.3f}</td>"
            f"<td>{m['tp']}</td></tr>"
        )

    # ── Missing care top items ──────────────────────────────────────
    def _top_missing(summary, n=10):
        rows = ""
        for cond in summary.get("opportunities_by_condition", [])[:5]:
            for opp in cond.get("opportunities", [])[:3]:
                rows += f"<tr><td>{cond['condition']}</td><td>{opp['action']}</td><td>{opp['count']}</td></tr>"
        return rows

    missing_a_rows = _top_missing(missing_a)
    missing_b_rows = _top_missing(missing_b)

    # ── Per-condition comparison ────────────────────────────────────
    per_cond_rows = ""
    for c in comparison.get("per_condition", []):
        diff = c["diff"]
        cls = "val-better" if diff > 0.01 else ("val-worse" if diff < -0.01 else "")
        per_cond_rows += (
            f"<tr><td>{c['condition']}</td>"
            f"<td>{c['count']}</td>"
            f"<td>{c['adherence_rate_a']:.0%}</td>"
            f"<td>{c['adherence_rate_b']:.0%}</td>"
            f"<td class='{cls}'>{diff:+.0%}</td></tr>"
        )

    # ── Per-patient comparison (top rows) ───────────────────────────
    patient_rows = ""
    for p in comparison.get("patients", []):
        agree_cls = "val-better" if p["agreement"] else "val-worse"
        patient_rows += (
            f"<tr><td title='{p['pat_id']}'>{p['pat_id'][:12]}...</td>"
            f"<td>{p['score_a']:.0%}</td>"
            f"<td>{p['score_b']:.0%}</td>"
            f"<td>{p['score_diff']:+.0%}</td>"
            f"<td>{p['diagnoses_a']}</td>"
            f"<td class='{agree_cls}'>{'Yes' if p['agreement'] else 'No'}</td></tr>"
        )

    # ── Full HTML ───────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GuidelineGuard — Model Comparison Report</title>
<style>
    :root {{
        --green: #16a34a; --amber: #d97706; --red: #dc2626;
        --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
        --text: #1e293b; --muted: #64748b; --blue: #3b82f6; --orange: #f97316;
    }}
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6;
        padding: 2rem; max-width: 1200px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{
        font-size: 1.25rem; margin: 2.5rem 0 1rem;
        border-bottom: 2px solid var(--border); padding-bottom: 0.5rem;
    }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .badge-model {{
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;
    }}
    .badge-a {{ background: #dbeafe; color: var(--blue); }}
    .badge-b {{ background: #ffedd5; color: var(--orange); }}

    /* Stat cards */
    .stats-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem; margin-bottom: 2rem;
    }}
    .stat-card {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 1.25rem; text-align: center;
    }}
    .stat-value {{ font-size: 1.8rem; font-weight: 700; }}
    .stat-label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-sub {{ font-size: 0.85rem; color: var(--muted); }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 1.5rem; }}
    th, td {{ padding: 0.6rem 0.9rem; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
    th {{ background: #f1f5f9; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    tr:last-child td {{ border-bottom: none; }}
    .val-better {{ color: var(--green); font-weight: 600; }}
    .val-worse {{ color: var(--red); font-weight: 600; }}

    /* Charts */
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
    .chart-card {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 1.25rem;
    }}
    .chart-card h3 {{ font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; }}
    .chart-full {{ grid-column: 1 / -1; }}
    .chart {{ width: 100%; height: auto; }}

    /* Two-col layout */
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}

    /* Kappa interpretation */
    .kappa-hint {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; }}

    .footer {{
        margin-top: 3rem; padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--muted); font-size: 0.8rem; text-align: center;
    }}
    @media print {{
        body {{ padding: 1rem; font-size: 0.85rem; }}
        h2 {{ break-before: avoid; }}
        table, .chart-card {{ break-inside: avoid; }}
    }}
    @media (max-width: 768px) {{
        .chart-grid, .two-col {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<h1>Model Comparison Report</h1>
<p class="subtitle">
    <span class="badge-model badge-a">{label_a}</span>
    <span class="badge-model badge-b">{label_b}</span>
    &mdash; {comparison['total_patients_compared']} patients, {cross['total_diagnoses_compared']} diagnoses
    &mdash; Generated {generated_at}
</p>

<!-- ── Executive Summary ─────────────────────────────────────────── -->

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{comparison['total_patients_compared']}</div>
        <div class="stat-label">Patients Compared</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{cross['exact_match_accuracy']:.0%}</div>
        <div class="stat-label">Exact Match</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{cross['agreement_rate']:.0%}</div>
        <div class="stat-label">Direction Agreement</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{cross['cohen_kappa_5class']:.2f}</div>
        <div class="stat-label">Cohen's Kappa (5-class)</div>
        <div class="stat-sub">{_kappa_label(cross['cohen_kappa_5class'])}</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{cross['pearson_correlation']:.2f}</div>
        <div class="stat-label">Pearson Correlation</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{cross['auroc'] if cross['auroc'] is not None else 'N/A'}</div>
        <div class="stat-label">AUROC</div>
    </div>
</div>

<!-- ── Side-by-Side System Metrics ───────────────────────────────── -->

<h2>System-Level Metrics</h2>
<table>
    <thead><tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th><th>Diff</th></tr></thead>
    <tbody>{exec_rows}</tbody>
</table>

<!-- ── Charts ────────────────────────────────────────────────────── -->

<h2>Visual Comparison</h2>
<div class="chart-grid">
    <div class="chart-card">
        <h3>Score Class Distribution</h3>
        {score_dist_svg}
    </div>
    <div class="chart-card">
        <h3>Compliance Breakdown</h3>
        {compliance_svg}
    </div>
    <div class="chart-card chart-full">
        <h3>Confusion Matrix</h3>
        {confusion_svg}
    </div>
</div>

<!-- ── Cross-Model Classification ────────────────────────────────── -->

<h2>Cross-Model Agreement Metrics</h2>
<div class="two-col">
    <div>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{cross_rows}</tbody>
        </table>
        <p class="kappa-hint">Kappa interpretation: &lt;0 poor, 0.0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.0 near-perfect</p>
    </div>
    <div>
        <table>
            <thead><tr><th>Score Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>{prf_rows if prf_rows else '<tr><td colspan="5" style="text-align:center;color:var(--muted)">No overlapping classes</td></tr>'}</tbody>
        </table>
    </div>
</div>

<!-- ── Per-Condition Comparison ───────────────────────────────────── -->

<h2>Adherence by Condition</h2>
<table>
    <thead><tr><th>Condition</th><th>Cases</th><th>{label_a}</th><th>{label_b}</th><th>Diff</th></tr></thead>
    <tbody>{per_cond_rows if per_cond_rows else '<tr><td colspan="5" style="text-align:center;color:var(--muted)">No condition data</td></tr>'}</tbody>
</table>

<!-- ── Extractor Evaluation ──────────────────────────────────────── -->

<h2>Extractor (SNOMED Categorisation) Quality</h2>
<p style="color:var(--muted);margin-bottom:0.75rem;">
    {extractor['total_concepts']} concepts evaluated &middot;
    {extractor['total_with_rules']} with rule-based ground truth &middot;
    Rule match rate: <strong>{extractor['rule_match_rate']:.0%}</strong>
</p>
<table>
    <thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th></tr></thead>
    <tbody>{ext_rows}</tbody>
</table>

{_build_scorer_eval_section(scorer_evals, label_a, label_b)}
{_build_agent_eval_section(agent_eval)}

<!-- ── Missing Care Opportunities ────────────────────────────────── -->

<h2>Missing Care Opportunities</h2>
<div class="two-col">
    <div>
        <h3 style="font-size:0.95rem;margin-bottom:0.75rem;">{label_a} — {missing_a.get('total_opportunities', 0)} total gaps</h3>
        <table>
            <thead><tr><th>Condition</th><th>Missing Action</th><th>Count</th></tr></thead>
            <tbody>{missing_a_rows if missing_a_rows else '<tr><td colspan="3" style="text-align:center;color:var(--muted)">None</td></tr>'}</tbody>
        </table>
    </div>
    <div>
        <h3 style="font-size:0.95rem;margin-bottom:0.75rem;">{label_b} — {missing_b.get('total_opportunities', 0)} total gaps</h3>
        <table>
            <thead><tr><th>Condition</th><th>Missing Action</th><th>Count</th></tr></thead>
            <tbody>{missing_b_rows if missing_b_rows else '<tr><td colspan="3" style="text-align:center;color:var(--muted)">None</td></tr>'}</tbody>
        </table>
    </div>
</div>

<!-- ── Per-Patient Comparison ────────────────────────────────────── -->

<h2>Per-Patient Comparison</h2>
<table>
    <thead><tr>
        <th>Patient ID</th><th>{label_a}</th><th>{label_b}</th>
        <th>Diff</th><th>Diagnoses</th><th>Agree</th>
    </tr></thead>
    <tbody>{patient_rows}</tbody>
</table>

<div class="footer">
    GuidelineGuard &mdash; MSK Clinical Guideline Adherence Audit &mdash; Model Comparison Report
</div>

</body>
</html>"""


def _kappa_label(kappa: float) -> str:
    """Return a human-readable interpretation of Cohen's kappa."""
    if kappa < 0:
        return "Poor"
    if kappa <= 0.20:
        return "Slight"
    if kappa <= 0.40:
        return "Fair"
    if kappa <= 0.60:
        return "Moderate"
    if kappa <= 0.80:
        return "Substantial"
    return "Near-perfect"
