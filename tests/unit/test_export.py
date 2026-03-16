"""
Tests for the export service (CSV and HTML report generation).

Uses an in-memory SQLite database (via aiosqlite) to test actual
query and rendering logic without needing PostgreSQL.
"""

import csv
import io
import json
import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.models.audit import AuditJob, AuditResult
from src.models.base import Base
from src.models.patient import Patient
from src.services.export import (
    _collect_chart_data,
    _svg_comparison_compliance,
    _svg_comparison_scores,
    _svg_compliance_donut,
    _svg_condition_bars,
    _svg_confusion_matrix,
    _svg_score_distribution,
    export_charts_to_png,
    generate_csv,
    generate_html_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def async_session():
    """Create an in-memory SQLite database with all tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False,
    )
    async with factory() as session:
        yield session

    await engine.dispose()


# ── Helpers ───────────────────────────────────────────────────────────


def _make_patient(session: AsyncSession, pat_id: str) -> Patient:
    p = Patient(pat_id=pat_id)
    session.add(p)
    return p


def _make_details_json(pat_id: str, scores: list[dict]) -> str:
    adherent = sum(1 for s in scores if s.get("score") == 1)
    non_adherent = sum(1 for s in scores if s.get("score") == -1)
    total_scored = adherent + non_adherent
    return json.dumps({
        "pat_id": pat_id,
        "total_diagnoses": len(scores),
        "adherent": adherent,
        "non_adherent": non_adherent,
        "errors": 0,
        "aggregate_score": adherent / total_scored if total_scored > 0 else 0,
        "scores": scores,
    })


async def _add_completed_result(
    session: AsyncSession,
    patient: Patient,
    overall_score: float,
    details_scores: list[dict],
    job_id: int | None = None,
) -> AuditResult:
    await session.flush()
    adherent = sum(1 for s in details_scores if s.get("score") == 1)
    non_adherent_count = sum(1 for s in details_scores if s.get("score") == -1)
    result = AuditResult(
        patient_id=patient.id,
        job_id=job_id,
        overall_score=overall_score,
        diagnoses_found=len(details_scores),
        guidelines_followed=adherent,
        guidelines_not_followed=non_adherent_count,
        details_json=_make_details_json(patient.pat_id, details_scores),
        status="completed",
    )
    session.add(result)
    await session.flush()
    return result


async def _make_job(session: AsyncSession) -> AuditJob:
    job = AuditJob(
        status="completed",
        total_patients=2,
        processed_patients=2,
        failed_patients=0,
    )
    session.add(job)
    await session.flush()
    return job


# ── Test: CSV Export ──────────────────────────────────────────────────


class TestCSVExport:

    @pytest.mark.asyncio
    async def test_empty_returns_header_only(self, async_session):
        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # header only
        assert rows[0][0] == "pat_id"

    @pytest.mark.asyncio
    async def test_single_patient_one_diagnosis(self, async_session):
        p = _make_patient(async_session, "PAT-001")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Back pain",
                "index_date": "2024-01-15",
                "score": 1,
                "explanation": "Physio referral made",
                "guidelines_followed": ["Physiotherapy referral"],
                "guidelines_not_followed": [],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        header = rows[0]
        assert "judgement" in header
        assert "confidence" in header
        assert "cited_guideline_text" in header
        data = rows[1]
        assert data[0] == "PAT-001"      # pat_id
        assert data[1] == "1.0"           # overall_score
        assert data[2] == "Back pain"     # diagnosis
        assert data[4] == "1"             # score
        assert "Physio" in data[7]        # explanation (shifted by new columns)

    @pytest.mark.asyncio
    async def test_multiple_diagnoses_per_patient(self, async_session):
        p = _make_patient(async_session, "PAT-002")
        await _add_completed_result(async_session, p, 0.5, [
            {
                "diagnosis": "Hip pain",
                "score": 1,
                "explanation": "Referred to physio",
                "guidelines_followed": ["Physio"],
                "guidelines_not_followed": [],
            },
            {
                "diagnosis": "Shoulder pain",
                "score": -1,
                "explanation": "No actions taken",
                "guidelines_followed": [],
                "guidelines_not_followed": ["Exercise therapy"],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 3  # header + 2 diagnosis rows
        assert rows[1][2] == "Hip pain"
        assert rows[2][2] == "Shoulder pain"

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session):
        p1 = _make_patient(async_session, "PAT-A")
        p2 = _make_patient(async_session, "PAT-B")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])  # no job_id

        csv_str = await generate_csv(async_session, job_id=job.id)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 (only PAT-A)
        assert rows[1][0] == "PAT-A"

    @pytest.mark.asyncio
    async def test_guidelines_joined_with_semicolons(self, async_session):
        p = _make_patient(async_session, "PAT-003")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Knee pain",
                "score": 1,
                "explanation": "Multiple guidelines followed",
                "guidelines_followed": ["Physio referral", "Exercise advice", "Weight management"],
                "guidelines_not_followed": ["Imaging"],
            },
        ])

        csv_str = await generate_csv(async_session)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        followed = rows[1][9]  # guidelines_followed column
        assert "Physio referral" in followed
        assert "Exercise advice" in followed
        assert ";" in followed

        not_followed = rows[1][10]  # guidelines_not_followed column
        assert not_followed == "Imaging"


# ── Test: HTML Report ─────────────────────────────────────────────────


class TestHTMLReport:

    @pytest.mark.asyncio
    async def test_empty_report(self, async_session):
        html = await generate_html_report(async_session)

        assert "<!DOCTYPE html>" in html
        assert "ClinAuditAI Audit Report" in html
        assert "0" in html  # 0 patients

    @pytest.mark.asyncio
    async def test_report_contains_stats(self, async_session):
        for pat_id, score in [("P1", 0.5), ("P2", 1.0)]:
            p = _make_patient(async_session, pat_id)
            await _add_completed_result(async_session, p, score, [
                {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
                 "guidelines_followed": ["Physio"], "guidelines_not_followed": []},
            ])

        html = await generate_html_report(async_session)

        assert "2" in html            # total patients
        assert "Patients Audited" in html
        assert "Mean Adherence" in html
        assert "Median Adherence" in html

    @pytest.mark.asyncio
    async def test_report_contains_conditions(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Low back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "explanation": "No referral",
             "guidelines_followed": [], "guidelines_not_followed": ["Physio"]},
        ])

        html = await generate_html_report(async_session)

        assert "Low back pain" in html
        assert "Knee pain" in html
        assert "Adherence by Condition" in html

    @pytest.mark.asyncio
    async def test_report_contains_patient_details(self, async_session):
        p = _make_patient(async_session, "abc123-test-patient-uuid")
        await _add_completed_result(async_session, p, 1.0, [
            {
                "diagnosis": "Finger pain",
                "score": 1,
                "explanation": "Physiotherapy referral was appropriate",
                "guidelines_followed": ["Physiotherapy referral"],
                "guidelines_not_followed": [],
            },
        ])

        html = await generate_html_report(async_session)

        assert "abc123-test-" in html  # truncated pat_id
        assert "Finger pain" in html
        assert "Physiotherapy referral was appropriate" in html
        assert "+1 Partial" in html  # score 1 now shows as Partial

    @pytest.mark.asyncio
    async def test_report_has_score_badges(self, async_session):
        p = _make_patient(async_session, "P1")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])

        html = await generate_html_report(async_session)

        assert "+1 Partial" in html
        assert "-1 Non-compliant" in html

    @pytest.mark.asyncio
    async def test_report_is_self_contained(self, async_session):
        """HTML report has inline CSS, no external dependencies."""
        html = await generate_html_report(async_session)

        assert "<style>" in html
        # No external stylesheet references
        assert 'rel="stylesheet"' not in html
        assert "<script" not in html

    @pytest.mark.asyncio
    async def test_job_id_scoping(self, async_session):
        p1 = _make_patient(async_session, "IN-JOB")
        p2 = _make_patient(async_session, "NOT-IN-JOB")
        job = await _make_job(async_session)

        await _add_completed_result(async_session, p1, 0.8, [
            {"diagnosis": "Back pain", "score": 1, "explanation": "OK",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.3, [
            {"diagnosis": "Hip pain", "score": -1, "explanation": "Bad",
             "guidelines_followed": [], "guidelines_not_followed": []},
        ])

        html = await generate_html_report(async_session, job_id=job.id)

        assert "IN-JOB" in html
        assert "NOT-IN-JOB" not in html
        assert "1" in html  # 1 patient

    @pytest.mark.asyncio
    async def test_report_contains_svg_charts(self, async_session):
        """HTML report includes SVG chart elements when data is present."""
        p = _make_patient(async_session, "P-CHART")
        await _add_completed_result(async_session, p, 0.8, [
            {"diagnosis": "Back pain", "score": 2, "judgement": "Compliant",
             "explanation": "OK", "guidelines_followed": ["Physio"],
             "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "judgement": "Non-compliant",
             "explanation": "Bad", "guidelines_followed": [],
             "guidelines_not_followed": ["Exercise"]},
        ])

        html = await generate_html_report(async_session)

        assert "<svg" in html
        assert "Score Distribution" in html
        assert "Compliance Breakdown" in html
        assert "chart-grid" in html

    @pytest.mark.asyncio
    async def test_empty_report_no_charts(self, async_session):
        """Empty reports should not render SVG charts when no data."""
        html = await generate_html_report(async_session)

        assert "<svg" not in html


# ── Test: SVG Chart Helpers ──────────────────────────────────────────


class TestSVGScoreDistribution:

    def test_basic_histogram(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        svg = _svg_score_distribution(scores)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "0-20%" in svg
        assert "80-100%" in svg
        # Each bin has exactly 1 count
        assert ">1<" in svg

    def test_empty_scores(self):
        assert _svg_score_distribution([]) == ""

    def test_all_scores_in_one_bin(self):
        scores = [0.85, 0.9, 0.95, 1.0]
        svg = _svg_score_distribution(scores)
        assert "<svg" in svg
        assert ">4<" in svg  # all 4 in the 80-100% bin

    def test_contains_rect_bars(self):
        scores = [0.5, 0.6]
        svg = _svg_score_distribution(scores)
        assert "<rect" in svg


class TestSVGComplianceDonut:

    def test_basic_donut(self):
        counts = {
            "compliant": 5, "partial": 3, "not_relevant": 1,
            "non_compliant": 2, "risky": 1,
        }
        svg = _svg_compliance_donut(counts)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "<circle" in svg
        assert "12" in svg  # total diagnoses = 12
        assert "diagnoses" in svg

    def test_empty_counts(self):
        assert _svg_compliance_donut({}) == ""
        assert _svg_compliance_donut({"compliant": 0, "risky": 0}) == ""

    def test_single_category(self):
        svg = _svg_compliance_donut({"compliant": 10})
        assert "<svg" in svg
        assert "+2 Compliant: 10" in svg

    def test_legend_labels(self):
        counts = {"compliant": 3, "non_compliant": 2}
        svg = _svg_compliance_donut(counts)
        assert "+2 Compliant: 3" in svg
        assert "-1 Non-compliant: 2" in svg


class TestSVGConditionBars:

    def test_basic_horizontal_bars(self):
        rows = [
            ("Back pain", 10, 8, 2, 0.8),
            ("Knee pain", 5, 1, 4, 0.2),
        ]
        svg = _svg_condition_bars(rows)
        assert "<svg" in svg
        assert "Back pain" in svg
        assert "Knee pain" in svg
        assert "80%" in svg
        assert "20%" in svg

    def test_empty_rows(self):
        assert _svg_condition_bars([]) == ""

    def test_truncates_long_labels(self):
        rows = [("Very long condition name that exceeds limit", 5, 3, 2, 0.6)]
        svg = _svg_condition_bars(rows)
        assert "..." in svg
        assert "Very long condition name that exceeds limit" not in svg

    def test_zero_rate(self):
        rows = [("Shoulder pain", 3, 0, 3, 0.0)]
        svg = _svg_condition_bars(rows)
        assert "0%" in svg


# ── Test: Chart Data Collection ──────────────────────────────────────


class TestCollectChartData:

    @pytest.mark.asyncio
    async def test_collects_scores_and_levels(self, async_session):
        p = _make_patient(async_session, "P-DATA")
        await _add_completed_result(async_session, p, 0.75, [
            {"diagnosis": "Back pain", "score": 2, "judgement": "Compliant",
             "explanation": "OK", "guidelines_followed": [],
             "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "judgement": "Non-compliant",
             "explanation": "Bad", "guidelines_followed": [],
             "guidelines_not_followed": []},
        ])

        scores, level_counts, condition_rows = await _collect_chart_data(
            async_session,
        )

        assert scores == [0.75]
        assert level_counts["compliant"] == 1
        assert level_counts["non_compliant"] == 1
        assert len(condition_rows) == 2

    @pytest.mark.asyncio
    async def test_empty_data(self, async_session):
        scores, level_counts, condition_rows = await _collect_chart_data(
            async_session,
        )

        assert scores == []
        assert all(v == 0 for v in level_counts.values())
        assert condition_rows == []


# ── Test: PNG Chart Export ───────────────────────────────────────────


def _fake_svg2png(**kwargs):
    """Return fake PNG bytes (PNG header + minimal data)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


class TestExportChartsToPNG:

    @pytest.mark.asyncio
    async def test_creates_png_files(self, async_session, tmp_path):
        """Charts are saved as PNG files when data is present."""
        p = _make_patient(async_session, "P-PNG")
        await _add_completed_result(async_session, p, 0.8, [
            {"diagnosis": "Back pain", "score": 2, "judgement": "Compliant",
             "explanation": "OK", "guidelines_followed": ["Physio"],
             "guidelines_not_followed": []},
            {"diagnosis": "Knee pain", "score": -1, "judgement": "Non-compliant",
             "explanation": "Bad", "guidelines_followed": [],
             "guidelines_not_followed": ["Exercise"]},
        ])

        output_dir = str(tmp_path / "charts")
        with patch("src.services.export.cairosvg") as mock_cairo:
            mock_cairo.svg2png = _fake_svg2png
            saved = await export_charts_to_png(
                async_session, output_dir,
            )

        assert len(saved) == 3
        for path in saved:
            assert os.path.exists(path)
            assert path.endswith(".png")
            assert os.path.getsize(path) > 0

        expected_names = {
            "score_distribution.png",
            "compliance_breakdown.png",
            "condition_adherence.png",
        }
        actual_names = {os.path.basename(p) for p in saved}
        assert actual_names == expected_names

    @pytest.mark.asyncio
    async def test_empty_data_no_files(self, async_session, tmp_path):
        """No files created when there's no audit data."""
        output_dir = str(tmp_path / "empty_charts")
        with patch("src.services.export.cairosvg") as mock_cairo:
            mock_cairo.svg2png = _fake_svg2png
            saved = await export_charts_to_png(async_session, output_dir)

        assert saved == []
        assert not os.path.exists(output_dir)

    @pytest.mark.asyncio
    async def test_creates_output_dir(self, async_session, tmp_path):
        """Output directory is created if it doesn't exist."""
        p = _make_patient(async_session, "P-DIR")
        await _add_completed_result(async_session, p, 0.5, [
            {"diagnosis": "Hip pain", "score": 1, "judgement": "Partial",
             "explanation": "OK", "guidelines_followed": [],
             "guidelines_not_followed": []},
        ])

        nested_dir = str(tmp_path / "deep" / "nested" / "charts")
        with patch("src.services.export.cairosvg") as mock_cairo:
            mock_cairo.svg2png = _fake_svg2png
            saved = await export_charts_to_png(async_session, nested_dir)

        assert len(saved) > 0
        assert os.path.isdir(nested_dir)

    @pytest.mark.asyncio
    async def test_job_id_filter(self, async_session, tmp_path):
        """Charts only include data from the specified job."""
        job = await _make_job(async_session)
        p1 = _make_patient(async_session, "P-JOB")
        p2 = _make_patient(async_session, "P-OTHER")
        await _add_completed_result(async_session, p1, 0.9, [
            {"diagnosis": "Back pain", "score": 2, "judgement": "Compliant",
             "explanation": "OK", "guidelines_followed": [],
             "guidelines_not_followed": []},
        ], job_id=job.id)
        await _add_completed_result(async_session, p2, 0.1, [
            {"diagnosis": "Knee pain", "score": -2, "judgement": "Risky",
             "explanation": "Bad", "guidelines_followed": [],
             "guidelines_not_followed": []},
        ])

        output_dir = str(tmp_path / "job_charts")
        with patch("src.services.export.cairosvg") as mock_cairo:
            mock_cairo.svg2png = _fake_svg2png
            saved = await export_charts_to_png(
                async_session, output_dir, job_id=job.id,
            )

        assert len(saved) > 0


# ── Test: Comparison Chart SVGs ──────────────────────────────────────


class TestComparisonCharts:

    def test_confusion_matrix_svg(self):
        """Confusion matrix should generate valid SVG with labels."""
        matrix = [
            [5, 0, 0, 0, 0],
            [0, 3, 1, 0, 0],
            [0, 0, 8, 0, 0],
            [0, 0, 1, 6, 0],
            [0, 0, 0, 0, 2],
        ]
        labels = ["-2", "-1", "0", "+1", "+2"]
        svg = _svg_confusion_matrix(matrix, labels)
        assert "<svg" in svg
        assert "Model A" in svg
        assert "Model B" in svg
        for label in labels:
            assert label in svg

    def test_confusion_matrix_empty(self):
        """Empty labels should return empty string."""
        assert _svg_confusion_matrix([], []) == ""

    def test_comparison_scores_svg(self):
        """Score comparison should generate valid SVG with both models."""
        scores_a = {"+2": 10, "+1": 5, "0": 3, "-1": 2, "-2": 0}
        scores_b = {"+2": 8, "+1": 7, "0": 2, "-1": 3, "-2": 1}
        svg = _svg_comparison_scores(scores_a, scores_b, "OpenAI", "Ollama")
        assert "<svg" in svg
        assert "OpenAI" in svg
        assert "Ollama" in svg

    def test_comparison_compliance_svg(self):
        """Side-by-side donut chart should generate valid SVG."""
        levels_a = {"compliant": 10, "partial": 5, "not_relevant": 3,
                    "non_compliant": 2, "risky": 1}
        levels_b = {"compliant": 8, "partial": 7, "not_relevant": 2,
                    "non_compliant": 3, "risky": 0}
        svg = _svg_comparison_compliance(levels_a, levels_b, "OpenAI", "Ollama")
        assert "<svg" in svg
        assert "OpenAI" in svg
        assert "Ollama" in svg
