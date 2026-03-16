"""
Model comparison service — compare audit results across providers.

Reads results from two different job_ids (one per provider/model run)
and produces side-by-side comparison metrics including Cohen's kappa,
Pearson correlation, and per-condition adherence deltas.
"""

import json
import logging
import math
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config.settings import get_settings
from src.models.audit import AuditJob, AuditResult
from src.models.patient import Patient

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class DiagnosisComparison:
    """Side-by-side comparison for a single diagnosis."""

    diagnosis: str
    index_date: str | None
    score_a: int | None
    score_b: int | None
    judgement_a: str | None
    judgement_b: str | None
    agreement: bool  # same sign direction


@dataclass
class PatientComparison:
    """Side-by-side comparison for a single patient."""

    pat_id: str
    score_a: float | None  # overall_score from job A
    score_b: float | None  # overall_score from job B
    score_diff: float | None  # score_a - score_b
    diagnoses_a: int
    diagnoses_b: int
    per_diagnosis: list[DiagnosisComparison] = field(default_factory=list)
    agreement: bool = True  # all diagnoses agree in direction


@dataclass
class ConditionComparison:
    """Per-condition adherence comparison across two jobs."""

    condition: str
    count: int
    adherence_rate_a: float
    adherence_rate_b: float
    diff: float  # adherence_rate_a - adherence_rate_b


@dataclass
class ComparisonResult:
    """Full comparison between two batch runs."""

    job_a_id: int | None
    job_b_id: int | None
    job_a_model: str | None
    job_b_model: str | None
    total_patients_compared: int
    patients: list[PatientComparison] = field(default_factory=list)
    per_condition: list[ConditionComparison] = field(default_factory=list)
    mean_score_a: float = 0.0
    mean_score_b: float = 0.0
    mean_abs_diff: float = 0.0
    score_correlation: float = 0.0
    agreement_rate: float = 0.0  # fraction of diagnoses with same direction
    cohen_kappa: float = 0.0

    def summary(self) -> dict:
        """JSON-serializable summary."""
        return {
            "job_a_id": self.job_a_id,
            "job_b_id": self.job_b_id,
            "job_a_model": self.job_a_model,
            "job_b_model": self.job_b_model,
            "total_patients_compared": self.total_patients_compared,
            "mean_score_a": round(self.mean_score_a, 4),
            "mean_score_b": round(self.mean_score_b, 4),
            "mean_abs_diff": round(self.mean_abs_diff, 4),
            "score_correlation": round(self.score_correlation, 4),
            "agreement_rate": round(self.agreement_rate, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "patients": [
                {
                    "pat_id": p.pat_id,
                    "score_a": round(p.score_a, 4) if p.score_a is not None else None,
                    "score_b": round(p.score_b, 4) if p.score_b is not None else None,
                    "score_diff": round(p.score_diff, 4) if p.score_diff is not None else None,
                    "diagnoses_a": p.diagnoses_a,
                    "diagnoses_b": p.diagnoses_b,
                    "agreement": p.agreement,
                    "per_diagnosis": [
                        {
                            "diagnosis": d.diagnosis,
                            "index_date": d.index_date,
                            "score_a": d.score_a,
                            "score_b": d.score_b,
                            "judgement_a": d.judgement_a,
                            "judgement_b": d.judgement_b,
                            "agreement": d.agreement,
                        }
                        for d in p.per_diagnosis
                    ],
                }
                for p in self.patients
            ],
            "per_condition": [
                {
                    "condition": c.condition,
                    "count": c.count,
                    "adherence_rate_a": round(c.adherence_rate_a, 4),
                    "adherence_rate_b": round(c.adherence_rate_b, 4),
                    "diff": round(c.diff, 4),
                }
                for c in self.per_condition
            ],
        }


# ── Statistical helpers ───────────────────────────────────────────────


def _classify_direction(score: int) -> int:
    """Bin a 5-level score into 3 classes for kappa: 0=non-adherent, 1=neutral, 2=adherent."""
    if score >= 1:
        return 2  # adherent
    elif score == 0:
        return 1  # neutral
    else:
        return 0  # non-adherent


def compute_cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """
    Compute Cohen's kappa for inter-rater agreement.

    Labels should already be binned into classes (e.g., 0, 1, 2).
    Returns kappa in [-1, 1], where 1 = perfect agreement.
    """
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0

    n = len(labels_a)
    classes = sorted(set(labels_a) | set(labels_b))
    k = len(classes)
    if k == 0:
        return 0.0

    class_idx = {c: i for i, c in enumerate(classes)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[class_idx[a]][class_idx[b]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement (by chance)
    p_e = 0.0
    for i in range(k):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(k))
        p_e += (row_sum * col_sum) / (n * n)

    if p_e == 1.0:
        return 1.0  # trivial case: all in one class

    return (p_o - p_e) / (1.0 - p_e)


def compute_pearson(values_a: list[float], values_b: list[float]) -> float:
    """
    Compute Pearson correlation coefficient.

    Returns r in [-1, 1]. Returns 0.0 for degenerate cases
    (< 2 data points, zero variance).
    """
    n = len(values_a)
    if n < 2 or len(values_b) != n:
        return 0.0

    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    var_a = sum((a - mean_a) ** 2 for a in values_a)
    var_b = sum((b - mean_b) ** 2 for b in values_b)

    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        return 0.0

    return cov / denom


# ── Main comparison function ──────────────────────────────────────────


async def compare_jobs(
    session: AsyncSession,
    job_a_id: int | None = None,
    job_b_id: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
) -> ComparisonResult:
    """
    Compare audit results from two batch jobs or two models side-by-side.

    Accepts either job IDs or model names (or a mix). When using model
    names, aggregates results across all jobs for that model.

    Returns a ComparisonResult with per-patient diffs, per-condition
    adherence deltas, and statistical agreement metrics.
    """
    results_a, model_a_name, resolved_a_id = await _resolve_results(
        session, job_a_id, model_a, "Model A",
    )
    results_b, model_b_name, resolved_b_id = await _resolve_results(
        session, job_b_id, model_b, "Model B",
    )

    # Index by pat_id
    map_a: dict[str, AuditResult] = {r.patient.pat_id: r for r in results_a if r.patient}
    map_b: dict[str, AuditResult] = {r.patient.pat_id: r for r in results_b if r.patient}

    # Only compare patients present in both
    common_pat_ids = sorted(set(map_a.keys()) & set(map_b.keys()))

    patients: list[PatientComparison] = []
    all_directions_a: list[int] = []
    all_directions_b: list[int] = []
    scores_a_list: list[float] = []
    scores_b_list: list[float] = []
    condition_scores: dict[str, dict] = {}  # {term: {adherent_a, total_a, ...}}
    agreement_count = 0
    total_diagnoses_compared = 0

    for pat_id in common_pat_ids:
        ra = map_a[pat_id]
        rb = map_b[pat_id]

        score_a = ra.overall_score
        score_b = rb.overall_score
        score_diff = None
        if score_a is not None and score_b is not None:
            score_diff = score_a - score_b
            scores_a_list.append(score_a)
            scores_b_list.append(score_b)

        # Parse per-diagnosis details
        details_a = _parse_details(ra.details_json)
        details_b = _parse_details(rb.details_json)

        # Match diagnoses by (diagnosis, index_date)
        diag_map_a = {(d["diagnosis"], d.get("index_date")): d for d in details_a}
        diag_map_b = {(d["diagnosis"], d.get("index_date")): d for d in details_b}
        common_diags = sorted(set(diag_map_a.keys()) & set(diag_map_b.keys()))

        per_diag: list[DiagnosisComparison] = []
        patient_agrees = True
        for diag_key in common_diags:
            da = diag_map_a[diag_key]
            db = diag_map_b[diag_key]
            sa = da.get("score")
            sb = db.get("score")

            dir_a = _classify_direction(sa) if sa is not None else None
            dir_b = _classify_direction(sb) if sb is not None else None
            agrees = dir_a == dir_b if dir_a is not None and dir_b is not None else False

            if dir_a is not None and dir_b is not None:
                all_directions_a.append(dir_a)
                all_directions_b.append(dir_b)
                total_diagnoses_compared += 1
                if agrees:
                    agreement_count += 1
                else:
                    patient_agrees = False

            # Per-condition tracking
            term = diag_key[0]
            if term not in condition_scores:
                condition_scores[term] = {
                    "adherent_a": 0, "total_a": 0,
                    "adherent_b": 0, "total_b": 0,
                }
            if sa is not None:
                condition_scores[term]["total_a"] += 1
                if sa >= 1:
                    condition_scores[term]["adherent_a"] += 1
            if sb is not None:
                condition_scores[term]["total_b"] += 1
                if sb >= 1:
                    condition_scores[term]["adherent_b"] += 1

            per_diag.append(DiagnosisComparison(
                diagnosis=diag_key[0],
                index_date=diag_key[1],
                score_a=sa,
                score_b=sb,
                judgement_a=da.get("judgement"),
                judgement_b=db.get("judgement"),
                agreement=agrees,
            ))

        patients.append(PatientComparison(
            pat_id=pat_id,
            score_a=score_a,
            score_b=score_b,
            score_diff=score_diff,
            diagnoses_a=len(details_a),
            diagnoses_b=len(details_b),
            per_diagnosis=per_diag,
            agreement=patient_agrees,
        ))

    # Aggregate metrics
    mean_a = sum(scores_a_list) / len(scores_a_list) if scores_a_list else 0.0
    mean_b = sum(scores_b_list) / len(scores_b_list) if scores_b_list else 0.0
    mean_abs = (
        sum(abs(a - b) for a, b in zip(scores_a_list, scores_b_list)) / len(scores_a_list)
        if scores_a_list else 0.0
    )
    correlation = compute_pearson(scores_a_list, scores_b_list)
    agree_rate = agreement_count / total_diagnoses_compared if total_diagnoses_compared > 0 else 0.0
    kappa = compute_cohen_kappa(all_directions_a, all_directions_b)

    # Per-condition comparison
    per_condition: list[ConditionComparison] = []
    for term, cs in sorted(condition_scores.items(), key=lambda x: x[1]["total_a"], reverse=True):
        count = max(cs["total_a"], cs["total_b"])
        rate_a = cs["adherent_a"] / cs["total_a"] if cs["total_a"] > 0 else 0.0
        rate_b = cs["adherent_b"] / cs["total_b"] if cs["total_b"] > 0 else 0.0
        per_condition.append(ConditionComparison(
            condition=term,
            count=count,
            adherence_rate_a=rate_a,
            adherence_rate_b=rate_b,
            diff=rate_a - rate_b,
        ))

    return ComparisonResult(
        job_a_id=resolved_a_id or 0,
        job_b_id=resolved_b_id or 0,
        job_a_model=model_a_name,
        job_b_model=model_b_name,
        total_patients_compared=len(common_pat_ids),
        patients=patients,
        per_condition=per_condition,
        mean_score_a=mean_a,
        mean_score_b=mean_b,
        mean_abs_diff=mean_abs,
        score_correlation=correlation,
        agreement_rate=agree_rate,
        cohen_kappa=kappa,
    )


# ── Private helpers ───────────────────────────────────────────────────


async def _load_job_results(
    session: AsyncSession,
    job_id: int,
) -> list[AuditResult]:
    """Load all completed results for a job with patient relationship."""
    query = (
        select(AuditResult)
        .where(AuditResult.job_id == job_id)
        .where(AuditResult.status == "completed")
        .options(selectinload(AuditResult.patient))
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def _load_model_results(
    session: AsyncSession,
    model: str,
) -> list[AuditResult]:
    """Load all completed results for a model across all jobs.

    When a patient has results in multiple jobs for the same model,
    the latest result (highest audit_result.id) is used.
    """
    job_ids_q = select(AuditJob.id).where(AuditJob.provider == model)
    query = (
        select(AuditResult)
        .where(AuditResult.job_id.in_(job_ids_q))
        .where(AuditResult.status == "completed")
        .options(selectinload(AuditResult.patient))
        .order_by(AuditResult.id.desc())
    )
    result = await session.execute(query)
    all_results = list(result.scalars().all())

    # Deduplicate: keep the latest result per patient
    seen: set[int] = set()
    deduped: list[AuditResult] = []
    for r in all_results:
        if r.patient_id not in seen:
            seen.add(r.patient_id)
            deduped.append(r)
    return deduped


async def _resolve_results(
    session: AsyncSession,
    job_id: int | None,
    model: str | None,
    label: str,
) -> tuple[list[AuditResult], str | None, int | None]:
    """Resolve results from either a job_id or model name.

    Returns (results, model_name, job_id_or_none).
    """
    settings = get_settings()
    if job_id is not None:
        job = await session.get(AuditJob, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        results = await _load_job_results(session, job_id)
        model_name = settings.model_name_for_provider(
            getattr(job, "provider", None),
        )
        return results, model_name, job_id
    elif model is not None:
        results = await _load_model_results(session, model)
        if not results:
            raise ValueError(f"No completed results for model '{model}'")
        return results, model, None
    else:
        raise ValueError(f"{label} requires either job_id or model")


def _parse_details(details_json: str | None) -> list[dict]:
    """Parse the per-diagnosis scores from details_json."""
    if not details_json:
        return []
    try:
        details = json.loads(details_json)
        return details.get("scores", [])
    except (json.JSONDecodeError, AttributeError):
        return []


# ── Cross-model classification metrics ───────────────────────────────


def _compute_auroc(labels: list[int], scores: list[float]) -> float | None:
    """
    Compute AUROC via trapezoidal rule (no sklearn needed).

    Args:
        labels: Binary labels (1 = positive, 0 = negative).
        scores: Continuous prediction scores (higher = more positive).

    Returns:
        AUROC in [0, 1], or None if insufficient data.
    """
    if not labels or len(labels) != len(scores):
        return None

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0

    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return round(auc, 4)


async def compute_cross_model_classification(
    session: AsyncSession,
    job_a_id: int | None = None,
    job_b_id: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
) -> dict:
    """
    Enhanced cross-model comparison with confusion matrix and classification metrics.

    Accepts either job IDs or model names (or a mix). When using model
    names, aggregates results across all jobs for that model.

    Computes:
    - 5×5 confusion matrix (Model A vs Model B scores)
    - Per-class precision/recall/F1
    - 5-class and 3-class Cohen's kappa
    - Exact-match accuracy
    - AUROC (adherent vs non-adherent, using confidence as score)
    """
    results_a, model_a_name, resolved_a_id = await _resolve_results(
        session, job_a_id, model_a, "Model A",
    )
    results_b, model_b_name, resolved_b_id = await _resolve_results(
        session, job_b_id, model_b, "Model B",
    )

    map_a = {r.patient.pat_id: r for r in results_a if r.patient}
    map_b = {r.patient.pat_id: r for r in results_b if r.patient}
    common_pat_ids = sorted(set(map_a.keys()) & set(map_b.keys()))

    # Collect paired scores at diagnosis level
    CLASS_LABELS = [-2, -1, 0, 1, 2]
    LABEL_STRS = ["-2", "-1", "0", "+1", "+2"]
    matrix = [[0] * 5 for _ in range(5)]  # matrix[row_a][col_b]

    scores_a_raw: list[int] = []
    scores_b_raw: list[int] = []
    directions_a: list[int] = []
    directions_b: list[int] = []
    auroc_labels: list[int] = []
    auroc_scores: list[float] = []
    exact_matches = 0
    total_compared = 0

    for pat_id in common_pat_ids:
        details_a = _parse_details(map_a[pat_id].details_json)
        details_b = _parse_details(map_b[pat_id].details_json)

        diag_map_a = {(d["diagnosis"], d.get("index_date")): d for d in details_a}
        diag_map_b = {(d["diagnosis"], d.get("index_date")): d for d in details_b}
        common_diags = set(diag_map_a.keys()) & set(diag_map_b.keys())

        for diag_key in common_diags:
            da = diag_map_a[diag_key]
            db = diag_map_b[diag_key]
            sa = da.get("score")
            sb = db.get("score")

            if sa is None or sb is None:
                continue
            if sa not in CLASS_LABELS or sb not in CLASS_LABELS:
                continue

            total_compared += 1
            scores_a_raw.append(sa)
            scores_b_raw.append(sb)

            # Confusion matrix
            row = sa + 2  # offset: -2→0, -1→1, 0→2, 1→3, 2→4
            col = sb + 2
            matrix[row][col] += 1

            if sa == sb:
                exact_matches += 1

            # 3-class directions for kappa comparison
            directions_a.append(_classify_direction(sa))
            directions_b.append(_classify_direction(sb))

            # AUROC: binarize adherent (>=1) vs non-adherent (<=-1), skip 0
            if sa >= 1 or sa <= -1:
                binary_label = 1 if sa >= 1 else 0
                conf_a = da.get("confidence", 0.5)
                # For non-adherent, use 1-confidence as score
                auroc_score = conf_a if sa >= 1 else (1.0 - conf_a)
                auroc_labels.append(binary_label)
                auroc_scores.append(auroc_score)

    # Per-class P/R/F1
    per_class_metrics = {}
    for i, label in enumerate(LABEL_STRS):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(5)) - tp  # column sum minus diagonal
        fn = sum(matrix[i][c] for c in range(5)) - tp  # row sum minus diagonal
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        support = sum(matrix[i])  # row sum = how many Model A gave this class
        per_class_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Kappas
    kappa_5class = compute_cohen_kappa(scores_a_raw, scores_b_raw)
    kappa_3class = compute_cohen_kappa(directions_a, directions_b)

    # Accuracy
    accuracy = exact_matches / total_compared if total_compared > 0 else 0.0

    # Agreement rate (same direction)
    agree_count = sum(1 for a, b in zip(directions_a, directions_b) if a == b)
    agreement_rate = agree_count / total_compared if total_compared > 0 else 0.0

    # Pearson correlation
    correlation = compute_pearson(
        [float(s) for s in scores_a_raw],
        [float(s) for s in scores_b_raw],
    )

    # AUROC
    auroc = _compute_auroc(auroc_labels, auroc_scores)

    return {
        "job_a_id": resolved_a_id or 0,
        "job_b_id": resolved_b_id or 0,
        "job_a_model": model_a_name,
        "job_b_model": model_b_name,
        "total_diagnoses_compared": total_compared,
        "confusion_matrix": {
            "labels": LABEL_STRS,
            "matrix": matrix,
        },
        "per_class_metrics": per_class_metrics,
        "cohen_kappa_5class": round(kappa_5class, 4),
        "cohen_kappa_3class": round(kappa_3class, 4),
        "exact_match_accuracy": round(accuracy, 4),
        "auroc": auroc,
        "agreement_rate": round(agreement_rate, 4),
        "pearson_correlation": round(correlation, 4),
    }
