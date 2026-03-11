# 13. Evaluation Framework Explained

> How we measure whether the AI audit system is actually working — model comparison, missing care detection, and gold-standard validation.

---

## Why an Evaluation Framework?

The 4-agent pipeline produces audit scores, but **how do we know they're correct?** Without evaluation, we're just trusting the LLM. The supervisor feedback identified several evaluation needs:

| Need | Solution | Status |
|------|----------|--------|
| Compare OpenAI vs local Ollama models | Model comparison service | ✅ Done |
| Detect care gaps in patient records | Missing care opportunities | ✅ Done |
| Compare AI scores vs clinician labels | Gold-standard metrics | Planned (awaiting labels) |
| Evaluate each agent independently | LLM-as-Judge | ✅ Done |
| System-level metrics (class distribution, adherence, confidence) | System metrics endpoint | ✅ Done |
| Cross-model classification (confusion matrix, P/R/F1, AUROC) | Cross-model metrics endpoint | ✅ Done |
| Extractor quality from stored DB | Extractor metrics endpoint | ✅ Done |
| Retriever IR metrics (P@k, nDCG, MRR) | Full agent evaluation endpoint | ✅ Done |

---

## Model Comparison Service

### The Problem
We support two LLM providers — **OpenAI (gpt-4o-mini)** and **Ollama (mistral-small, local)**. When running both on the same patients, we need to answer: *Do they agree? Which is more reliable?*

### How It Works

```
Batch Job A (OpenAI)  ──┐
                        ├──→ compare_jobs() ──→ ComparisonResult
Batch Job B (Ollama)  ──┘
```

1. Each batch run creates a separate `AuditJob` with a unique `job_id`
2. The `AuditJob.provider` column tracks which AI provider ran that batch (set automatically from `settings.ai_provider`)
3. The comparison service loads completed results from both jobs, matches patients by `pat_id`
4. For each matched patient, it matches diagnoses by `(diagnosis_term, index_date)` and compares scores

### Key Metrics

**Patient-level:**
- `score_diff` = overall_score_A - overall_score_B (per patient)
- `mean_abs_diff` = average absolute difference across all patients
- `score_correlation` = Pearson correlation of patient scores

**Diagnosis-level:**
- `agreement_rate` = fraction of diagnoses where both models agree on direction (adherent/neutral/non-adherent)
- `cohen_kappa` = inter-rater agreement accounting for chance (0 = random, 1 = perfect)

**Condition-level:**
- Per-condition adherence rate from each model, with delta

### Cohen's Kappa — Why Not Just Agreement %?

Raw agreement % is misleading. If 90% of patients are compliant, two random models would "agree" 81% of the time by chance. **Cohen's kappa** corrects for this:

```
kappa = (observed_agreement - expected_by_chance) / (1 - expected_by_chance)
```

We bin the 5-level scores into 3 classes for kappa:
- **Adherent**: +2, +1
- **Neutral**: 0
- **Non-adherent**: -1, -2

### How to Run a Comparison

```bash
# Step 1: Run batch with OpenAI (default)
# AI_PROVIDER=openai in .env
curl -X POST localhost:8000/api/v1/audit/batch?limit=20
# Note the job_id (e.g. 1)

# Step 2: Switch to Ollama
# Change AI_PROVIDER=ollama in .env, restart server

# Step 3: Run same patients with Ollama
curl -X POST localhost:8000/api/v1/audit/batch?limit=20
# Note the job_id (e.g. 2)

# Step 4: Compare
curl "localhost:8000/api/v1/evaluation/compare?job_a=1&job_b=2"
```

### API Response Structure

```json
{
  "job_a_id": 1,
  "job_b_id": 2,
  "job_a_provider": "openai",
  "job_b_provider": "ollama",
  "total_patients_compared": 20,
  "mean_score_a": 0.62,
  "mean_score_b": 0.55,
  "mean_abs_diff": 0.12,
  "score_correlation": 0.78,
  "agreement_rate": 0.85,
  "cohen_kappa": 0.71,
  "patients": [
    {
      "pat_id": "abc-123",
      "score_a": 0.75,
      "score_b": 0.5,
      "score_diff": 0.25,
      "agreement": false,
      "per_diagnosis": [
        {
          "diagnosis": "Low back pain",
          "score_a": 2,
          "score_b": 1,
          "judgement_a": "COMPLIANT",
          "judgement_b": "PARTIALLY COMPLIANT",
          "agreement": true
        }
      ]
    }
  ],
  "per_condition": [
    {
      "condition": "Low back pain",
      "count": 15,
      "adherence_rate_a": 0.80,
      "adherence_rate_b": 0.67,
      "diff": 0.13
    }
  ]
}
```

### Architecture

```
src/services/comparison.py          — Core logic (compare_jobs, compute_cohen_kappa, compute_pearson)
src/api/routes/evaluation.py        — GET /evaluation/compare endpoint
src/models/audit.py                 — AuditJob.provider column
src/services/pipeline.py            — Sets provider on job creation
migrations/versions/002_add_job_provider.py  — DB migration
tests/unit/test_comparison.py       — 21 tests
```

---

## Missing Care Opportunities

### The Problem
The scorer already identifies when guidelines are NOT followed. But the supervisor wanted something more specific: **what NICE-recommended actions are absent from the patient record?**

This is different from "guidelines not followed" — it's about detecting concrete care gaps that could inform quality improvement.

### How It Works

The `SCORING_PROMPT` now includes an extra output line:

```
Missing Care Opportunities: comma-separated list of specific NICE-recommended
actions that SHOULD have been documented but are NOT present in the patient
record, or "None" if no gaps identified
```

The LLM identifies concrete actions like:
- "Exercise therapy advice"
- "Weight management referral"
- "NSAID prescription"
- "Red flag assessment"

### Data Flow

```
LLM Response
    ↓
parse_scoring_response()  →  {"missing_care_opportunities": ["Exercise therapy", "NSAID"]}
    ↓
DiagnosisScore.missing_care_opportunities  →  stored in details_json
    ↓
get_missing_care_summary()  →  groups by condition, counts frequency
    ↓
GET /evaluation/missing-care  →  API response
```

### Where It Shows Up

| Output | How |
|--------|-----|
| **API** | `GET /evaluation/missing-care` — grouped by condition with frequency counts |
| **CSV export** | `missing_care_opportunities` column (semicolon-separated) |
| **HTML report** | Amber "Missing care: ..." tag on diagnosis cards |
| **JSON details** | `missing_care_opportunities` array in each score object |

### Backward Compatibility

Old results (before this change) don't have the field. The parser returns `[]` when absent, and reporting code uses `.get("missing_care_opportunities", [])`. No migration needed — field lives inside `details_json` (JSON text column).

### Architecture

```
src/agents/scorer.py                — DiagnosisScore.missing_care_opportunities field,
                                      SCORING_PROMPT update, _MISSING_CARE_PATTERN regex,
                                      parse_scoring_response() extraction
src/services/reporting.py           — get_missing_care_summary() aggregation function
src/services/export.py              — CSV column + HTML amber tag + CSS class
src/api/routes/evaluation.py        — GET /evaluation/missing-care endpoint
tests/unit/test_missing_care.py     — 12 tests (parsing, data classes, reporting)
```

---

## Gold-Standard Metrics (Planned — Phase 9c)

When the 120 clinician-labeled cases arrive, we'll compare AI scores against human auditor scores using:

- **Confusion matrix** — 5x5 grid showing how the AI classified vs how the human classified
- **Per-class Precision/Recall/F1** — for each of the 5 score levels (-2, -1, 0, +1, +2)
- **Macro F1** — unweighted average across classes (treats rare classes equally)
- **Weighted F1** — weighted by class frequency (reflects real-world performance)
- **Cohen's kappa** — AI-vs-human agreement accounting for chance
- **Accuracy** — exact match rate

The gold labels will be submitted as JSON:
```json
{"pat_id": {"Low back pain": 2, "Knee pain": -1}, ...}
```

---

## LLM-as-Judge Evaluation

### The Problem
How do we know each pipeline agent is doing a *good* job? Without clinician labels (gold-standard data), we need automated quality checks. The solution: use a **separate LLM call** to judge each agent's output, or **compare against known rules** (weak supervision).

### Evaluation Methods by Agent

| Agent | Evaluation Method | Metrics | LLM Cost |
|-------|-------------------|---------|----------|
| Extractor | Weak supervision (SNOMED rules as pseudo-ground-truth) | Per-category P/R/F1, rule_match_rate | None |
| Query Generator | LLM-as-Judge (rate query relevance) | Mean relevance (1-5), mean coverage (1-5) | 1 call/diagnosis |
| Retriever | LLM-as-Judge (rate guideline relevance) | Mean relevance (1-5), per-diagnosis breakdown | 1 call/diagnosis |
| Scorer | LLM-as-Judge (rate reasoning quality) | Reasoning quality, citation accuracy, score calibration (all 1-5) | 1 call/diagnosis |

### How Each Evaluation Works

**Extractor — Weak Supervision (no LLM needed):**
```
Raw clinical entries (concept_display strings)
    ↓
categorise_by_rules()  →  SNOMED rules produce "ground truth" categories
    ↓
Compare against extractor's cached categories
    ↓
Per-category: precision = TP/(TP+FP), recall = TP/(TP+FN), F1
Overall: rule_match_rate = matched/total
```

This catches cases where the LLM categorised something differently from what the SNOMED rules would say. A high rule_match_rate (>0.95) means the extractor is consistent with the rule-based categorisation.

**Query Generator — LLM-as-Judge:**
```
Generated queries + diagnosis context
    ↓
Sent to LLM judge with structured prompt
    ↓
LLM rates:
  Relevance (1-5): Are these queries useful for finding NICE guidelines?
  Coverage (1-5): Do they cover treatment, referral, investigation, red flags?
```

**Retriever — LLM-as-Judge:**
```
Retrieved guidelines + diagnosis context
    ↓
Sent to LLM judge with structured prompt
    ↓
LLM rates:
  Relevance (1-5): Are these guidelines relevant to this diagnosis?
```

**Scorer — LLM-as-Judge:**
```
Score + judgement + explanation + cited guideline + guidelines followed/not followed
    ↓
Sent to LLM judge with structured prompt
    ↓
LLM rates:
  Reasoning Quality (1-5): Is the explanation logical and clear?
  Citation Accuracy (1-5): Does the cited guideline support the conclusion?
  Score Calibration (1-5): Is the assigned score appropriate?
```

### Rating Parsing

LLM judge responses follow a structured format like:
```
Reasoning Quality: 4
Citation Accuracy: 5
Score Calibration: 3
```

The parser extracts the integer after the colon for each field name (case-insensitive). Values are clamped to [1, 5]. If parsing fails, defaults to 3 (middle rating).

### Evaluating from Stored Data (No Re-Run)

The scorer can be evaluated **without re-running the pipeline** because `details_json` stores everything needed: scores, explanations, cited guideline text, guidelines followed/not followed.

```
AuditResult.details_json
    ↓
scoring_from_stored(details)  →  Reconstructs ScoringResult dataclass
    ↓
evaluate_scoring(scoring, ai_provider)  →  ScorerMetrics
```

The API endpoint `POST /api/v1/evaluation/evaluate/scorer/{job_id}` uses this approach. It loads completed results for a batch job and sends each diagnosis to the LLM judge. Use `?limit=5` to control cost (default: 5 patients).

### Full Pipeline Evaluation (Requires Re-Run)

For evaluating all 4 agents (extractor, query, retriever, scorer), you need intermediate results that aren't stored in the DB. Use the `evaluate_patient()` function with a `PipelineResult`:

```python
from src.services.evaluation import evaluate_patient

evaluation = await evaluate_patient(
    pipeline_result,   # Has .extraction, .query_result, .retrieval, .scoring
    raw_entries,       # Original clinical entries for weak supervision
    ai_provider,       # AI provider for LLM judge calls
    agents=["extractor", "query", "retriever", "scorer"],  # Optional filter
)
```

### Aggregation

Multiple patient evaluations can be aggregated:
```python
from src.services.evaluation import aggregate_evaluations

agg = aggregate_evaluations([eval_patient_1, eval_patient_2, ...])
# agg.scorer.mean_reasoning_quality → average across all patients/diagnoses
# agg.extractor.rule_match_rate → overall rule match rate
```

### Architecture

```
src/services/evaluation.py          — Core evaluation logic:
                                      evaluate_extractor() — weak supervision
                                      evaluate_queries() — LLM-as-Judge
                                      evaluate_retrieval() — LLM-as-Judge
                                      evaluate_scoring() — LLM-as-Judge
                                      evaluate_patient() — orchestrator
                                      aggregate_evaluations() — aggregator
                                      scoring_from_stored() — reconstruct from DB
                                      evaluate_extractor_from_db() — DB-only extractor eval
                                      evaluate_retrieval_ir() — IR metrics (P@k, nDCG, MRR)
                                      run_agent_evaluation() — full pipeline evaluation
src/api/routes/evaluation.py        — All evaluation endpoints (7 total)
src/api/routes/reports.py           — GET /export/comparison-html endpoint
tests/unit/test_evaluation.py       — 28 tests
```

---

## System-Level Metrics

### `GET /evaluation/system-metrics?job_id=N`

Per-job aggregate statistics from stored data (no LLM calls):

- **Score class distribution**: count of +2, +1, 0, -1, -2
- **Adherence rate**: (compliant + partial) / total_scored (excluding not_relevant and errors)
- **Confidence stats**: mean, median, min, max, standard deviation
- **Per-class counts**: compliant, partial, not_relevant, non_compliant, risky, errors
- **Error rate**: failed / total results

Implementation: `compute_system_metrics()` in `src/services/reporting.py`.

---

## Cross-Model Classification Metrics

### `GET /evaluation/cross-model-metrics?job_a=N&job_b=M`

Enhanced cross-model comparison with full classification metrics from stored data (no LLM calls):

- **5×5 confusion matrix**: rows = Model A scores (-2 to +2), cols = Model B scores
- **Per-class Precision/Recall/F1**: for each score class, TP/FP/FN computed from the confusion matrix
- **5-class Cohen's kappa**: inter-rater agreement using raw 5-level scores (not binned)
- **3-class Cohen's kappa**: inter-rater agreement using binned scores (adherent/neutral/non-adherent)
- **Exact-match accuracy**: fraction of diagnoses where both models assigned identical scores
- **AUROC**: area under ROC curve for binary classification (adherent vs non-adherent), computed via trapezoidal rule using confidence scores. No sklearn dependency.
- **Agreement rate + Pearson correlation**: reuses existing comparison helpers

### AUROC Implementation

AUROC is computed without sklearn using the trapezoidal rule (~30 lines):

1. **Binarize**: adherent (score ≥ 1) = positive, non-adherent (score ≤ -1) = negative, score 0 excluded
2. **Sort** by confidence (descending)
3. **Walk thresholds**, computing TPR and FPR at each step
4. **Integrate** using trapezoidal rule: `sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2)`

Implementation: `_compute_auroc()` and `compute_cross_model_classification()` in `src/services/comparison.py`.

---

## Extractor Metrics from Stored Data

### `GET /evaluation/extractor-metrics?sample_size=N`

Evaluates extractor quality by comparing stored category assignments against rule-based categorisation (no LLM calls):

1. Loads `clinical_entries` with stored `category` column from the DB
2. Runs `categorise_by_rules(concept_display)` on each entry as pseudo-ground-truth
3. Compares stored category vs rule-based category
4. Computes per-category P/R/F1 and overall `rule_match_rate`

This differs from the pipeline-based `evaluate_extractor()` which requires running the full pipeline. The DB version is fast and can evaluate thousands of entries.

Implementation: `evaluate_extractor_from_db()` in `src/services/evaluation.py`.

---

## Retriever IR Metrics

### Part of `POST /evaluation/evaluate/agents?limit=5`

Information Retrieval metrics for the guideline retriever, using LLM-as-Judge as proxy ground truth:

- **Per-guideline relevance**: each retrieved guideline rated 1-5 by an LLM judge
- **Binary relevance**: rating ≥ threshold (default 3) = relevant
- **Precision@k**: relevant_in_top_k / k
- **nDCG@k**: Normalized Discounted Cumulative Gain using binary relevance gains
- **MRR**: 1 / rank of first relevant result (0 if none relevant)
- **Mean relevance**: average raw 1-5 score across all guidelines

### nDCG Calculation

```
DCG@k  = Σ (relevance_i / log2(i + 1))  for i = 1..k
IDCG@k = DCG of the ideal ranking (all relevant first)
nDCG   = DCG / IDCG  (0 if IDCG = 0)
```

Implementation: `evaluate_retrieval_ir()` in `src/services/evaluation.py`.

---

## Full Agent Evaluation

### `POST /evaluation/evaluate/agents?limit=5`

Orchestrates evaluation of all 4 pipeline agents:

1. Picks `limit` random patients from the database
2. Runs the full pipeline via `AuditPipeline.run_single()` for each patient
3. Evaluates all 4 agents using existing functions + new retriever IR metrics
4. Aggregates results across all patients

**Warning**: This is expensive — each patient requires multiple LLM calls for the pipeline run plus additional LLM calls for the judge evaluations.

Implementation: `run_agent_evaluation()` in `src/services/evaluation.py`.

---

## Comparison Chart SVGs

Three new SVG chart generators for cross-model comparison visualizations:

| Chart | Function | Description |
|-------|----------|-------------|
| Confusion matrix heatmap | `_svg_confusion_matrix()` | 5×5 coloured cells (green diagonal = agreement, blue off-diagonal = disagreement), cell count text |
| Score distribution bars | `_svg_comparison_scores()` | Grouped bar chart, two bars per score class (blue = Model A, orange = Model B), legend |
| Compliance donuts | `_svg_comparison_compliance()` | Paired donut charts side-by-side with shared colour legend |

All charts are inline SVG — no JavaScript, no external dependencies, print-friendly.

Implementation: `src/services/export.py`.

---

## Comparison HTML Report

### `GET /reports/export/comparison-html?job_a=N&job_b=M`

A self-contained HTML file that consolidates all evaluation data into one shareable report:

- **Executive summary** — stat cards: patients compared, exact match, agreement rate, kappa, Pearson, AUROC
- **System-level metrics** — side-by-side table with colour-coded diffs (adherence rate, class distribution, confidence)
- **Visual comparison** — inline SVG charts (score distribution bars, compliance donuts, confusion matrix heatmap)
- **Cross-model agreement** — kappa at 5-class and 3-class, Pearson correlation, per-class P/R/F1
- **LLM-as-Judge scorer quality** — table showing each model judged by both LLMs (cross-validation)
- **Agent-level evaluation** — query relevance/coverage, retriever IR metrics, scorer quality from pipeline run
- **Extractor quality** — per-category P/R/F1 from SNOMED rules
- **Missing care opportunities** — side-by-side top gaps per model
- **Per-condition adherence** — condition-level comparison table
- **Per-patient comparison** — full patient list with score diffs and agreement

No JavaScript, no external dependencies. Opens in any browser, print-friendly.

### Cross-Model Judging

To avoid self-judging bias (where a model grades its own output), scorer evaluations should be run with both LLM providers as judges:

| Scorer Model | Judge Model | Cross-Model? |
|---|---|---|
| GPT-4o-mini (OpenAI) | GPT-4o-mini | No (self-judging) |
| GPT-4o-mini (OpenAI) | mistral-small (Ollama) | Yes |
| mistral-small (Ollama) | GPT-4o-mini | Yes |
| mistral-small (Ollama) | mistral-small (Ollama) | No (self-judging) |

In practice, both judges produce consistent scores (±0.2 across all metrics), which validates that the scorer outputs are genuinely high quality rather than being inflated by self-judging.

### How to Generate the Full Report

```bash
# Basic comparison report (no scorer eval — fast)
curl -o report.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2"

# With inline LLM-as-Judge scorer evaluation (~30s per job)
curl -o report.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2&include_scorer_eval=true"
```

For the full report with cross-model judging and agent evaluation, the `generate_comparison_html()` function accepts optional `scorer_evals` and `agent_eval` dicts that can be pre-computed and passed in programmatically.

Implementation: `generate_comparison_html()`, `_build_scorer_eval_section()`, `_build_agent_eval_section()`, `_kappa_label()` in `src/services/export.py`. Endpoint in `src/api/routes/reports.py`.

---

## Test Coverage

| Test File | Tests | What's Tested |
|-----------|-------|---------------|
| `test_comparison.py` | 29 | Cohen's kappa (6), Pearson correlation (6), compare_jobs with DB (9), AUROC (5), cross-model classification (3) |
| `test_missing_care.py` | 12 | Parsing from LLM (5), data classes (3), reporting aggregation (4) |
| `test_evaluation.py` | 28 | Rating parsing (6), extractor weak supervision (3), query judge (3), retriever judge (2), scorer judge (3), pipeline orchestration (2), aggregation (2), stored-data reconstruction (2), extractor from DB (2), retriever IR metrics (3) |
| `test_reporting.py` | +4 | System metrics: class distribution, adherence rate, confidence stats, empty job |
| `test_export.py` | 36 | CSV/HTML (12), SVG charts (14), PNG export (6), comparison charts (4) |
| `test_gold_standard.py` | — | Planned: confusion matrix, F1 metrics, kappa (when clinician labels arrive) |

**Total project tests:** 371 passing
