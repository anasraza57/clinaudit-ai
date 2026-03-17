# 12. API Reference — Every Endpoint Explained

## Overview

ClinAuditAI exposes 20+ REST endpoints across 6 groups (Health, Data, Audit, Reports, Exports, Evaluation). This document explains the core endpoints: what they do, when to use them, what the parameters mean, and what the response looks like. For evaluation endpoints, see also `13-evaluation-framework-explained.md`.

All endpoints are also available interactively at **http://localhost:8000/docs** (Swagger UI) when the server is running.

---

## How the Endpoints Fit Together

The typical workflow goes:

```
1. Import data     →  POST /data/import/patients
                      POST /data/import/guidelines

2. Verify import   →  GET  /data/stats

3. Run audits      →  POST /audit/patient/{pat_id}     (one patient)
                      POST /audit/batch                 (many patients)

4. Track progress  →  GET  /audit/jobs/{job_id}

5. View results    →  GET  /audit/jobs/{job_id}/results (all results from a batch)
                      GET  /audit/results/{pat_id}      (all results for one patient)

6. Analyse         →  GET  /reports/dashboard
                      GET  /reports/conditions
                      GET  /reports/non-adherent
                      GET  /reports/score-distribution
```

Steps 1–2 happen once during setup. Steps 3–6 are the ongoing usage.

---

## Group 1: Health (System Status)

### `GET /health` — Liveness Check

**When to use:** To check if the server process is running. Monitoring tools (Docker, load balancers) call this automatically.

**What it checks:** Nothing — if the server responds, it's alive.

**Example response:**
```json
{
  "status": "healthy",
  "service": "ClinAuditAI",
  "environment": "development",
  "timestamp": "2026-03-02T12:00:00+00:00"
}
```

---

### `GET /health/ready` — Readiness Check

**When to use:** To check if the server and its dependencies are configured correctly. Goes beyond liveness — checks if the AI provider API key is set.

**What it checks:** AI provider configuration (is the OpenAI key present?).

**Example response:**
```json
{
  "status": "ready",
  "checks": {
    "ai_provider": "configured"
  },
  "timestamp": "2026-03-02T12:00:00+00:00"
}
```

If the API key is missing, status becomes `"degraded"` and ai_provider shows `"missing_api_key"`.

---

## Group 2: Data (Import & Statistics)

### `GET /api/v1/data/stats` — Database Row Counts

**When to use:** After importing data, to verify everything loaded correctly. Also useful as a quick sanity check.

**What it returns:** Row counts for the three main tables.

**Example response:**
```json
{
  "patients": 4327,
  "clinical_entries": 21530,
  "guidelines": 1656
}
```

If you see `0` for any of these, the import hasn't been run yet.

---

### `POST /api/v1/data/import/patients` — Import Patient Data

**When to use:** During initial setup, to load patient records from `data/msk_valid_notes.csv` into PostgreSQL. This is the API equivalent of running `python3 scripts/import_data.py`.

**What it does:**
1. Reads the CSV file (path from `PATIENT_DATA_PATH` in `.env`)
2. Creates `Patient` records (one per unique `pat_id`)
3. Creates `ClinicalEntry` records (one per row — diagnoses, treatments, referrals, etc.)
4. Skips patients that already exist (matched by `pat_id`)

**Idempotent:** Safe to call multiple times. Second call imports 0 new records.

**Example response:**
```json
{
  "status": "ok",
  "summary": {
    "new_patients": 4327,
    "skipped_patients": 0,
    "new_entries": 21530
  }
}
```

---

### `POST /api/v1/data/import/guidelines` — Import Guidelines

**When to use:** During initial setup, to load NICE guideline documents from `data/guidelines.csv.gz` into PostgreSQL.

**What it does:**
1. Reads the CSV (supports `.csv` or `.csv.gz`)
2. Creates `Guideline` records in the database
3. Skips guidelines that already exist

**Note:** This imports guidelines into the **database** (for browsing/linking). The **FAISS index** (for vector search) is a separate file built by `scripts/build_index.py`. Both use the same source CSV but serve different purposes.

**Example response:**
```json
{
  "status": "ok",
  "summary": {
    "new_guidelines": 1656,
    "skipped": 0
  }
}
```

---

## Group 3: Audit (Pipeline Execution)

These endpoints run the 4-agent pipeline and store results. This is where the actual clinical audit happens.

### `POST /api/v1/audit/patient/{pat_id}` — Audit a Single Patient

**When to use:** To audit one specific patient. Good for testing, debugging, or reviewing individual cases.

**What it does (takes 5–15 seconds):**
1. Looks up the patient's clinical entries from the database
2. **Consultation Insight Agent** — groups entries by episode, categorises each (diagnosis, treatment, referral, etc.)
3. **Audit Query Generator** — generates 1–3 search queries per diagnosis
4. **Guideline Evidence Finder** — encodes queries with PubMedBERT, searches FAISS index for relevant NICE guidelines
5. **Compliance Auditor Agent** — sends each diagnosis + treatments + guidelines to the LLM, which evaluates adherence
6. Stores the result in the `audit_results` table
7. Returns the result immediately

**Parameters:**
- `pat_id` (path, required) — the patient's ID (e.g. `001e1fe6-5660-e486-1501-6f9a8a4c2ec8`)

**Example request:**
```
POST /api/v1/audit/patient/001e1fe6-5660-e486-1501-6f9a8a4c2ec8
```

**Example response (success):**
```json
{
  "status": "completed",
  "pat_id": "001e1fe6-5660-e486-1501-6f9a8a4c2ec8",
  "result": {
    "pat_id": "001e1fe6-...",
    "overall_score": 0.5,
    "total_diagnoses": 2,
    "adherent_count": 1,
    "non_adherent_count": 1,
    "error_count": 0,
    "scores": [
      {
        "diagnosis": "Low back pain",
        "score": 1,
        "explanation": "Patient was appropriately prescribed NSAIDs and referred to physiotherapy, consistent with NICE NG59.",
        "guidelines_followed": ["NICE NG59 - Low back pain and sciatica"],
        "guidelines_not_followed": []
      },
      {
        "diagnosis": "Osteoarthritis of knee",
        "score": -1,
        "explanation": "No documented exercise therapy or weight management advice, which are first-line recommendations.",
        "guidelines_followed": [],
        "guidelines_not_followed": ["NICE CG177 - Osteoarthritis: care and management"]
      }
    ]
  }
}
```

**Understanding the scores:**
- `overall_score: 0.5` — 50% of diagnoses were adherent (1 out of 2)
- `score: 1` — this diagnosis was managed according to guidelines (adherent)
- `score: -1` — this diagnosis was NOT managed according to guidelines (non-adherent)
- `explanation` — the LLM's reasoning for the score
- `guidelines_followed` / `guidelines_not_followed` — specific NICE guidelines referenced

**Example response (failed):**
```json
{
  "status": "failed",
  "pat_id": "001e1fe6-...",
  "error": "No diagnoses found in clinical entries",
  "stage_reached": "extractor"
}
```

This means the pipeline stopped early — the patient had no diagnosis entries to evaluate.

---

### `POST /api/v1/audit/batch` — Start a Batch Audit

**When to use:** To audit many patients at once. The job runs in the background so you don't have to wait.

**What it does:**
1. Creates an `AuditJob` record in the database
2. Returns the job ID immediately
3. Processes patients one-by-one in the background (each with its own DB session for memory safety)
4. Commits progress after every patient (with per-patient timeout of 300s)
5. Marks the job as completed when done (or "failed" if the server crashes — recovered on next startup)

**Parameters (all optional query params):**
- `limit` — maximum number of patients to audit. Omit to audit all.
- `pat_ids` — specific patient IDs to audit. Omit to audit all (or `limit` patients).

**Example requests:**
```bash
# Audit ALL 4,327 patients (takes hours — each calls the LLM)
POST /api/v1/audit/batch

# Audit 50 patients
POST /api/v1/audit/batch?limit=50

# Audit specific patients
POST /api/v1/audit/batch?pat_ids=001e1fe6-...&pat_ids=002f3a7b-...
```

**Example response:**
```json
{
  "status": "accepted",
  "job_id": 1,
  "total_patients": 50,
  "message": "Batch audit started. Poll GET /api/v1/audit/jobs/1 for status."
}
```

**Important:** This only starts the job. To see progress, poll the job status endpoint.

---

### `GET /api/v1/audit/jobs/{job_id}` — Check Job Status

**When to use:** After starting a batch audit, to check how far along it is.

**Parameters:**
- `job_id` (path, required) — the job ID returned by the batch endpoint

**Example request:**
```
GET /api/v1/audit/jobs/1
```

**Example response (running):**
```json
{
  "job_id": 1,
  "status": "running",
  "total_patients": 50,
  "processed_patients": 23,
  "failed_patients": 2,
  "started_at": "2026-03-02T10:30:00+00:00",
  "completed_at": null,
  "error_message": null
}
```

**Status values:**
- `pending` — job created but hasn't started processing yet
- `running` — actively processing patients
- `completed` — all patients processed
- `failed` — the job itself crashed (not individual patients — those are counted in `failed_patients`)

**Tip:** Poll every 5–10 seconds. When `status` changes to `completed`, the results are ready.

---

### `GET /api/v1/audit/jobs/{job_id}/results` — Batch Job Results

**When to use:** After a batch job completes, to see all the results from that specific run.

**Parameters:**
- `job_id` (path, required) — the job ID
- `page` (query, default: 1) — page number
- `page_size` (query, default: 20, max: 100) — results per page

**Ordering:** Results are sorted by `pat_id` then `AuditResult.id` for stable, deterministic pagination. The same page always returns the same results.

**Example request:**
```
GET /api/v1/audit/jobs/1/results?page=1&page_size=10
```

**Example response:**
```json
{
  "job_id": 1,
  "total": 50,
  "page": 1,
  "page_size": 10,
  "total_pages": 5,
  "results": [
    {
      "pat_id": "001e1fe6-...",
      "overall_score": 0.5,
      "diagnoses_found": 2,
      "guidelines_followed": 1,
      "guidelines_not_followed": 1,
      "status": "completed",
      "error_message": null,
      "details": { ... }
    },
    ...
  ]
}
```

The `details` field contains the full scoring breakdown (same structure as the single patient audit response).

---

### `GET /api/v1/audit/results/{pat_id}` — Patient Results

**When to use:** To look up all audit results for a specific patient, across all jobs. A patient may have been audited multiple times if you ran multiple batch jobs.

**Parameters:**
- `pat_id` (path, required) — the patient's ID

**Example request:**
```
GET /api/v1/audit/results/001e1fe6-5660-e486-1501-6f9a8a4c2ec8
```

**Example response:**
```json
{
  "pat_id": "001e1fe6-...",
  "total_results": 2,
  "results": [
    {
      "pat_id": "001e1fe6-...",
      "overall_score": 0.5,
      "diagnoses_found": 2,
      "guidelines_followed": 1,
      "guidelines_not_followed": 1,
      "status": "completed",
      "error_message": null,
      "details": { ... }
    },
    {
      "pat_id": "001e1fe6-...",
      "overall_score": 0.33,
      "diagnoses_found": 3,
      "guidelines_followed": 1,
      "guidelines_not_followed": 2,
      "status": "completed",
      "error_message": null,
      "details": { ... }
    }
  ]
}
```

**Why might there be multiple results?** If you ran a batch job on Monday and another on Friday, the patient would have two results. The most recent is listed first.

### Difference between the two results endpoints

| Endpoint | Question it answers |
|----------|-------------------|
| `GET /audit/jobs/{job_id}/results` | "What were ALL the results from batch run #3?" |
| `GET /audit/results/{pat_id}` | "What are all the results for THIS specific patient?" |

One is scoped by job (batch run), the other by patient. Both return the same result structure.

---

## Group 4: Reports (Analytics)

These endpoints aggregate audit results into summary statistics. They only work after you've run some audits — if no results exist, they return zeros/empty lists.

All report endpoints accept an optional `?job_id=N` query parameter to scope the report to a specific batch run. Omit it to report across all completed audits.

### `GET /api/v1/reports/dashboard` — Dashboard Summary

**When to use:** For a high-level overview of audit results. "How are we doing overall?"

**Parameters:**
- `job_id` (query, optional) — scope to a specific batch job

**Example request:**
```
GET /api/v1/reports/dashboard
GET /api/v1/reports/dashboard?job_id=1
```

**Example response:**
```json
{
  "total_audited": 50,
  "total_failed": 3,
  "failure_rate": 0.057,
  "score_stats": {
    "mean": 0.42,
    "median": 0.33,
    "min": 0.0,
    "max": 1.0
  }
}
```

**Understanding the numbers:**
- `total_audited: 50` — 50 patients were successfully audited
- `total_failed: 3` — 3 patients' audits failed (no diagnoses, LLM error, etc.)
- `failure_rate: 0.057` — 5.7% of audits failed (3 out of 53 total)
- `mean: 0.42` — on average, 42% of each patient's diagnoses were adherent
- `median: 0.33` — the middle patient had 33% adherence (median is less affected by outliers than mean)
- `min: 0.0` — the worst patient had 0% adherence (all diagnoses non-adherent)
- `max: 1.0` — the best patient had 100% adherence (all diagnoses adherent)

---

### `GET /api/v1/reports/conditions` — Per-Condition Breakdown

**When to use:** To see which medical conditions have the best/worst guideline adherence. "Which conditions are doctors not following guidelines for?"

**Parameters:**
- `job_id` (query, optional) — scope to a specific batch job
- `min_count` (query, default: 1) — only include conditions with at least this many cases
- `sort_by` (query, default: `"count"`) — `"count"` (most common first) or `"adherence_rate"` (worst adherence first)

**Example request:**
```
GET /api/v1/reports/conditions?sort_by=adherence_rate&min_count=3
```

**Example response:**
```json
[
  {
    "diagnosis": "Osteoarthritis of knee",
    "total_cases": 15,
    "adherent": 3,
    "non_adherent": 11,
    "errors": 1,
    "adherence_rate": 0.214
  },
  {
    "diagnosis": "Low back pain",
    "total_cases": 22,
    "adherent": 12,
    "non_adherent": 9,
    "errors": 1,
    "adherence_rate": 0.571
  },
  {
    "diagnosis": "Carpal tunnel syndrome",
    "total_cases": 5,
    "adherent": 4,
    "non_adherent": 1,
    "errors": 0,
    "adherence_rate": 0.8
  }
]
```

**Understanding the numbers:**
- `total_cases: 15` — 15 patients had this diagnosis
- `adherent: 3` — 3 were scored +1 (treatment followed guidelines)
- `non_adherent: 11` — 11 were scored -1 (treatment didn't follow guidelines)
- `errors: 1` — 1 had a scoring error (LLM parse failure, excluded from rate)
- `adherence_rate: 0.214` — 21.4% of scoreable cases were adherent (3 out of 14, excluding errors)

**Use case:** A clinical director could look at this and say "We need to improve our osteoarthritis management — only 21% adherence."

---

### `GET /api/v1/reports/non-adherent` — Non-Adherent Cases

**When to use:** For clinical review — see every case flagged as non-adherent, with the LLM's explanation of what went wrong and which guidelines weren't followed.

**Parameters:**
- `job_id` (query, optional) — scope to a specific batch job
- `page` (query, default: 1) — page number
- `page_size` (query, default: 50, max: 200) — results per page

**Example request:**
```
GET /api/v1/reports/non-adherent?page=1&page_size=10
```

**Example response:**
```json
{
  "total": 87,
  "page": 1,
  "page_size": 10,
  "total_pages": 9,
  "cases": [
    {
      "pat_id": "001e1fe6-...",
      "diagnosis": "Osteoarthritis of knee",
      "index_date": "2023-05-15",
      "explanation": "No documented exercise therapy or weight management advice. NICE CG177 recommends exercise as a core treatment for all patients with osteoarthritis before considering pharmacological options.",
      "guidelines_not_followed": [
        "NICE CG177 - Osteoarthritis: care and management"
      ]
    },
    {
      "pat_id": "002f3a7b-...",
      "diagnosis": "Gout",
      "index_date": "2023-08-22",
      "explanation": "Patient prescribed allopurinol without documented urate level monitoring. NICE CG176 recommends checking serum urate levels before and during treatment.",
      "guidelines_not_followed": [
        "NICE CG176 - Gout: diagnosis and management"
      ]
    },
    ...
  ]
}
```

**Use case:** A clinical auditor reviews these cases one by one. The `explanation` tells them exactly what the LLM found, and `guidelines_not_followed` points to the specific NICE guideline for reference.

---

### `GET /api/v1/reports/score-distribution` — Score Histogram

**When to use:** To visualise how adherence scores are distributed across the patient population. "Are most patients clustered at 0%, or is it a spread?"

**Parameters:**
- `job_id` (query, optional) — scope to a specific batch job
- `bins` (query, default: 10, range: 2–100) — number of histogram bins

**Example request:**
```
GET /api/v1/reports/score-distribution?bins=5
```

**Example response:**
```json
{
  "bins": [
    {"bin_start": 0.0, "bin_end": 0.2, "count": 18},
    {"bin_start": 0.2, "bin_end": 0.4, "count": 12},
    {"bin_start": 0.4, "bin_end": 0.6, "count": 8},
    {"bin_start": 0.6, "bin_end": 0.8, "count": 7},
    {"bin_start": 0.8, "bin_end": 1.0, "count": 5}
  ],
  "total": 50
}
```

**Understanding the numbers:**
- 18 patients scored between 0.0–0.2 (very low adherence)
- 5 patients scored between 0.8–1.0 (high adherence)
- The distribution skews left — most patients have low adherence scores

**Use case:** If you're building a frontend dashboard, this data feeds directly into a bar chart / histogram.

---

## Common Patterns

### The `?job_id=N` and `?model=X` Parameters

All report and export endpoints accept `?job_id=N` to scope to a specific batch run, and `?model=X` to scope to all results from a specific LLM model (combining results across multiple batch jobs that used the same model).

```bash
# Results from Monday's run
GET /api/v1/reports/dashboard?job_id=1

# Results from Friday's run (maybe after changing LLM settings)
GET /api/v1/reports/dashboard?job_id=2

# All results from GPT-4.1-mini (across all jobs that used this model)
GET /api/v1/reports/dashboard?model=gpt-4.1-mini

# All results from Mistral-Small
GET /api/v1/reports/conditions?model=mistral-small&sort_by=adherence_rate
```

Omit both `job_id` and `model` to see results across ALL completed audits. The comparison HTML export also supports `?model_a=X&model_b=Y` as an alternative to `?job_a=N&job_b=M`.

### Pagination

Three endpoints use pagination: batch job results, non-adherent cases, and patient results. They all follow the same pattern:

```json
{
  "total": 87,
  "page": 1,
  "page_size": 10,
  "total_pages": 9,
  "results": [...]
}
```

- `total` — total number of items across all pages
- `page` / `page_size` — current page and items per page
- `total_pages` — calculated from total and page_size
- Request page 2 with `?page=2&page_size=10`

### Error Responses

All endpoints return standard HTTP error codes:

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Normal response |
| 400 | Bad request | `POST /audit/batch` with no patients in database |
| 404 | Not found | `GET /audit/jobs/999` — job doesn't exist |
| 422 | Validation error | `?page=-1` — invalid parameter value |
| 500 | Server error | Database connection lost, LLM API failure |

FastAPI automatically returns 422 with details when query parameters fail validation (wrong type, out of range, etc.).
