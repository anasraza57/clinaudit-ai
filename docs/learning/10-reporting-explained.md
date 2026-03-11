# 10. Reporting Endpoints — Explained

## What This Phase Built

Phase 7a added **read-only analytics endpoints** on top of the audit results produced by the pipeline. While Phase 6 built the "write path" (run audits, store results), this phase builds the "read path" (aggregate and present results).

**New files:**
- `src/services/reporting.py` — computation layer (4 public functions + 1 helper)
- `src/api/routes/reports.py` — thin route layer (4 GET endpoints + Pydantic schemas)
- `tests/unit/test_reporting.py` — 26 tests using in-memory SQLite

## Architecture: Service Layer Pattern

We split reporting into two layers:

```
HTTP Request → Route (reports.py) → Service (reporting.py) → Database
                  │                        │
                  │ Pydantic schemas        │ SQLAlchemy queries
                  │ Query params            │ JSON parsing
                  │ HTTP concerns           │ Business logic
```

**Why separate?**
- **Testability:** Service functions can be tested directly with a database session — no need for HTTP clients or FastAPI TestClient.
- **Extensibility:** When gold-standard validation comes (Phase 7b), we add `get_validation_metrics()` to the service and a new endpoint to the routes. The pattern is established.
- **Readability:** Routes are 5 lines each (parse params → call service → return). All logic lives in the service.

## The Four Report Functions

### 1. `get_dashboard_stats()` — High-Level Summary

**What it returns:** Total audited/failed counts, mean/median/min/max adherence score, failure rate.

**How it works:** Pure SQL aggregation — `COUNT`, `AVG`, `MIN`, `MAX` via SQLAlchemy `func`. The one exception is **median**, which has no SQL standard function, so we load all scores into Python and compute it there (perfectly fine for ~4,327 values).

**Key design choice:** Uses only SQL columns (`overall_score`, `status`), never parses `details_json`. This keeps the function fast and avoids coupling to the JSON schema.

### 2. `get_condition_breakdown()` — Per-Diagnosis Adherence

**What it returns:** For each diagnosis term: total cases, adherent count, non-adherent count, adherence rate.

**How it works:** Loads all completed AuditResults, parses each `details_json` to extract the `scores` array, groups by `diagnosis` term, and counts adherent (+1) vs non-adherent (-1) per group.

**Parameters:**
- `min_count` — filters out diagnoses with fewer than N total cases (default 1)
- `sort_by` — `"count"` (descending, default) or `"adherence_rate"` (ascending, worst-first)

**Why Python-side aggregation?** The `details_json` column is `TEXT`, not PostgreSQL `JSONB`. We can't use SQL JSON functions. But with ~4,327 results, loading them all into Python and parsing with `json.loads()` takes milliseconds. No need for the complexity of database-side JSON extraction.

### 3. `get_non_adherent_cases()` — Clinical Review List

**What it returns:** Paginated list of every diagnosis that scored -1 (non-adherent), with patient ID, explanation, and guidelines not followed.

**How it works:** Similar to condition breakdown — parses `details_json`, but filters for `score == -1` entries. Also eager-loads the `Patient` relationship via `selectinload` to get the patient's UUID (`pat_id`).

**Pagination:** Returns `page`, `page_size`, `total`, `total_pages`, and the `cases` slice. Handles edge cases like requesting a page beyond the data (returns empty `cases` with correct `total`).

**Purpose:** This is the endpoint clinicians would use to review cases flagged as non-adherent. Each case includes the explanation (from the Scorer Agent's LLM output) and the specific guidelines that weren't followed.

### 4. `get_score_distribution()` — Score Histogram

**What it returns:** A histogram dividing the 0.0–1.0 range into equal bins, with the count of patients in each bin.

**How it works:** Loads all `overall_score` values (SQL only), then bins them in Python. The last bin includes the right edge (so a score of exactly 1.0 falls in [0.9, 1.0] not nowhere).

**Parameters:**
- `bins` — number of histogram bins (default 10, range 2–100)

**Edge cases:** An empty dataset returns `{"bins": [], "total": 0}`.

## The Shared Helper: `_load_completed_results()`

All functions that parse `details_json` share the same base query: "give me all AuditResults with status='completed', optionally filtered by job_id." This helper avoids duplication:

```python
async def _load_completed_results(session, job_id=None, include_details=False):
    query = select(AuditResult).where(AuditResult.status == "completed")
    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)
    if include_details:
        query = query.options(selectinload(AuditResult.patient))
    result = await session.execute(query)
    return list(result.scalars().all())
```

The `include_details` flag controls whether to eager-load the Patient relationship (an extra SQL query). Only `get_non_adherent_cases()` needs this (to show `pat_id`); `get_condition_breakdown()` doesn't.

## Job Scoping

All 4 endpoints accept an optional `?job_id=N` query parameter. This scopes the report to a single batch job, letting you compare results across different runs. If omitted, the report covers all completed results in the database.

## Testing Strategy: In-Memory SQLite

**The problem:** Reporting functions run real SQL queries with `func.count()`, `func.avg()`, `select().where()`, etc. Mocking `session.execute()` would be extremely fragile — `get_dashboard_stats()` alone makes 4 separate execute calls, each returning different types (`scalar`, `one`, `all`).

**The solution:** We use `aiosqlite` to create an in-memory SQLite database for each test:

```python
@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # ... yield session
```

This gives us:
- **Real SQL execution** — tests validate actual query logic, not mock behaviour
- **No external dependencies** — no PostgreSQL needed, runs anywhere
- **Speed** — 26 tests complete in under 1 second
- **Schema sync** — uses the same SQLAlchemy models, so tables match production

**Helper functions** (`_make_patient`, `_add_completed_result`, `_add_failed_result`, `_make_job`) make test setup concise — each test creates its own isolated data and verifies the output.

## Pydantic Response Schemas

Each endpoint has typed response models:

```python
class DashboardResponse(BaseModel):
    total_audited: int
    total_failed: int
    failure_rate: float
    score_stats: ScoreStatsSchema  # mean, median, min, max

class ConditionBreakdownItem(BaseModel):
    diagnosis: str
    total_cases: int
    adherent: int
    non_adherent: int
    errors: int
    adherence_rate: float
```

FastAPI uses these for:
- **Validation** — ensures service functions return the right structure
- **Serialization** — converts Python dicts to proper JSON
- **Documentation** — Swagger UI shows the exact response shape

## Inline SVG Charts (Phase 9e)

The HTML report now includes **three inline SVG charts** generated server-side in Python. No JavaScript, no external dependencies — the charts are pure SVG elements embedded directly in the HTML.

### Chart Types

1. **Score Distribution Bar Chart** (`_svg_score_distribution`) — Histogram of patient-level adherence scores binned into 5 ranges (0-20%, 20-40%, 40-60%, 60-80%, 80-100%). Bars are colour-coded from red (low) to green (high). Y-axis shows patient counts.

2. **Compliance Donut Chart** (`_svg_compliance_donut`) — Donut/ring chart showing the distribution of 5-level compliance scores (+2 Compliant, +1 Partial, 0 N/R, -1 Non-compliant, -2 Risky) across all diagnoses. Centre shows total diagnosis count. Legend lists each level with its count.

3. **Per-Condition Horizontal Bar Chart** (`_svg_condition_bars`) — Horizontal bars showing adherence rate per diagnosis. Bars are colour-coded by threshold (green ≥70%, amber ≥40%, red <40%). Long condition names are auto-truncated.

### Design Decisions

- **Inline SVG over Chart.js/D3** — No JavaScript means the report works offline, in email clients, and in print. SVG is vector-based so it scales perfectly.
- **Server-side generation** — Charts are computed at report generation time, not in the browser. This keeps the HTML completely self-contained.
- **Conditional rendering** — Charts section only appears when there is data. Empty reports render cleanly without blank chart areas.
- **2-column grid layout** — Score distribution and compliance donut sit side-by-side; condition bars span full width below.

### How It Works

During `generate_html_report()`, the code collects `scores` (patient-level floats) and `level_counts` (dict of compliance levels) as it processes results. These are passed to `_build_html()`, which calls the three chart helpers and embeds the SVG output into the HTML template.

### PNG Export for Reports

For embedding charts in Word/LaTeX reports, the system can export charts as standalone PNG files:

```bash
# Save charts from all audit data
DB_HOST=localhost python scripts/export_charts.py --output exports/charts

# Scope to a specific batch job, higher DPI for print
DB_HOST=localhost python scripts/export_charts.py --output exports/charts --job-id 1 --dpi 300
```

This produces:
- `score_distribution.png` — bar chart histogram
- `compliance_breakdown.png` — compliance donut chart
- `condition_adherence.png` — per-condition horizontal bars

**Implementation**: `export_charts_to_png()` in `src/services/export.py` reuses the same SVG chart helpers, then converts to PNG via `cairosvg`. The data collection is shared with the HTML report via `_collect_chart_data()` — no duplication.

**Dependencies**: Requires `cairosvg` (Python) + `cairo` system library (`brew install cairo` on macOS). The import is wrapped in `try/except OSError` so the rest of the export module works even without cairo installed.

## What's Next

Gold-standard validation will be added when the 120 clinician-labeled cases arrive. The framework (`src/services/gold_standard.py`) is already planned: confusion matrix, per-class P/R/F1, weighted kappa, all computed from stored results.
