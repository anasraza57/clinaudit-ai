# Pipeline Integration Explained

## What Does Phase 6 Do?

Phases 2-5 built the four agents individually. Phase 6 **wires them together** into a single, end-to-end audit pipeline and exposes it via REST API endpoints.

Before Phase 6:
```
We had 4 agents, each tested independently with mock data.
No way to actually run an audit on a real patient.
```

After Phase 6:
```
POST /api/v1/audit/patient/001e1fe6-5660-e486-1501-6f9a8a4c2ec8
→ Reads patient from DB
→ Runs all 4 agents in sequence
→ Stores result in DB
→ Returns scoring result
```

## The Pipeline Orchestrator

### Why a Pipeline Class?

Both Hiruni and Cyprian used **LangGraph** to wire their agents together. We chose a simple Python class instead (Decision 003 in the PROJECT_BIBLE). Here's why this works better for us:

**LangGraph approach** (what they did):
```python
workflow = StateGraph(State)
workflow.add_node("extractor", extractor_node)
workflow.add_node("query", query_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("scorer", scorer_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "query")
workflow.add_edge("query", "retriever")
workflow.add_edge("retriever", "scorer")
workflow.add_edge("scorer", END)
graph = workflow.compile()
```

**Our approach** (what we did):
```python
# Stage 1: Extract
extraction = self._extractor.extract(pat_id, entries)

# Stage 2: Generate queries
query_result = await self._query_agent.generate_queries(extraction)

# Stage 3: Retrieve guidelines
retrieval = self._retriever.retrieve(query_result)

# Stage 4: Score adherence
scoring = await self._scorer.score(extraction, retrieval)
```

Both achieve the same result. But our version:
- Is readable at a glance (no graph compilation needed)
- Makes the data flow explicit (each stage's output feeds the next)
- Shows that the Scorer receives **two** inputs (extraction + retrieval)
- Is trivially testable (mock any stage, verify the rest)
- Has no external dependency

The LangGraph version adds a `StateGraph`, `TypedDict` state, edge definitions, and a compiled graph — all for what is fundamentally **four function calls in sequence**.

### The AuditPipeline Class

```
src/services/pipeline.py
```

The class has three responsibilities:

1. **Initialisation** — creates all 4 agents with their dependencies (AI provider, embedder, vector store)
2. **Single patient audit** — `run_single(session, pat_id)` runs the full pipeline
3. **Batch processing** — `run_batch(session, pat_ids)` processes many patients with job tracking

### Data Flow Through the Pipeline

```
Database
  │
  ├── Patient "001e1fe6..."
  │   ├── ClinicalEntry: Finger fracture (2024-09-04)
  │   ├── ClinicalEntry: Blood pressure recorded (2024-06-17)
  │   ├── ClinicalEntry: Ibuprofen prescribed (2024-09-04)
  │   └── ClinicalEntry: Physiotherapy referral (2024-09-10)
  │
  ▼
[_load_patient_entries] → list of dicts
  │
  ▼
[Stage 1: Extractor] → ExtractionResult
  │  Groups by index_date, categorises each entry
  │  Identifies 1 diagnosis: "Finger fracture"
  │
  ▼
[Stage 2: Query Agent] → QueryResult
  │  Generates 1-3 search queries per diagnosis
  │  e.g., "NICE guidelines for finger fracture management"
  │
  ▼
[Stage 3: Retriever] → RetrievalResult
  │  Embeds queries with PubMedBERT
  │  Searches FAISS for top-5 guideline matches
  │  Deduplicates across queries
  │
  ▼
[Stage 4: Scorer] → ScoringResult
  │  Sends diagnosis + treatments + guidelines to LLM
  │  Gets: Score +1, "Appropriate treatment prescribed"
  │
  ▼
[_store_result] → AuditResult row in database
  │  overall_score: 1.0
  │  diagnoses_found: 1
  │  guidelines_followed: 1
  │  details_json: {"scores": [...]}
  │
  ▼
API Response
```

### Diagnosis Deduplication

A single patient may have the same diagnosis appearing multiple times — e.g., "Finger pain" recorded 4 times across 2 episodes. Without dedup, this triggers 4 identical LLM query calls, 4 identical PubMedBERT encodings, 4 identical FAISS searches, and 4 identical scorer LLM calls. Worse, duplicate entries flow all the way to the reports, showing the same diagnosis twice in the non-adherent list.

Dedup operates at **two layers** in each pipeline stage:

**Layer 1 — Entry-level dedup** (eliminates duplicate output entries):
Each stage tracks `seen_pairs: set[tuple[str, str]]` keyed by `(diagnosis_term, index_date)`. If the same diagnosis appears twice in the same episode (e.g. "Finger pain" coded twice on 2024-03-07), the second occurrence is **skipped entirely**. This ensures each diagnosis appears exactly once per episode in all downstream output — no duplicate entries in reports.

**Layer 2 — Term-level caching** (avoids redundant compute):
When the same diagnosis term appears in a *different* episode (different index_date), a new output entry is created but cached results are reused:

| Stage | Cache Key | What's Reused |
|-------|-----------|-------------|
| **Query Agent** | `diagnosis_term` | Generated queries (template or LLM) |
| **Retriever** | `diagnosis_term` | Embeddings + FAISS search results |
| **Scorer** | N/A (each episode scored independently) | — |

The Query Agent and Retriever cache by diagnosis term alone — the same diagnosis always produces the same queries and the same guideline matches regardless of which episode it appears in. The Scorer doesn't need term-level caching because different episodes may have different treatments/referrals.

For our example patient with 4x "Finger pain" + 2x "Finger fracture" across 2 episodes:
- **Before:** 4 LLM query calls, 6 encode+search, 6 scorer LLM calls = **10 LLM calls**, **6 report entries** (with duplicates)
- **After:** 1 LLM query call, 2 encode+search, 2 scorer LLM calls = **3 LLM calls**, **4 report entries** (no duplicates)

At batch scale (4,327 patients), this saves thousands of LLM calls and eliminates confusing duplicate entries in reports.

### Early Exit Points

The pipeline can stop early at several points:

1. **No patient found** — patient UUID doesn't exist in DB
2. **No clinical entries** — patient exists but has no clinical data
3. **No diagnoses** — entries exist but none are categorised as diagnoses (all treatments/admin/etc.)
4. **Agent failure** — any stage throws an exception (caught, stored as error)

In each case, a `PipelineResult` is still returned with the `error` field set and `stage_reached` indicating where it stopped. The result is stored in the DB as a "failed" `AuditResult` so we know it was attempted.

### Error Handling Strategy

```python
try:
    # Run all 4 stages...
except Exception as e:
    logger.error("Pipeline failed for patient %s at stage %s: %s",
                 pat_id, pipeline_result.stage_reached, e)
    pipeline_result.error = str(e)

# Always store result (even on failure)
await self._store_result(session, pat_id, pipeline_result, job_id)
```

The key principle: **never lose work**. Even if the Scorer fails for patient 42, we:
- Log the error with patient ID and stage
- Store a "failed" AuditResult in the DB
- Continue processing patient 43

### Crash Resilience (Batch Processing)

The batch pipeline has several layers of protection against crashes:

#### 1. Per-Patient Session Isolation

Each patient gets its own short-lived DB session:

```python
for pat_id in pat_ids:
    async with factory() as session:          # Fresh session per patient
        result = await asyncio.wait_for(
            pipeline.run_single(session, pat_id),
            timeout=patient_timeout,           # 300s default
        )
        # Update progress and commit
        await session.commit()
    # Session closed → identity map freed → memory stays constant
```

**Why this matters:** The original code used a single session for the entire batch. SQLAlchemy's identity map (which tracks every ORM object) grew with every patient — all Patient, ClinicalEntry, and AuditResult objects accumulated in memory. For thousands of patients, this caused out-of-memory crashes.

With per-patient sessions, the identity map is discarded after each patient, keeping memory usage constant regardless of batch size.

#### 2. Timeout Protection

Two levels of timeout prevent indefinite hangs:

| Level | Setting | Default | What it protects |
|-------|---------|---------|-----------------|
| **Per-LLM call** | `openai_request_timeout` | 60s | Single OpenAI API call (was 10 min default) |
| **Per-patient** | `pipeline_patient_timeout` | 300s | Entire pipeline for one patient (all 4 stages) |

If a patient times out, the batch continues with the next patient. The timed-out patient is recorded as "failed" with an error message.

#### 3. Error Recovery

When a patient fails (timeout or exception), the `_save_patient_error_and_progress()` helper opens a **clean session** to:
- Store a "failed" AuditResult for that patient (so we know it was attempted)
- Update the job's progress counters

This is needed because a timeout kills the patient's session, so we need a fresh session to record the failure.

#### 4. Garbage Collection

Every 10 patients, `gc.collect()` is called to force Python's garbage collector to free memory from completed patients. Also called after the batch completes.

#### 5. Stale Job Recovery

If the server crashes mid-batch, jobs get stuck as "pending" or "running" forever. On startup, `_recover_stale_jobs()` finds any stuck jobs and marks them as "failed":

```python
# In main.py lifespan handler:
async def _recover_stale_jobs(logger):
    # Find jobs stuck as pending or running
    stale = select(AuditJob).where(
        or_(AuditJob.status == "pending", AuditJob.status == "running")
    )
    # Mark them as failed
    for job in stale_jobs:
        job.status = "failed"
        job.error_message = "Server restarted before job could complete"
```

This runs once on every server startup — no more ghost jobs.

### SNOMED Category Loading and Persistence

The Extractor needs SNOMED categories to classify entries. The 1,261 unique concepts are loaded **once** before processing, with a three-step flow:

```python
# load_categories_from_db() does:
# 1. Read concepts with category already set in DB → skip
# 2. Classify remaining concepts (rules + batched LLM)
# 3. Write new categories back to DB → never redo this work
# 4. Populate in-memory cache → Extractor uses this during pipeline
await pipeline.load_categories_from_db(session)

# Now process patients (categories are in memory)
for pat_id in pat_ids:
    await pipeline.run_single(session, pat_id)
```

**First run:** ~7 batched LLM calls (50 concepts per prompt) to classify ~322 unmatched concepts. Categories are written to the `clinical_entries.category` column in the database.

**Every subsequent run:** All categories load from DB. Zero LLM calls. If the server crashes between runs, no categorisation work is lost.

This was a critical fix — the original implementation made 322 individual LLM calls (one per concept), which caused the server to crash from memory exhaustion.

## The API Endpoints

### Endpoint Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/audit/patient/{pat_id}` | POST | Audit a single patient (synchronous) |
| `/api/v1/audit/batch` | POST | Start a batch audit job (background, supports `?limit=N`) |
| `/api/v1/audit/jobs/{job_id}` | GET | Check batch job progress |
| `/api/v1/audit/jobs/{job_id}/results` | GET | Get paginated results for a batch job |
| `/api/v1/audit/results/{pat_id}` | GET | Get all results for a specific patient |

### Service Loading on Request

Both the single-patient and batch endpoints verify that critical services are loaded before processing. After uvicorn's `--reload` restarts the server, singletons may not be populated yet:

```python
# In both audit_single_patient and _run_batch_background:
if not pipeline._retriever._embedder.is_loaded:
    pipeline._retriever._embedder.load()
if not pipeline._retriever._vector_store.is_loaded:
    pipeline._retriever._vector_store.load()
```

The vector store also auto-decompresses `guidelines.csv.gz` if the uncompressed CSV is missing:

```python
# In vector_store.load():
if not csv_path.exists():
    gz_path = Path(str(csv_path) + ".gz")
    if gz_path.exists():
        # Decompresses once, then CSV exists for future loads
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
```

This means the repo only stores the compressed `.gz` file, and the system self-bootstraps on first run.

### Single Patient Audit

```
POST /api/v1/audit/patient/001e1fe6-5660-e486-1501-6f9a8a4c2ec8
```

This runs **synchronously** — the request blocks until all 4 stages complete. Good for testing individual patients or debugging.

Response on success:
```json
{
  "status": "completed",
  "pat_id": "001e1fe6-...",
  "result": {
    "pat_id": "001e1fe6-...",
    "total_diagnoses": 2,
    "adherent": 1,
    "non_adherent": 1,
    "aggregate_score": 0.5,
    "scores": [...]
  }
}
```

### Batch Audit

```
POST /api/v1/audit/batch                    // audit all patients
POST /api/v1/audit/batch?limit=50           // audit 50 patients
POST /api/v1/audit/batch?pat_ids=001e&pat_ids=002f  // specific patients
```

This starts a **background job** — the request returns immediately with a job ID, and the pipeline processes patients in the background.

```json
{
  "status": "accepted",
  "job_id": 1,
  "total_patients": 4327,
  "message": "Poll GET /api/v1/audit/jobs/1 for status."
}
```

The background task creates a **fresh DB session per patient** (for memory isolation) and **commits progress after every patient** so polling sees real-time updates.

### Job Status

```
GET /api/v1/audit/jobs/1
```

```json
{
  "job_id": 1,
  "status": "running",
  "total_patients": 4327,
  "processed_patients": 150,
  "failed_patients": 3,
  "started_at": "2026-03-02T10:30:00Z",
  "completed_at": null
}
```

### Job Results

```
GET /api/v1/audit/jobs/1/results?page=1&page_size=20
```

Returns paginated audit results for a specific batch job. Each result includes the patient's overall score, diagnosis counts, and the full scoring breakdown.

## Database Storage

### AuditJob Table

Tracks batch runs:
- `status`: pending → running → completed/failed
- `total_patients`, `processed_patients`, `failed_patients`: progress counters
- `started_at`, `completed_at`: timing

### AuditResult Table

Stores per-patient outcomes:
- `patient_id`: FK to patients table
- `job_id`: FK to audit_jobs (null if run individually)
- `overall_score`: 0.0 to 1.0 (aggregate adherence)
- `diagnoses_found`, `guidelines_followed`, `guidelines_not_followed`: summary counts
- `details_json`: full ScoringResult as JSON (per-diagnosis scores, explanations, guidelines)
- `status`: completed or failed
- `error_message`: what went wrong (if failed)

The `details_json` field stores the complete scoring breakdown, so we never lose information. The summary columns (`overall_score`, etc.) enable efficient querying without parsing JSON.

## Comparison: Cyprian's Approach vs Ours

| Aspect | Cyprian | Ours |
|--------|---------|------|
| **Orchestration** | LangGraph + Flask JSON-RPC servers | Simple Python class with function composition |
| **Communication** | HTTP between Flask servers on different ports | Direct function calls |
| **State** | Global mutable dict | Dataclass per stage, passed explicitly |
| **DB storage** | None (in-memory only) | PostgreSQL with AuditJob + AuditResult |
| **Batch processing** | Not supported | Background tasks with progress tracking |
| **Error handling** | None (crashes on failure) | Per-patient error capture, continues processing |
| **API** | Flask JSON-RPC (custom protocol) | FastAPI REST (standard, auto-documented) |
| **Category loading** | Per-patient | Once, persisted to DB, cached in memory |
| **Crash recovery** | None | Stale job recovery + per-patient session isolation |
| **Timeouts** | None | 60s per LLM call + 300s per patient |

## Test Coverage

14 new tests covering:

- **PipelineResult** (5 tests): success/failure detection, summary output
- **AuditPipeline** (9 tests): initialisation, category loading, successful pipeline, no entries, no diagnoses, scorer error, result storage, job ID passing, stage ordering

## What Happens Next?

Phase 7 (Validation & Reporting) will:
1. Import the 120 gold-standard human audit results
2. Run our pipeline against those 120 patients
3. Compare our AI scores vs human auditor scores
4. Calculate accuracy, agreement (Cohen's kappa), and other metrics
5. Build reporting endpoints for aggregate analysis
