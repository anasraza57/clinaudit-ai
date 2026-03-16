# PROJECT BIBLE — ClinAuditAI

> **Last Updated:** 2026-03-16
> **Status:** Phase 9 COMPLETE + Phase 10a Retriever Relevance Filter ✅ + Phase 10b Comprehensive Evaluation Endpoints ✅ + Phase 10c Comparison HTML Report ✅ + Deterministic evaluation ordering ✅ + Phase 11d Comprehensive Individual Reports ✅ — Individual HTML reports now include all evaluation sections (system metrics, extractor, scorer eval, agent eval, missing care) with `use_saved_evals` support

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Reference Codebases Analysis](#2-reference-codebases-analysis)
3. [Architecture & Tech Stack](#3-architecture--tech-stack)
4. [Master Roadmap](#4-master-roadmap)
5. [Progress Tracker](#5-progress-tracker)
6. [Decisions Log](#6-decisions-log)
7. [Current State Summary](#7-current-state-summary)
8. [Known Issues / Tech Debt](#8-known-issues--tech-debt)
9. [Environment & Setup](#9-environment--setup)

---

## 1. Project Overview

### What We're Building

An AI-powered clinical audit system that automatically evaluates whether GP (general practitioner) consultations for **musculoskeletal (MSK) conditions** adhere to **NICE clinical guidelines**.

### The Problem

In the UK, MSK conditions (back pain, osteoarthritis, fractures, etc.) account for ~15% of all GP appointments (~14 million visits/year). NICE publishes evidence-based guidelines on how these conditions should be managed — what to prescribe, when to refer to a specialist, what imaging to order, etc.

Currently, checking whether doctors follow these guidelines requires **manual chart review** by trained clinicians. This is:
- Extremely slow (the CrossCover trial could only audit 120 out of 10,000+ cases manually)
- Expensive (requires expert clinician time)
- Inconsistent (different auditors may judge differently)
- Impossible to scale

### The Solution

A 4-agent AI pipeline that processes patient records and scores them against clinical guidelines:

```
Patient Record → [Consultation Insight Agent] → [Audit Query Generator] → [Guideline Evidence Finder] → [Compliance Auditor Agent] → Audit Report
```

1. **Consultation Insight Agent** (`ConsultationInsightAgent`) — Reads structured patient data (SNOMED-coded clinical entries), categorises each entry as a diagnosis, treatment, procedure, referral, etc.
2. **Audit Query Generator** (`AuditQueryGenerator`) — Takes extracted clinical concepts and generates targeted search queries for finding relevant guidelines.
3. **Guideline Evidence Finder** (`GuidelineEvidenceFinder`) — Uses semantic search (PubMedBERT embeddings + FAISS vector index) to find the most relevant NICE guideline passages for each query.
4. **Compliance Auditor Agent** (`ComplianceAuditorAgent`) — Compares documented clinical decisions against retrieved guidelines using an LLM, producing per-diagnosis adherence scores on a 5-level scale (-2 to +2) with confidence scores and NICE guideline citations, plus a normalised aggregate score (0.0 to 1.0).

### The Data

- **Input:** ~4,327 anonymised MSK patients (21,530 clinical event rows) from the CrossCover clinical trial, coded in SNOMED CT
- **Knowledge Base:** 1,656 NICE guideline documents, embedded as vectors in a FAISS index
- **Validation:** 120 cases manually audited by expert clinicians (gold standard for measuring system accuracy)

### End Goal

A system that can:
- Process the full patient dataset and produce audit scores for every patient
- Be validated against the 120 gold-standard human audits
- Generate aggregate reports showing guideline adherence patterns across the dataset
- Be extended to other clinical domains beyond MSK

### Origin

This project builds upon the **ClinAuditAI** framework (Shahriyear, 2024) and two MSc dissertations from Keele University:
- **Hiruni Vidanapathirana** — built the Extractor + Query agents
- **Cyprian Toroitich** — built the Retriever + Scorer agents

We are **not copying** their work. We are analysing it, taking what's good, fixing what's bad, and rebuilding the entire system as a unified, production-grade pipeline.

---

## 2. Reference Codebases Analysis

### 2A. ClinAuditAI Paper (Shahriyear, 2024)

**What it is:** The foundational IEEE paper that defines the 4-agent architecture.

**What it did well:**
- Clean conceptual architecture — the 4-agent split is logical and well-motivated
- Used Llama-3 70B (strong open-source model)
- Tested across 8 medical specialties with scored results
- Clear scoring rubric (+1/-1 per diagnosis)
- Good use of RAG to ground LLM judgments in real guidelines

**What it did poorly / limitations:**
- Only tested on synthetic/example medical notes (not real patient data)
- No validation against human auditor judgments
- Limited to 300-1000 word free-text notes — our data is structured SNOMED codes, not free text
- Scoring is binary (+1/-1) with no nuance (partial adherence not captured)
- No error handling or production considerations discussed

**What we're taking:**
- The 4-agent architecture (Extractor → Query → Retriever → Scorer)
- The RAG approach for grounding scores in real guidelines
- PubMedBERT for medical embeddings
- FAISS for vector search
- The basic scoring concept (per-diagnosis evaluation)

**What we're improving:**
- Adapting for structured SNOMED data (not free-text notes)
- Adding nuanced scoring (not just binary +1/-1)
- Validation against gold-standard human audits
- Production-grade error handling, logging, configurability
- Unified LLM with provider abstraction

---

### 2B. Hiruni's Implementation (Extractor + Query Agent)

**Files:** `Hiruni/extractor.py`, `Hiruni/query_agent.py`, `Hiruni/pipeline.py`, `Hiruni/snomed/`

**What she built:**
- `ExtractorAgent` class that iterates through clinical entries and categorises each via FHIR/SNOMED lookup
- `HadesFHIRClient` that queries a local FHIR server to get semantic tags (disorder, procedure, finding, etc.)
- `ClinicalNote` and `ClinicalEntry` data models
- `QueryAgent` class that uses a local Mistral-7B (via llama_cpp) to generate guideline search queries
- LangGraph pipeline wiring Extractor → Query Agent
- Data loading and cleaning utilities (`build_note`, `safe_str`)

**What she did well:**
- SNOMED CT integration via FHIR is the correct approach for standardised medical coding
- Clean data model separation (ClinicalNote/ClinicalEntry)
- Semantic tag extraction from FSN (Fully Specified Name) is clever — parses "(disorder)", "(procedure)", etc.
- Proper date handling with fallbacks

**What she did poorly:**
- Hardcoded Windows paths (`C:\Users\hirun\agentic-msk\...`) everywhere
- Requires a local FHIR server (HADES) that isn't included or documented for setup
- Mistral-7B via llama_cpp is underpowered for medical query generation
- No error handling on FHIR lookups (network failures crash the pipeline)
- No logging whatsoever
- No tests
- `build_note` mixes data transformation with I/O concerns
- The LangGraph state management is minimal — no retry, no error nodes
- Hardcoded model path
- Date parsing is fragile (assumes specific formats)

**What we're taking:**
- The concept of SNOMED semantic tag extraction for categorisation
- The ClinicalNote/ClinicalEntry data model pattern (redesigned)
- The idea of FHIR-based lookups (but we need an alternative since HADES isn't available)

**What we're replacing:**
- FHIR server dependency → rule-based regex categoriser + LLM fallback (84%/16% split, no external server)
- Mistral-7B for query generation → three-tier approach: hand-crafted templates for common MSK conditions + LLM (via provider abstraction) for rare diagnoses + default fallback. Templates produce queries optimised for PubMedBERT similarity.
- Hardcoded paths → environment variables and config
- Raw LangGraph → our own clean pipeline orchestration

---

### 2C. Cyprian's Implementation (Retriever + Scorer Agent)

**Files:** `Cyprian/scorer_deployed.ipynb`, `Cyprian/guidelines.csv`, `Cyprian/guidelines.index`

**What he built:**
- FAISS vector index over 1,656 NICE guideline documents
- PubMedBERT Matryoshka embeddings for encoding guidelines and queries
- Retriever Agent that searches the FAISS index and returns top-5 guideline chunks
- Scorer Agent that uses GPT-3.5-turbo to evaluate adherence per diagnosis
- Flask JSON-RPC servers for inter-agent communication (ports 5000/5001)
- LangGraph workflow wiring Retriever → Scorer
- 5 test cases with expected scores

**What he did well:**
- PubMedBERT is the right embedding model for medical domain — much better than general-purpose embeddings
- FAISS with cosine similarity is efficient and appropriate
- The scoring prompt is well-structured (diagnosis + treatments + guidelines → score + explanation)
- The JSON-RPC A2A pattern shows understanding of microservice communication
- Pre-built FAISS indices are included (ready to use)

**What he did poorly:**
- Google Colab notebook — not reproducible, not deployable
- Uses GPT-3.5-turbo (weakest GPT model for complex medical reasoning)
- Flask servers running in threads — fragile, not production-ready
- Global mutable state (`scorer_state` dict) — race conditions, not thread-safe
- Hardcoded Colab paths (`/content/...`)
- API key stored via `google.colab.userdata` — not portable
- No error handling on OpenAI API calls
- No logging
- No tests (the 5 "test cases" are manual integration tests, not automated)
- Scoring prompt truncates guidelines to 500 chars — loses critical context
- Only passes treatments to the scorer prompt — ignores referrals, investigations, and procedures
- The `expected_score` for test case 2 has a syntax error (missing value)
- Regex for score parsing is case-sensitive but searches lowercase content — critical bug that causes all scores to default to -1
- JSON-RPC is over-engineered for a single-process pipeline

**What we're taking:**
- FAISS index and PubMedBERT embedding approach (proven, appropriate)
- The pre-built `guidelines.index` and `guidelines.csv` (valuable assets)
- The scoring prompt structure (diagnosis + treatment + guidelines → evaluation)
- Cosine similarity for retrieval

**What we're replacing:**
- Colab notebook → proper Python modules
- GPT-3.5-turbo → GPT-4o-mini or better (via abstraction layer)
- Flask JSON-RPC → direct function calls within a unified pipeline
- Global mutable state → proper state management (singleton Embedder + VectorStore with load/unload)
- Truncated guidelines (500 chars, 1 guideline) → intelligent formatting with up to 2,000 chars and all top-K guidelines
- Only treatments in scorer → full clinical context (treatments + referrals + investigations + procedures)
- Case-sensitive regex bug → case-insensitive parsing with robust edge case handling
- Manual test cases → automated test suite (32 scorer tests)
- Naive "guidelines for concept_name" queries → expert-crafted templates + LLM queries from Query Agent
- No deduplication → merge + dedup results across multiple queries per diagnosis
- `faiss.normalize_L2` on non-writable tensors → numpy normalization (fixed segfault bug)

---

### 2D. Shared Data Assets

| Asset | Location | Status | Notes |
|-------|----------|--------|-------|
| Raw patient data | `Original Data/Data_extract_30062025.txt` | Available | 409K rows, 17K patients, tab-separated |
| SQL extraction script | `Original Data/Data_extraction_20062025.sql` | Available | Documents exact data lineage |
| Cleaned patient data | `Cleaned Data/msk_valid_notes.csv` | Available | 21.5K rows, 4.3K patients, CSV |
| NICE guidelines | `Cyprian/guidelines.csv` | Available | 1,656 documents with clean text |
| FAISS index | `Cyprian/guidelines.index` | Available | Pre-built, 4.9MB, 768-dim vectors |
| Guidelines JSONL | `Cyprian/open_guidelines.jsonl` | Available | Raw guidelines in JSONL format |

**In our repo (`data/` directory, tracked in git):**

| Asset | Size | Notes |
|-------|------|-------|
| `data/msk_valid_notes.csv` | 2.5 MB | Cleaned patient data (4,327 patients, 21,530 entries) |
| `data/guidelines.csv.gz` | 24 MB | Compressed NICE guidelines (1,656 documents) |
| `data/guidelines.index` | 4.9 MB | FAISS index (768-dim PubMedBERT vectors) — rebuilt via `scripts/build_index.py` |

---

## 3. Architecture & Tech Stack

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ClinAuditAI                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐ │
│  │Extractor │───>│  Query   │───>│ Retriever │───>│  Scorer  │ │
│  │  Agent   │    │  Agent   │    │   Agent   │    │  Agent   │ │
│  └──────────┘    └──────────┘    └───────────┘    └──────────┘ │
│       │                               │                │       │
│       v                               v                v       │
│  ┌──────────┐                   ┌───────────┐    ┌──────────┐  │
│  │  SNOMED  │                   │   FAISS   │    │   LLM    │  │
│  │  Lookup  │                   │   Index   │    │ Provider │  │
│  └──────────┘                   └───────────┘    └──────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI REST API  │  PostgreSQL  │  Docker  │  Logging/Config  │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Language** | Python 3.11+ | All reference code is Python; strongest AI/ML ecosystem |
| **Web Framework** | FastAPI | Async support, auto-generated OpenAPI docs, Pydantic validation, type-safe. Flask (used by Cyprian) is too minimal. |
| **Database** | PostgreSQL | Production-grade, stores audit results/patient data/job tracking. SQLite is not suitable for concurrent access. |
| **ORM** | SQLAlchemy 2.0 + Alembic | Industry standard, async support, migration management |
| **Vector Search** | FAISS | Already proven in reference code, fast, no external service needed at this scale |
| **Embeddings** | PubMedBERT Matryoshka | Domain-specific medical embeddings, proven effective in Cyprian's work |
| **LLM (default)** | OpenAI GPT-4o-mini | Good balance of cost/quality for medical reasoning. Abstraction layer allows swapping. |
| **LLM Abstraction** | Custom provider pattern | Strategy pattern — swap providers via env var, zero code changes |
| **Pipeline Orchestration** | Custom pipeline (not LangGraph) | LangGraph adds complexity without proportional benefit for a linear 4-step pipeline. Simple, testable functions are better. |
| **Medical Coding** | Rule-based regex + LLM fallback | Two-tier SNOMED categoriser: regex patterns handle 84% of concepts, LLM classifies the remaining 16% in batches of 50. Categories persisted to DB — classified once, never repeated. No FHIR server needed. |
| **Query Generation** | Templates + LLM + defaults | Three-tier: hand-crafted templates for ~15 common MSK conditions, LLM for rare diagnoses, generic defaults as fallback. Templates optimised for PubMedBERT similarity. |
| **Guideline Scoring** | LLM with structured prompt | 5-level per-diagnosis scoring via LLM (temperature=0): +2 COMPLIANT, +1 PARTIAL, 0 NOT RELEVANT, -1 NON-COMPLIANT, -2 RISKY. Each score includes confidence (0.0-1.0) and cited NICE guideline text. Benefit-of-the-doubt for sparse coded data. Aggregate = `mean((score + 2) / 4)` normalised to [0.0, 1.0], errors excluded. Backward-compatible with legacy binary (+1/-1) results. Full details in `docs/learning/08-scorer-agent-explained.md`. |
| **Configuration** | Pydantic Settings | Type-safe, validates on startup, reads from .env files |
| **Logging** | Python `logging` + `structlog` | Structured JSON logs, correlation IDs, proper levels |
| **Containerisation** | Docker + Docker Compose | Reproducible environments, one-command setup |
| **Testing** | pytest + pytest-asyncio + aiosqlite | Standard Python testing, async support, in-memory SQLite for query tests |
| **Task Runner** | Makefile | Simple, universal, documents common commands |

### Why NOT LangGraph?

The reference implementations use LangGraph for pipeline orchestration. We're **not** using it because:
1. Our pipeline is **linear** (A → B → C → D) — LangGraph's graph capabilities are overkill
2. LangGraph adds a **heavy dependency** with its own state management conventions
3. Simple function composition is **easier to test**, debug, and understand
4. LangGraph's state typing is awkward for complex nested data
5. We lose nothing by using plain Python — and gain simplicity and full control

Instead, we'll build a `Pipeline` class that chains agent functions together with proper error handling, logging, and state passing.

---

## 4. Master Roadmap

### Phase 0: Foundation & Scaffolding ✅ COMPLETE
- [x] Create PROJECT_BIBLE.md
- [x] Set up project directory structure
- [x] Set up configuration system (Pydantic Settings, .env)
- [x] Set up logging infrastructure
- [x] Set up AI/LLM abstraction layer (base + OpenAI provider)
- [x] Set up Docker + Docker Compose (app + PostgreSQL)
- [x] Create health check endpoint
- [x] Verify app starts and endpoints work (9/9 tests passing)
- [x] Create initial learning docs
- [x] Update PROJECT_BIBLE.md
- [x] Push to GitHub (github.com/anasraza57/clinaudit-ai)

### Phase 1: Data Layer ✅ COMPLETE
- [x] Set up database connection (SQLAlchemy async engine + session)
- [x] Set up Alembic for migrations
- [x] Design database schema (patients, clinical_entries, audit_results, guidelines, jobs)
- [x] Create SQLAlchemy models (Patient, ClinicalEntry, AuditJob, AuditResult, Guideline)
- [x] Create initial migration (001_initial_schema.py)
- [x] Build data import pipeline (CSV → database) — `src/services/data_import.py`
- [x] Build FAISS index management (load, query, unload) — `src/services/vector_store.py`
- [x] API endpoints for data import — `/api/v1/data/import/patients`, `/api/v1/data/import/guidelines`
- [x] App startup hooks — auto-connect DB + auto-load FAISS index
- [x] Write tests for data layer — 34/34 tests passing
- [x] Update learning docs — `03-data-layer-explained.md`
- [x] Update PROJECT_BIBLE.md
- **Note:** Actual data import (running against live DB) deferred to when Docker is started

### Phase 2: Extractor Agent ✅ COMPLETE
- [x] Design SNOMED concept categorisation — two-tier: rule-based (84%) + LLM fallback (16%)
- [x] Build SNOMED Categoriser service — `src/services/snomed_categoriser.py`
- [x] Build Extractor Agent — `src/agents/extractor.py`
- [x] Categorise entries into: diagnosis, treatment, procedure, referral, investigation, administrative, other
- [x] Output structured ExtractionResult with episodes grouped by index_date
- [x] Write tests — 82/82 passing (categoriser: 40 parametrised + 3, extractor: 9)
- [x] Update learning docs — `05-extractor-agent-explained.md`
- [x] Update PROJECT_BIBLE.md

### Phase 3: Query Agent ✅ COMPLETE
- [x] Design query generation — three-tier: template queries for common MSK diagnoses, LLM for unusual diagnoses, default fallback
- [x] Build Query Agent — `src/agents/query.py` with template matching + LLM generation + defaults
- [x] Hand-craft query templates for ~15 common MSK conditions optimised for PubMedBERT/FAISS retrieval
- [x] Generate 1-3 targeted search queries per diagnosis
- [x] Write tests — 117/117 passing (35 new Query Agent tests)
- [x] Update learning docs — `06-query-agent-explained.md`
- [x] Update PROJECT_BIBLE.md

### Phase 4: Retriever Agent ✅ COMPLETE
- [x] Build PubMedBERT embedding service — `src/services/embedder.py` (singleton, load/encode/unload)
- [x] Build Retriever Agent — `src/agents/retriever.py` (embed queries, search FAISS, merge, dedup, rank)
- [x] Multi-query aggregation with deduplication (same guideline from multiple queries kept once with best score)
- [x] Fix faiss.normalize_L2 segfault — replaced with numpy normalization
- [x] Write tests — 144/144 passing (13 embedder + 14 retriever, using tiny BERT model for speed)
- [x] Update learning docs — `07-retriever-agent-explained.md`
- [x] Update PROJECT_BIBLE.md

### Phase 5: Scorer Agent ✅ COMPLETE
- [x] Design scoring prompt and rubric — structured prompt with diagnosis, treatments, referrals, investigations, procedures, guidelines
- [x] Build Scorer Agent — `src/agents/scorer.py` using LLM abstraction, temperature=0 for deterministic scoring
- [x] Implement per-diagnosis scoring — +1 (adherent) / -1 (non-adherent) with explanations, guidelines followed/not followed
- [x] Implement aggregate score calculation — proportion of adherent diagnoses (errors excluded)
- [x] Implement response parsing — case-insensitive regex with robust edge case handling
- [x] Implement guideline formatting — intelligent truncation with rank ordering, configurable max chars
- [x] Write tests — 176/176 passing (32 new Scorer tests)
- [x] Update learning docs — `08-scorer-agent-explained.md`
- [x] Update PROJECT_BIBLE.md

### Phase 6: Pipeline Integration ✅ COMPLETE
- [x] Build Pipeline orchestrator — `src/services/pipeline.py` (chains all 4 agents with DB I/O)
- [x] Implement single-patient audit endpoint — `POST /api/v1/audit/patient/{pat_id}`
- [x] Implement batch audit endpoint — `POST /api/v1/audit/batch` (background processing)
- [x] Implement job tracking — `GET /api/v1/audit/jobs/{job_id}` (progress polling)
- [x] Implement result retrieval — `GET /api/v1/audit/jobs/{job_id}/results` (paginated) and `GET /api/v1/audit/results/{pat_id}` (per-patient)
- [x] Error handling — per-patient error capture, early exits, continues on failure
- [x] SNOMED category pre-loading — load once, cache across all patients
- [x] Write tests — 190/190 passing (14 new pipeline tests)
- [x] Update learning docs — `09-pipeline-integration-explained.md`
- [x] Update PROJECT_BIBLE.md

### Phase 7a: Reporting Endpoints ✅ COMPLETE
- [x] Build reporting service — `src/services/reporting.py` (dashboard stats, condition breakdown, non-adherent cases, score distribution)
- [x] Build report API endpoints — `src/api/routes/reports.py` (4 GET endpoints under `/api/v1/reports/`)
- [x] Register reports router in `src/main.py`
- [x] Write tests — 216/216 passing (26 new reporting tests using in-memory SQLite via aiosqlite)
- [x] Update learning docs — `docs/learning/10-reporting-explained.md`
- [x] Update PROJECT_BIBLE.md

### Post-Phase 7a: Crash Fixes & Resilience ✅ COMPLETE
- ✅ Fixed `flush()` → `commit()` in batch handler — job status was invisible to polling clients due to PostgreSQL transaction isolation (2026-03-02)
- ✅ Per-patient session isolation — fresh DB session per patient prevents SQLAlchemy identity map memory growth (2026-03-02)
- ✅ Batched SNOMED LLM categorisation — 322 individual API calls → 7 batched calls (50 concepts per prompt, JSON format) (2026-03-02)
- ✅ Category persistence to DB — categories written to `clinical_entries.category` column, never re-classified (2026-03-02)
- ✅ OpenAI client timeout (60s) + retries (max_retries=2) — prevents indefinite LLM hangs (2026-03-02)
- ✅ Per-patient pipeline timeout (300s via asyncio.wait_for) — one slow patient can't stall the batch (2026-03-02)
- ✅ `_save_patient_error_and_progress()` helper — stores failed AuditResult + updates job progress in clean session after timeout/error (2026-03-02)
- ✅ `_recover_stale_jobs()` on startup — marks stuck jobs as "failed" when server restarts (2026-03-02)
- ✅ `gc.collect()` every 10 patients + after batch completion — forces garbage collection (2026-03-02)
- ✅ Pre-loaded PubMedBERT at startup — prevents crash from lazy loading during HTTP requests (2026-03-02)
- ✅ Embedder tensor cleanup — `.detach()` before `.numpy()`, explicit `del outputs, inputs` after encoding, `np.ascontiguousarray()` output for FAISS compatibility (2026-03-02)
- ✅ Retriever batch encoding — switched from individual `encode()` per query to `encode_batch()` per diagnosis. 6 forward passes instead of 18 for a 6-diagnosis patient (2026-03-02)
- ✅ FAISS contiguous array enforcement — `np.ascontiguousarray()` in vector store search to prevent memory alignment crashes (2026-03-02)
- ✅ Eliminated 'other' SNOMED category — removed from valid set, LLM prompts now force 6 real categories, all fallbacks default to 'administrative', added 10 new rule patterns, fixed 12 miscategorised entries in DB (2026-03-02)
- ✅ Fixed PubMedBERT segfault — `TOKENIZERS_PARALLELISM=false` disables HuggingFace Rust threads that conflict with uvicorn async event loop on macOS; `OMP_NUM_THREADS=1` prevents PyTorch internal threading conflicts (2026-03-02)
- ✅ Added `faulthandler.enable()` — prints Python traceback on segfaults instead of silent crashes (2026-03-02)
- ✅ Vector store load check in endpoints — both single and batch audit endpoints now verify embedder AND vector store are loaded before processing (2026-03-02)
- ✅ Auto-decompress `guidelines.csv.gz` — vector store auto-decompresses the .gz file on first load if uncompressed CSV is missing; only runs once (2026-03-02)
- ✅ Added `scripts/build_index.py` — rebuilds FAISS index from guidelines.csv (2026-03-02)
- ✅ Improved Swagger docs — response_model, summary, Field descriptions on all endpoints (2026-03-02)
- ✅ Added `GET /api/v1/data/stats`, `?limit=N` for batch, `GET /audit/jobs/{job_id}/results` pagination (2026-03-02)
- ✅ Learning docs updated: `05-extractor-agent-explained.md`, `09-pipeline-integration-explained.md` (2026-03-02)
- ✅ Diagnosis deduplication across pipeline — two layers: (1) **entry-level dedup** skips duplicate (term, index_date) pairs so each diagnosis appears exactly once per episode in pipeline output, eliminating duplicate entries in reports; (2) **term-level caching** reuses queries/embeddings/FAISS results across episodes for the same diagnosis term, avoiding redundant LLM/encoding work. All three agents (Query, Retriever, Scorer) implement both layers. 221 tests passing (+5 dedup tests) (2026-03-02)

- ✅ Fixed scoring prompt for sparse SNOMED data — prompt was too strict for coded records where many GP actions (verbal advice, prescriptions from separate systems) aren't captured. Referrals to physio/specialist now correctly score +1. Added context about coded data limitations. (2026-03-03)
- ✅ Fixed score regex parser for bracket format — LLM outputs `Score: [+1]` copying the bracket format from the prompt template. Regex `Score:\s*([+-]?1)` failed to match `[+1]`, defaulting ALL scores to -1. Fixed regex to `Score:\s*\[?([+-]?1)\]?` and removed brackets from prompt template. Added example output to prompt. 222 tests passing (+1 bracket parsing test). (2026-03-03)
- ✅ Batch re-audit verified — 10 patients: mean adherence 0.45, range 0.0–1.0. Patients with referrals score +1, patients with no actions score -1. (2026-03-03)

### Phase 7b: Gold-Standard Validation
- [ ] Import gold-standard audit data (120 cases)
- [ ] Run system against gold-standard cases
- [ ] Compare AI scores vs human auditor scores
- [ ] Generate accuracy/agreement metrics (Cohen's kappa, etc.)
- [ ] Add `get_validation_metrics()` to reporting service + new endpoint
- [ ] Write tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 9: Evaluation Framework & Model Comparison
- [x] **9a. Model comparison service** — compare two batch jobs (e.g. OpenAI vs Ollama) with Cohen's kappa, Pearson correlation, per-condition adherence deltas
- [x] **9b. Missing care opportunities** — scorer detects NICE-recommended actions not documented; new `missing_care_opportunities` field + reporting aggregation
- [ ] **9c. Gold-standard metrics** — confusion matrix, per-class P/R/F1, weighted F1, Cohen's kappa for comparing AI scores vs 120 clinician labels (data pending)
- [x] **9d. LLM-as-Judge evaluation** — evaluate each agent's output quality using a separate LLM call as judge (no human labels required)
- [x] **9e. Visualizations/charts** — inline SVG charts in HTML report (score distribution bar chart, compliance donut, per-condition bars)

### Phase 10a: Retriever Relevance Filtering
- [x] **Post-retrieval relevance filter** — two-layer filter (title keyword exclusion + L2 distance threshold) to remove irrelevant guidelines before they reach the scorer
- [x] **Configurable similarity threshold** — `retriever_min_similarity` setting (default 1.2 L2 distance)
- [ ] **Re-run batches** — validate improvement by running OpenAI + Ollama batches and comparing agreement rates

### Phase 10b: Comprehensive Evaluation Endpoints
- [x] **System-level metrics** — `GET /evaluation/system-metrics` — score class distribution (+2 to -2), adherence rate, confidence stats (mean/median/min/max/std), per-class counts, error rate
- [x] **Cross-model classification** — `GET /evaluation/cross-model-metrics` — 5×5 confusion matrix, per-class P/R/F1, 5-class & 3-class Cohen's kappa, exact-match accuracy, AUROC (trapezoidal rule), agreement rate, Pearson correlation
- [x] **Extractor metrics from DB** — `GET /evaluation/extractor-metrics` — evaluates SNOMED categorisation quality using rules as ground truth, per-category P/R/F1, no LLM needed
- [x] **Full agent evaluation** — `POST /evaluation/evaluate/agents` — runs pipeline on sample patients, evaluates all 4 agents including retriever IR metrics (Precision@k, nDCG, MRR)
- [x] **AUROC implementation** — trapezoidal rule (~30 lines), no sklearn dependency, binarized adherent vs non-adherent using confidence scores
- [x] **Retriever IR metrics** — `evaluate_retrieval_ir()` with per-guideline LLM-as-Judge ratings, Precision@k, nDCG@k, MRR
- [x] **Comparison chart SVGs** — confusion matrix heatmap, grouped bar chart (score distribution), paired donut charts (compliance)

### Phase 10c: Comparison HTML Report & Cross-Model Judging
- [x] **Comparison HTML report** — `GET /reports/export/comparison-html` generates self-contained HTML with all evaluation data: system metrics, charts, confusion matrix, P/R/F1, extractor quality, missing care, per-patient comparison
- [x] **LLM-as-Judge scorer eval section** — scorer quality table with both judges (GPT-4o-mini + mistral-small) × both models
- [x] **Agent-level evaluation section** — query relevance/coverage, retriever IR (P@k, nDCG, MRR), scorer quality from pipeline run
- [x] **Cross-model judging** — each model's output judged by both LLMs for cross-validation (eliminates self-judging bias)
- [x] **Fixed `run_agent_evaluation` bug** — was missing `embedder` and `vector_store` args to `AuditPipeline`
- [x] **All evaluation results generated** — scorer eval (4 combinations), agent eval (2 judges), saved to `exports/supervisor-report/`

### Phase 10: Polish & Documentation
- [ ] Performance optimisation (concurrent API calls, further caching — batch embeddings already done in retriever)
- [ ] Complete all learning documentation
- [ ] Complete README with full setup/run/test/deploy instructions
- [ ] Security review
- [ ] Final PROJECT_BIBLE.md update

---

## 5. Progress Tracker

### Phase 0: Foundation & Scaffolding ✅ COMPLETE
- ✅ Create PROJECT_BIBLE.md (2026-03-01)
- ✅ Set up project directory structure (2026-03-01)
- ✅ Set up configuration system — Pydantic Settings with .env (2026-03-01)
- ✅ Set up logging infrastructure — structured logging with dev/prod modes (2026-03-01)
- ✅ Set up AI/LLM abstraction layer — base + OpenAI provider + factory (2026-03-01)
- ✅ Set up Docker + Docker Compose — app + PostgreSQL (2026-03-01)
- ✅ Create health check endpoint — /health and /health/ready (2026-03-01)
- ✅ Verify app starts and endpoints work (2026-03-01) — tested locally, 9/9 tests pass
- ✅ Create initial learning docs — glossary, project overview, architecture (2026-03-01)
- ✅ Copy reference data files into data/ directory (2026-03-01)
- ✅ Create .env.example, .gitignore, README.md, Makefile (2026-03-01)
- ✅ Push to GitHub — github.com/anasraza57/clinaudit-ai (2026-03-01)
- ✅ Final PROJECT_BIBLE.md update for Phase 0 (2026-03-01)

### Phase 1: Data Layer ✅ COMPLETE
- ✅ Database connection — async SQLAlchemy engine + session management (2026-03-01)
- ✅ Alembic configured — `migrations/env.py` reads DB URL from Settings, uses our models' metadata (2026-03-01)
- ✅ Database schema — 5 tables: patients, clinical_entries, guidelines, audit_jobs, audit_results (2026-03-01)
- ✅ SQLAlchemy models — Patient, ClinicalEntry, AuditJob, AuditResult, Guideline with TimestampMixin (2026-03-01)
- ✅ Initial migration — `001_initial_schema.py` with full schema + indexes + foreign keys (2026-03-01)
- ✅ Data import service — `src/services/data_import.py` — idempotent CSV→DB import for patients and guidelines (2026-03-01)
- ✅ Vector store service — `src/services/vector_store.py` — FAISS index load/search/unload with singleton pattern (2026-03-01)
- ✅ API endpoints — POST `/api/v1/data/import/patients` and `/api/v1/data/import/guidelines` (2026-03-01)
- ✅ App startup — auto-init DB connection + auto-load FAISS index (graceful warnings if unavailable) (2026-03-01)
- ✅ Tests — 34/34 passing (models, vector store, data import, health, config, AI base) (2026-03-01)
- ✅ Learning doc — `docs/learning/03-data-layer-explained.md` (2026-03-01)
- ✅ Fixed torch version in requirements.txt (2.5.1 → 2.2.2 for Python 3.11 compat) (2026-03-01)

### Phase 2: Extractor Agent ✅ COMPLETE
- ✅ SNOMED Categoriser — rule-based keyword matching (84% coverage of 1,261 unique concepts) + LLM fallback (2026-03-01)
- ✅ Extractor Agent — groups entries by index_date, categorises each, outputs structured ExtractionResult (2026-03-01)
- ✅ Categories: diagnosis (463), referral (194), administrative (170), investigation (91), treatment (66), procedure (38) + 192 for LLM (2026-03-01)
- ✅ Tests — 82/82 passing (2026-03-01)
- ✅ Learning doc — `docs/learning/05-extractor-agent-explained.md` (2026-03-01)

### Phase 3: Query Agent ✅ COMPLETE
- ✅ Query Agent — three-tier query generation: templates for common MSK, LLM for rare, defaults as fallback (2026-03-01)
- ✅ Template queries — hand-crafted for ~15 common MSK conditions (low back pain, osteoarthritis, carpal tunnel, gout, etc.) (2026-03-01)
- ✅ LLM generation — prompt includes episode context (treatments, referrals, investigations) for targeted queries (2026-03-01)
- ✅ Data classes — DiagnosisQueries, QueryResult with summary() and all_queries() helpers (2026-03-01)
- ✅ Tests — 117/117 passing (35 new: template matching, default queries, agent with/without LLM, mock LLM, dataclasses) (2026-03-01)
- ✅ Learning doc — `docs/learning/06-query-agent-explained.md` (2026-03-01)

### Phase 4: Retriever Agent ✅ COMPLETE
- ✅ PubMedBERT Embedder service — loads model, encodes text to 768-dim vectors with mean pooling + L2 norm (2026-03-01)
- ✅ Retriever Agent — embeds queries, searches FAISS, merges/deduplicates across multiple queries per diagnosis (2026-03-01)
- ✅ Fixed faiss.normalize_L2 segfault — replaced with numpy normalization (torch tensors are non-writable) (2026-03-01)
- ✅ Data classes — GuidelineMatch, DiagnosisGuidelines (with guideline_texts/titles helpers), RetrievalResult (2026-03-01)
- ✅ Tests — 144/144 passing (13 embedder using bert-tiny for speed + 14 retriever with mocked embedder/store) (2026-03-01)
- ✅ Learning doc — `docs/learning/07-retriever-agent-explained.md` (2026-03-01)

### Phase 5: Scorer Agent ✅ COMPLETE
- ✅ Scorer Agent — `src/agents/scorer.py` with structured scoring prompt, per-diagnosis evaluation, aggregate calculation (2026-03-01)
- ✅ Scoring prompt — includes diagnosis, treatments, referrals, investigations, procedures, and full guideline text (2,000 chars vs Cyprian's 500) (2026-03-01)
- ✅ Response parsing — case-insensitive regex extracting score, explanation, guidelines followed, guidelines not followed (2026-03-01)
- ✅ Guideline formatting — intelligent truncation with rank ordering, configurable max chars (2026-03-01)
- ✅ Error handling — per-diagnosis error capture, errors excluded from aggregate, pipeline continues on failure (2026-03-01)
- ✅ Data classes — DiagnosisScore (per-diagnosis), ScoringResult (aggregate with summary()) (2026-03-01)
- ✅ Tests — 176/176 passing (32 new: 8 parsing + 8 data classes + 12 agent + 4 formatting) (2026-03-01)
- ✅ Learning doc — `docs/learning/08-scorer-agent-explained.md` (2026-03-01)

### Phase 6: Pipeline Integration ✅ COMPLETE
- ✅ Pipeline orchestrator — `src/services/pipeline.py` chains all 4 agents, handles DB I/O, error recovery (2026-03-02)
- ✅ Audit API endpoints — `src/api/routes/audit.py` with 4 endpoints: single patient, batch, job status, results (2026-03-02)
- ✅ Single patient audit — `POST /api/v1/audit/patient/{pat_id}` runs pipeline synchronously, returns scoring result (2026-03-02)
- ✅ Batch audit — `POST /api/v1/audit/batch` runs in background with FastAPI BackgroundTasks, tracks via AuditJob (2026-03-02)
- ✅ Job tracking — `GET /api/v1/audit/jobs/{job_id}` returns progress (processed/total/failed) (2026-03-02)
- ✅ Result retrieval — `GET /api/v1/audit/jobs/{job_id}/results` (paginated batch results) and `GET /api/v1/audit/results/{pat_id}` (per-patient results) (2026-03-02)
- ✅ Error handling — per-patient error capture, early exits for missing data/no diagnoses, continues on failure (2026-03-02)
- ✅ SNOMED category loading — `load_categories_from_db()` loads cached categories from DB, classifies new ones (rules + batched LLM), writes back to DB, populates in-memory cache (2026-03-02)
- ✅ Results stored in AuditResult table — overall_score, counts, full JSON breakdown in details_json (2026-03-02)
- ✅ Router registered in main.py — `app.include_router(audit_router, prefix="/api/v1")` (2026-03-02)
- ✅ Tests — 190/190 passing (14 new: 5 PipelineResult + 9 AuditPipeline) (2026-03-02)
- ✅ Learning doc — `docs/learning/09-pipeline-integration-explained.md` (2026-03-02)

### Phase 7a: Reporting Endpoints ✅ COMPLETE
- ✅ Reporting service — `src/services/reporting.py` with 4 public functions + 1 private helper (2026-03-02)
- ✅ `get_dashboard_stats()` — total audited/failed, mean/median/min/max adherence score, failure rate (SQL columns only) (2026-03-02)
- ✅ `get_condition_breakdown()` — adherence rates grouped by diagnosis term, min_count filter, sort by count or adherence_rate (2026-03-02)
- ✅ `get_non_adherent_cases()` — paginated list of score=-1 diagnoses with explanations for clinical review (2026-03-02)
- ✅ `get_score_distribution()` — histogram of patient-level overall_score, configurable bins (2026-03-02)
- ✅ `_load_completed_results()` — shared query helper, optional Patient eager-loading via selectinload (2026-03-02)
- ✅ Report API — `src/api/routes/reports.py` with 4 GET endpoints, Pydantic response schemas (2026-03-02)
- ✅ Router registered in `src/main.py` — `app.include_router(reports_router, prefix="/api/v1")` (2026-03-02)
- ✅ Added `aiosqlite==0.20.0` to requirements.txt for async SQLite testing (2026-03-02)
- ✅ Tests — 216/216 passing (26 new: 4 _load_completed_results + 6 dashboard + 5 condition breakdown + 6 non-adherent + 5 score distribution) (2026-03-02)
- ✅ Learning doc — `docs/learning/10-reporting-explained.md` (2026-03-02)
- ✅ Export service — `src/services/export.py` with CSV and HTML report generation (2026-03-03)
- ✅ `GET /api/v1/reports/export/csv` — downloadable CSV file (one row per diagnosis per patient) (2026-03-03)
- ✅ `GET /api/v1/reports/export/html` — self-contained HTML report with dashboard stats, condition breakdown, per-patient detail cards (2026-03-03)
- ✅ Tests — 234/234 passing (+12 new export tests: 5 CSV + 7 HTML) (2026-03-03)

### Phase 8: Supervisor Feedback Implementation ✅ COMPLETE

**Agent Renaming (Phase 8a)** — MSK-specific agent names per supervisor feedback:
- ✅ `ExtractorAgent` → `ConsultationInsightAgent` in `src/agents/extractor.py` (2026-03-10)
- ✅ `QueryAgent` → `AuditQueryGenerator` in `src/agents/query.py` (2026-03-10)
- ✅ `RetrieverAgent` → `GuidelineEvidenceFinder` in `src/agents/retriever.py` (2026-03-10)
- ✅ `ScorerAgent` → `ComplianceAuditorAgent` in `src/agents/scorer.py` (2026-03-10)
- ✅ Updated all imports in `pipeline.py` and all test files (2026-03-10)
- ✅ No backward-compat aliases — clean rename (2026-03-10)

**5-Level Graded Scoring (Phase 8b)** — replaces binary +1/-1 with clinically nuanced scale:
- ✅ New `AuditJudgement` IntEnum: -2 RISKY, -1 NON-COMPLIANT, 0 NOT RELEVANT, +1 PARTIAL, +2 COMPLIANT (2026-03-10)
- ✅ `DiagnosisScore` now includes: `judgement`, `confidence` (0.0-1.0), `cited_guideline_text` (direct NICE quote) (2026-03-10)
- ✅ `ScoringResult` has 5 per-level counters + backward-compat `adherent_count`/`non_adherent_count` properties (2026-03-10)
- ✅ New aggregate score formula: `mean((score + 2) / 4)` maps [-2,+2] to [0.0,1.0] (2026-03-10)
- ✅ Completely rewritten `SCORING_PROMPT` with 5-level instructions, confidence, and citation requirements (2026-03-10)
- ✅ New regex parser handles all 7 fields: score, judgement, confidence, cited_guideline_text, explanation, followed, not_followed (2026-03-10)
- ✅ `reporting.py` detects old vs new format via `"judgement" in ds` — full backward compatibility (2026-03-10)
- ✅ `export.py` updated: CSV has new columns (judgement, confidence, cited_guideline_text), HTML has 5-level badges + cited guideline blockquotes (2026-03-10)
- ✅ API schemas updated: `ConditionBreakdownItem` has 5-level counts, `NonAdherentCase` has judgement/confidence/citation (2026-03-10)
- ✅ All tests updated for 5-level format (2026-03-10)

**Ollama Local LLM Provider (Phase 8c)** — for patient data processing without external APIs:
- ✅ `OllamaProvider` in `src/ai/ollama_provider.py` — uses OpenAI SDK with Ollama's compatible endpoint (2026-03-10)
- ✅ Settings: `ollama_base_url`, `ollama_model`, `ollama_max_tokens`, `ollama_temperature`, `ollama_request_timeout` (2026-03-10)
- ✅ Factory: `"ollama"` and `"local"` (alias) registered in `_PROVIDERS` dict (2026-03-10)
- ✅ `embed()` raises clear error — ClinAuditAI uses PubMedBERT for embeddings (2026-03-10)
- ✅ Tests — 256/256 passing (+22 new: 14 scorer + 8 ollama) (2026-03-10)

### Phase 9: Evaluation Framework 🔄 IN PROGRESS

**Model Comparison Service (Phase 9a) ✅ COMPLETE:**
- ✅ Added `provider` column to `AuditJob` model — tracks which AI provider (openai, ollama) ran each batch (2026-03-11)
- ✅ Alembic migration `002_add_job_provider.py` — adds `provider` column to `audit_jobs` table (2026-03-11)
- ✅ Pipeline auto-sets `provider` on job creation from `settings.ai_provider` (2026-03-11)
- ✅ `src/services/comparison.py` — full comparison service: loads two jobs, matches patients by pat_id, matches diagnoses by (term, index_date), computes per-patient score diffs, per-condition adherence deltas, Cohen's kappa (3-class binning), Pearson correlation (2026-03-11)
- ✅ `src/api/routes/evaluation.py` — `GET /api/v1/evaluation/compare?job_a=1&job_b=2` endpoint with full Pydantic schemas (2026-03-11)
- ✅ Evaluation router registered in `src/main.py` (2026-03-11)
- ✅ 21 tests in `tests/unit/test_comparison.py` — Cohen's kappa (6), Pearson (6), compare_jobs (9) (2026-03-11)

**Missing Care Opportunities (Phase 9b) ✅ COMPLETE:**
- ✅ Added `missing_care_opportunities: list[str]` field to `DiagnosisScore` dataclass (2026-03-11)
- ✅ Updated `SCORING_PROMPT` output format — LLM now outputs "Missing Care Opportunities:" line (2026-03-11)
- ✅ Added `_MISSING_CARE_PATTERN` regex + parsing in `parse_scoring_response()` (2026-03-11)
- ✅ Updated `ScoringResult.summary()` to include missing care in JSON output (2026-03-11)
- ✅ Updated `_NOT_FOLLOWED_PATTERN` regex to stop at `Missing Care Opportunities:` boundary (2026-03-11)
- ✅ `get_missing_care_summary()` in `src/services/reporting.py` — groups by condition, counts frequency per action (2026-03-11)
- ✅ `GET /api/v1/evaluation/missing-care` endpoint with `MissingCareResponse` schema (2026-03-11)
- ✅ CSV export: added `missing_care_opportunities` column (2026-03-11)
- ✅ HTML export: amber "Missing care" tag on diagnosis cards (2026-03-11)
- ✅ 12 tests in `tests/unit/test_missing_care.py` — parsing (5), data classes (3), reporting (4) (2026-03-11)
- ✅ All tests: 289/289 passing (2026-03-11)

**LLM-as-Judge Evaluation (Phase 9d) ✅ COMPLETE:**
- ✅ `src/services/evaluation.py` — full evaluation service with per-agent evaluation functions (2026-03-11)
- ✅ Extractor evaluation: weak supervision using SNOMED rules as pseudo-ground-truth — per-category P/R/F1, rule_match_rate (no LLM needed) (2026-03-11)
- ✅ Query evaluation: LLM-as-Judge rates query relevance (1-5) and coverage (1-5) per diagnosis (2026-03-11)
- ✅ Retriever evaluation: LLM-as-Judge rates retrieved guideline relevance (1-5) per diagnosis (2026-03-11)
- ✅ Scorer evaluation: LLM-as-Judge rates reasoning quality, citation accuracy, score calibration (all 1-5) (2026-03-11)
- ✅ `evaluate_patient()` orchestrator — evaluates all (or selected) agents for a single patient's pipeline result (2026-03-11)
- ✅ `aggregate_evaluations()` — aggregates per-patient metrics across multiple patients (2026-03-11)
- ✅ `scoring_from_stored()` — reconstructs `ScoringResult` from stored `details_json` for evaluation without re-running pipeline (2026-03-11)
- ✅ `POST /api/v1/evaluation/evaluate/scorer/{job_id}` endpoint — evaluates scorer from stored data via LLM judge (2026-03-11)
- ✅ 23 tests in `tests/unit/test_evaluation.py` — rating parsing (6), extractor (3), query (3), retriever (2), scorer (3), pipeline (2), aggregation (2), stored data (2) (2026-03-11)
- ✅ All tests: 312/312 passing (2026-03-11)

**Visualizations/Charts (Phase 9e) ✅ COMPLETE:**
- ✅ `_svg_score_distribution()` — bar chart histogram of patient scores in 5 bins (0-20%, 20-40%, 40-60%, 60-80%, 80-100%), colour-coded red→green (2026-03-11)
- ✅ `_svg_compliance_donut()` — donut chart for 5-level compliance breakdown (+2/+1/0/-1/-2) with legend, total count in centre (2026-03-11)
- ✅ `_svg_condition_bars()` — horizontal bar chart of per-condition adherence rates, colour-coded by threshold, auto-truncates long labels (2026-03-11)
- ✅ Charts integrated into HTML report via `_build_html()` — renders in a 2-column grid (score dist + donut), full-width condition bars below (2026-03-11)
- ✅ Chart CSS: `.chart-grid`, `.chart-card`, `.chart-full` classes; responsive grid layout (2026-03-11)
- ✅ Charts only render when data is present — empty reports have no chart section (2026-03-11)
- ✅ All inline SVG — no JavaScript, no external dependencies, print-friendly (2026-03-11)
- ✅ `level_counts` dict tracked during HTML generation for donut chart data (2026-03-11)
- ✅ 14 new tests in `tests/unit/test_export.py` — score distribution (4), compliance donut (4), condition bars (4), HTML integration (2) (2026-03-11)
- ✅ All tests: 326/326 passing (2026-03-11)

**PNG Chart Export ✅ COMPLETE:**
- ✅ Added `cairosvg>=2.7.0` to `requirements.txt` — converts SVG to PNG (requires system cairo library) (2026-03-11)
- ✅ Extracted `_collect_chart_data()` helper — shared DB query + parsing for both HTML report and PNG export (2026-03-11)
- ✅ `export_charts_to_png(session, output_dir, job_id, dpi)` — generates 3 PNG chart files from audit data (2026-03-11)
- ✅ `scripts/export_charts.py` — CLI script: `python scripts/export_charts.py --output exports/charts --job-id 1 --dpi 200` (2026-03-11)
- ✅ Graceful degradation if cairo not installed — `try/except OSError` at import, clear error message (2026-03-11)
- ✅ 6 new tests (2 chart data + 4 PNG export with mocked cairosvg) — all tests: 332/332 passing (2026-03-11)

**Supervisor Feedback — Remaining Items:**
- ⬜ Gold-standard metrics framework — confusion matrix, P/R/F1, kappa (Phase 9c — skipped, awaiting 120 clinician labels)

### Phase 10a: Retriever Relevance Filtering ✅ COMPLETE

**Context:** 50-patient batch comparison (OpenAI Job 1 vs Ollama Job 2) revealed 80% of inter-model disagreements caused by FAISS retriever returning irrelevant guidelines (e.g., carpal tunnel → "chest pain" guidelines, foot pain → "diabetic foot" guidelines). Cohen's kappa was 0.43 (moderate); agreement rate 65.9%.

- ✅ Added `retriever_min_similarity: float = 1.2` setting — configurable L2 distance threshold (2026-03-11)
- ✅ Added `_TOPIC_KEYWORDS` mapping — 13 body-region/condition keyword groups for topic matching (2026-03-11)
- ✅ Added `_EXCLUDE_TITLE_TERMS` set — 20+ terms to flag obviously irrelevant guidelines (cancer, cardiac, diabetes, pregnancy, etc.) (2026-03-11)
- ✅ Added `_filter_irrelevant()` method to `GuidelineEvidenceFinder` — two-layer post-retrieval filter: (A) title exclusion + topic overlap check, (B) L2 distance threshold (2026-03-11)
- ✅ Fallback: if all guidelines filtered, returns single best match to avoid empty context for scorer (2026-03-11)
- ✅ 18 new tests: 10 helper tests (topic extraction, title exclusion) + 8 integration tests (filtering logic, regression tests for known bad cases) (2026-03-11)
- ✅ All tests: 350/350 passing (2026-03-11)

### Phase 10b: Comprehensive Evaluation Endpoints ✅ COMPLETE

**Context:** Supervisor feedback (`feedback.docx`) requires demonstrating the system works with concrete metrics and visualizations across both models (OpenAI Job 1, Ollama Job 2). Previous endpoints covered basic comparison and LLM-as-Judge but lacked system-level classification metrics, confusion matrix, extractor evaluation from stored data, retriever IR metrics, confidence statistics, and comparison charts.

**New Service Functions:**
- ✅ `compute_system_metrics()` in `src/services/reporting.py` — per-job: score class distribution (+2 to -2), adherence rate, confidence stats (mean/median/min/max/std), per-class counts, error rate (2026-03-11)
- ✅ `compute_cross_model_classification()` in `src/services/comparison.py` — 5×5 confusion matrix, per-class P/R/F1, 5-class & 3-class Cohen's kappa, exact-match accuracy, AUROC, agreement rate, Pearson correlation (2026-03-11)
- ✅ `_compute_auroc()` in `src/services/comparison.py` — trapezoidal rule AUROC (~30 lines), no sklearn dependency, binarized adherent vs non-adherent using confidence scores (2026-03-11)
- ✅ `evaluate_extractor_from_db()` in `src/services/evaluation.py` — loads clinical_entries with stored category, compares vs `categorise_by_rules()` ground truth, per-category P/R/F1 (no LLM needed) (2026-03-11)
- ✅ `evaluate_retrieval_ir()` in `src/services/evaluation.py` — per-guideline LLM-as-Judge ratings, Precision@k, nDCG@k, MRR, mean relevance (2026-03-11)
- ✅ `run_agent_evaluation()` in `src/services/evaluation.py` — orchestrates full pipeline evaluation: picks random patients, runs pipeline, evaluates all 4 agents including retriever IR metrics (2026-03-11)

**New Comparison Chart SVGs:**
- ✅ `_svg_confusion_matrix()` in `src/services/export.py` — heatmap grid with green diagonal, blue off-diagonal, cell counts (2026-03-11)
- ✅ `_svg_comparison_scores()` in `src/services/export.py` — grouped bar chart, two bars per score class, blue/orange colours, legend (2026-03-11)
- ✅ `_svg_comparison_compliance()` in `src/services/export.py` — paired donut charts side-by-side with shared legend (2026-03-11)

**New API Endpoints:**
- ✅ `GET /api/v1/evaluation/system-metrics?job_id=N` — `SystemMetricsResponse` schema (2026-03-11)
- ✅ `GET /api/v1/evaluation/cross-model-metrics?job_a=N&job_b=M` — `CrossModelMetricsResponse` schema (2026-03-11)
- ✅ `GET /api/v1/evaluation/extractor-metrics?sample_size=N` — `ExtractorMetricsResponse` schema (2026-03-11)
- ✅ `POST /api/v1/evaluation/evaluate/agents?limit=5` — `AgentEvaluationResponse` schema (2026-03-11)

**Tests:**
- ✅ 4 new tests in `test_reporting.py` — `TestComputeSystemMetrics` (class distribution, adherence rate, confidence stats, empty job) (2026-03-11)
- ✅ 8 new tests in `test_comparison.py` — `TestComputeAuroc` (5) + `TestCrossModelClassification` (3) (2026-03-11)
- ✅ 5 new tests in `test_evaluation.py` — `TestExtractorFromDB` (2) + `TestRetrieverIR` (3) (2026-03-11)
- ✅ 4 new tests in `test_export.py` — `TestComparisonCharts` (4) (2026-03-11)
- ✅ All tests: 371/371 passing (2026-03-11)

### Phase 10c: Comparison HTML Report & Cross-Model Judging ✅ COMPLETE

**Context:** Supervisor needs a single shareable file with all evaluation results. LLM-as-Judge scorer evaluation and full agent evaluation results needed to be included alongside system metrics and cross-model comparison.

**Comparison HTML Report:**
- ✅ `generate_comparison_html()` in `src/services/export.py` — pulls together all evaluation data (system metrics, charts, confusion matrix, P/R/F1, extractor, missing care, per-patient) into one self-contained HTML file (2026-03-11)
- ✅ `_build_scorer_eval_section()` — renders LLM-as-Judge scorer quality table with 4 scorer×judge combinations (2026-03-11)
- ✅ `_build_agent_eval_section()` — renders agent-level evaluation results from both judges side-by-side (query, retriever IR, scorer) (2026-03-11)
- ✅ `GET /api/v1/reports/export/comparison-html` endpoint with optional `include_scorer_eval` parameter (2026-03-11)
- ✅ `_kappa_label()` helper — human-readable Cohen's kappa interpretation (2026-03-11)

**Scorer Evaluation — LLM-as-Judge (100 patients, 156 diagnoses, scale 1-5):**

| Metric | mistral-small + Ollama Judge | mistral-small + OpenAI Judge | gpt-4.1-mini + Ollama Judge | gpt-4.1-mini + OpenAI Judge |
|---|---|---|---|---|
| Reasoning Quality | 4.58 | 4.37 | 4.75 | 4.77 |
| Citation Accuracy | 3.81 | 3.22 | 4.56 | 4.46 |
| Score Calibration | 4.56 | 4.51 | 4.73 | 4.79 |

- ✅ 100 patients per model, same patients across models (deterministic `pat_id` ordering) (2026-03-16)
- ✅ Both judges (Ollama/mistral-small, OpenAI/gpt-4.1-mini) evaluated both models — 4 result files (2026-03-16)
- ✅ GPT-4.1-mini consistently outperforms, especially on citation accuracy (+0.75 to +1.24 over mistral-small) (2026-03-16)
- ✅ Judges broadly agree on relative rankings — validates judge reliability (2026-03-16)

**Full Agent Evaluation — Pipeline + LLM Judge (50 patients, 88 diagnoses):**

| Metric | mistral-small | gpt-4.1-mini |
|---|---|---|
| Extractor match rate | 1.00 | 1.00 |
| Query relevance (1-5) | 4.30 | 4.55 |
| Query coverage (1-5) | 3.47 | 3.63 |
| Retriever relevance (1-5) | 3.08 | 3.32 |
| Retriever precision@k | 0.383 | 0.568 |
| Retriever recall@k | 0.849 | 0.951 |
| Retriever nDCG | 0.648 | 0.821 |
| Retriever MRR | 0.595 | 0.793 |
| Scorer reasoning (1-5) | 4.61 | 4.78 |
| Scorer citation (1-5) | 3.86 | 4.32 |
| Scorer calibration (1-5) | 4.60 | 4.77 |

- ✅ 50 patients per model, same patients across models (deterministic `pat_id` ordering) (2026-03-16)
- ✅ GPT-4.1-mini outperforms across all metrics; biggest gap in retriever IR (precision@k: 0.57 vs 0.38, nDCG: 0.82 vs 0.65) and citation accuracy (4.32 vs 3.86) (2026-03-16)
- ✅ Extractor identical (rule-based, model-independent) (2026-03-16)

**Evaluation result files (in `data/eval_results/`):**
- `scorer_eval_mistral_small_ollama_judge_100.json` — 100 patients, 156 diagnoses
- `scorer_eval_mistral_small_openai_judge_100.json` — 100 patients, 156 diagnoses
- `scorer_eval_gpt4_mini_ollama_judge_100.json` — 100 patients, 156 diagnoses
- `scorer_eval_gpt4_mini_openai_judge_100.json` — 100 patients, 156 diagnoses
- `agents_eval_mistral_small_50.json` — 50 patients, 88 diagnoses
- `agents_eval_gpt4_mini_50.json` — 50 patients, 88 diagnoses

**Generated Reports (in `exports/supervisor-report/`):**
- ✅ `comparison-report-full.html` — 38KB self-contained report with all sections (2026-03-11)
- ✅ JSON files: system metrics (×2), cross-model metrics, comparison, extractor metrics, missing care (×2), scorer evals (×4), agent evals (×2), dashboards (×2)
- ✅ CSV and HTML per-job reports

- ✅ All tests: 371/371 passing (2026-03-11)

### Phase 11b: Deterministic Evaluation Ordering & Resumable Evaluation ✅ COMPLETE

**Context:** Evaluation endpoints (`evaluate/scorer` and `evaluate/agents`) previously selected random patients, making cross-model comparison unfair (different patients per model) and preventing resumable evaluation runs.

**Deterministic `pat_id` sorting + `offset` parameter:**
- ✅ `POST /api/v1/evaluation/evaluate/scorer` — changed from `POST /evaluate/scorer/{job_id}` (path param) to query-param-based endpoint (`?model=&limit=&offset=`); results sorted by deterministic `pat_id` ordering; duplicate scorer endpoint removed (2026-03-16)
- ✅ `POST /api/v1/evaluation/evaluate/agents` — added `offset` param, changed from random patient selection to deterministic `pat_id` ordering (2026-03-16)
- ✅ `GET /api/v1/audit/jobs/{job_id}/results` — now sorted by `pat_id` then `AuditResult.id` for stable pagination (2026-03-16)
- ✅ `_load_results_for_scorer()` in `src/api/routes/reports.py` — uses deterministic `pat_id` sorting + offset for consistent result loading (2026-03-16)

**Cross-model fair comparison:**
- ✅ Same `offset`/`limit` values produce the same set of patients regardless of model — e.g. `?model=gpt-4.1-mini&limit=50&offset=0` and `?model=mistral-small&limit=50&offset=0` evaluate the exact same 50 patients (2026-03-16)

**Resumable evaluation:**
- ✅ First call `?model=X&offset=0&limit=20` evaluates first 20 patients; second call `?model=X&offset=20&limit=10` evaluates next 10 without re-evaluating the first 20 (2026-03-16)

**Model-based filtering on all report/export endpoints:**
- ✅ All report endpoints (`dashboard`, `conditions`, `non-adherent`, `score-distribution`) support `?model=` query param alongside `?job_id=` (2026-03-16)
- ✅ All export endpoints (`csv`, `html`) support `?model=` query param (2026-03-16)
- ✅ `/export/comparison-html` supports `?model_a=`/`?model_b=` params as alternative to `?job_a=`/`?job_b=` (2026-03-16)

### Phase 11d: Comprehensive Individual Reports ✅ COMPLETE

**Context:** Individual HTML reports (`/api/v1/reports/export/html`) previously only showed per-patient cards and basic dashboard stats. All the rich evaluation sections (system metrics, extractor quality, LLM-as-Judge scorer quality, agent-level evaluation, missing care opportunities) were only available in the comparison report. This meant you couldn't get a full single-model evaluation report without generating a comparison.

**Individual HTML report now includes all evaluation sections:**
- ✅ System Metrics section — score class distribution, adherence rate, confidence stats (2026-03-16)
- ✅ Extractor Quality section — per-category P/R/F1 from SNOMED rules (2026-03-16)
- ✅ LLM-as-Judge Scorer Quality section — both judges' ratings when `use_saved_evals=true` (2026-03-16)
- ✅ Agent-Level Evaluation section — query, retriever IR, scorer metrics when `use_saved_evals=true` (2026-03-16)
- ✅ Aggregated Missing Care Opportunities section (2026-03-16)

**New section builder functions in `src/services/export.py`:**
- ✅ `_build_system_metrics_html()` — renders system metrics table (2026-03-16)
- ✅ `_build_extractor_html()` — renders per-category P/R/F1 table (2026-03-16)
- ✅ `_build_scorer_eval_single_html()` — renders LLM-as-Judge scorer quality for a single model (2026-03-16)
- ✅ `_build_agent_eval_single_html()` — renders agent-level evaluation for a single model (2026-03-16)
- ✅ `_build_missing_care_html()` — renders aggregated missing care opportunities (2026-03-16)

**Bug fixes:**
- ✅ Fixed truncated patient IDs in comparison report's per-patient table — now shows full UUID (2026-03-16)
- ✅ Fixed variable shadowing bug where `missing_care` local variable in patient card loop was overriding the parameter (2026-03-16)

**Endpoint update:**
- ✅ `GET /api/v1/reports/export/html` now accepts `use_saved_evals=true` query parameter to load pre-computed evaluation results from `data/eval_results/` (2026-03-16)

**Regenerated reports:**
- `exports/report-gpt4-mini.html` (3.1 MB)
- `exports/report-mistral-small.html` (2.6 MB)
- `exports/comparison-report-full.html` (3.4 MB)

**Files changed:**
- `src/services/export.py` — 5 new section builder functions, variable shadowing fix, truncated patient ID fix
- `src/api/routes/reports.py` — `use_saved_evals` param on `export_html` endpoint

---

## 6. Decisions Log

### Decision 001: FastAPI over Flask (2026-03-01)
**Context:** Need a web framework for the API layer.
**Choice:** FastAPI
**Alternatives rejected:**
- Flask — too minimal, no built-in validation, no async, no auto-docs. Cyprian used Flask and the result was fragile threaded servers.
- Django — too heavy for an API-only service, brings ORM we don't need (using SQLAlchemy).
**Reasoning:** FastAPI gives us Pydantic validation, automatic OpenAPI docs, async support, and type safety out of the box.

### Decision 002: PostgreSQL over SQLite (2026-03-01)
**Context:** Need a database for storing patient data, audit results, job tracking.
**Choice:** PostgreSQL
**Alternatives rejected:**
- SQLite — no concurrent access, no production deployment, file-based.
- MongoDB — our data is relational (patients → entries → audits), not document-oriented.
**Reasoning:** PostgreSQL is the industry standard for relational data, handles concurrency, works in Docker, and scales.

### Decision 003: Custom pipeline over LangGraph (2026-03-01)
**Context:** Need to orchestrate 4 agents in sequence.
**Choice:** Custom Pipeline class with plain Python function composition.
**Alternatives rejected:**
- LangGraph — adds heavy dependency, complex state management, overkill for a linear pipeline. Both reference implementations used it but gained little from it.
- Celery — too heavy for this; we're not distributing across workers yet.
**Reasoning:** A linear pipeline of 4 functions doesn't need a graph framework. Simple is better. We retain full control, testability, and debuggability.

### Decision 004: OpenAI GPT-4o-mini as default LLM (2026-03-01)
**Context:** Need an LLM for query generation and scoring.
**Choice:** OpenAI GPT-4o-mini (default), with provider abstraction for swapping.
**Alternatives rejected:**
- Mistral-7B local (Hiruni's choice) — underpowered for medical reasoning, requires local GPU/CPU resources.
- GPT-3.5-turbo (Cyprian's choice) — cheapest but weakest at complex medical evaluation.
- GPT-4o — more capable but significantly more expensive; 4o-mini offers 90% of the quality at 10% of the cost.
**Reasoning:** GPT-4o-mini balances cost and quality. The abstraction layer means we can switch to any provider (Claude, Gemini, local models) by changing one env var.

### Decision 005: SNOMED lookup without FHIR server (2026-03-01)
**Context:** Hiruni's implementation required a local HADES FHIR server for SNOMED lookups. This server is not included and is complex to set up.
**Choice:** Two-tier SNOMED categoriser: rule-based regex patterns (84% coverage) + LLM fallback (16%).
**Alternatives rejected:**
- Requiring HADES FHIR server — creates a heavy external dependency, not included in project files, complex to configure.
- NHS SNOMED CT API — requires registration, rate limits, external dependency.
- Pure LLM classification — unnecessary cost when most concepts have obvious keywords.
**Reasoning:** The dataset has 1,261 unique concepts with human-readable display names. Regex patterns matching medical keywords and suffixes (-itis, -ectomy, -pathy, -osis) classify 1,069 concepts instantly and for free. The remaining 192 edge cases use the LLM in batches of 50 (7 API calls total). Each concept is classified once, persisted to the `clinical_entries.category` column, and never re-classified. **Implemented:** `src/services/snomed_categoriser.py`.

### Decision 006: Template-first query generation over pure LLM (2026-03-01)
**Context:** Need to generate search queries from diagnoses for FAISS guideline retrieval.
**Choice:** Three-tier approach: hand-crafted templates for common MSK diagnoses, LLM for rare diagnoses, default generic queries as fallback.
**Alternatives rejected:**
- Pure LLM generation (Hiruni's approach) — every diagnosis goes through LLM, even "Low back pain" where we know exactly what queries work best. Slower, costs money, and LLM doesn't know how our FAISS index is structured.
- Pure template generation — wouldn't handle unusual diagnoses like "Acquired hallux valgus" or "Dupuytren's contracture".
**Reasoning:** For common MSK conditions (~15 templates), we can write better queries than an LLM because we know how NICE guidelines are titled and structured. Templates are free, instant, deterministic, and can be empirically tuned against the FAISS index. For rare diagnoses, the LLM generates queries with episode context (treatments, referrals). Default queries ensure the pipeline never fails. **Implemented:** `src/agents/query.py`.

### Decision 007: Numpy L2 normalization over faiss.normalize_L2 (2026-03-01)
**Context:** Embedding vectors need L2 normalization before FAISS search (so inner product = cosine similarity). Cyprian used `faiss.normalize_L2()`.
**Choice:** Pure numpy normalization (`embedding / np.linalg.norm(embedding)`).
**Alternatives rejected:**
- `faiss.normalize_L2()` — causes segmentation fault when called on numpy arrays derived from PyTorch tensors (non-writable memory). Known compatibility issue between faiss-cpu, torch, and numpy on macOS.
**Reasoning:** Mathematically identical result. Numpy normalization creates a new array, avoiding the in-place modification that crashes. No performance difference for our use case. **Implemented:** `src/services/embedder.py`.

### Decision 008: Full clinical context in scorer prompt (2026-03-01)
**Context:** Cyprian's scorer only passed treatments to the LLM. NICE guidelines also recommend referrals (e.g., "refer to physiotherapy"), investigations (e.g., "order blood tests"), and procedures.
**Choice:** Include treatments, referrals, investigations, and procedures in the scoring prompt.
**Alternatives rejected:**
- Treatments only (Cyprian's approach) — misses critical guideline adherence signals. A patient correctly referred to physiotherapy would score as non-adherent.
**Reasoning:** NICE guidelines cover all aspects of care, not just prescriptions. Including the full clinical context gives the LLM a complete picture and produces more accurate adherence scores. **Implemented:** `src/agents/scorer.py`.

### Decision 009: Conservative default scoring on parse failure (2026-03-01)
**Context:** If the LLM's response can't be parsed (garbled output, unexpected format), what score should we assign?
**Choice:** Default to -1 (non-adherent).
**Alternatives considered:**
- Default to +1 — too optimistic, could hide real non-adherence.
- Default to 0 or null — would require a third score type, complicating downstream analysis.
- Throw an error — too aggressive, would halt processing.
**Reasoning:** Defaulting to -1 is the conservative, safe choice. It ensures unparseable responses get flagged for human review rather than silently passing. Combined with the `error` field on DiagnosisScore, these cases can be easily identified and investigated.

### Decision 010: Background tasks for batch processing (2026-03-02)
**Context:** Batch auditing 4,327 patients will take hours (LLM calls for scoring). The HTTP request would time out.
**Choice:** Use FastAPI `BackgroundTasks` to run the batch in a background coroutine with its own DB session.
**Alternatives rejected:**
- Celery — too heavy for a single-process deployment. No distributed workers needed.
- Synchronous batch endpoint — would time out for large batches.
- WebSocket streaming — more complex, no need for real-time updates (polling is fine).
**Reasoning:** `BackgroundTasks` is built into FastAPI, requires no external broker, and integrates with our async pipeline. The client creates a job, gets back a job ID, and polls for progress. The background task uses per-patient session isolation (memory safety), per-patient timeouts (300s via `asyncio.wait_for`), commits after every patient, and handles its own error recovery. Stale jobs from crashes are cleaned up automatically on startup. **Implemented:** `src/api/routes/audit.py`.

### Decision 011: Separate reporting route file and service layer (2026-03-02)
**Context:** Need reporting/analytics endpoints for reviewing audit results. Could add to existing `audit.py` routes or create separate files.
**Choice:** Separate `src/services/reporting.py` (computation) + `src/api/routes/reports.py` (thin route layer).
**Alternatives rejected:**
- Adding to `audit.py` — audit routes handle pipeline execution (write path); reporting is read-only analytics (read path). Mixing them would grow `audit.py` and conflate concerns.
- No service layer — putting SQL queries in route handlers. Would make functions untestable without HTTP client, harder to extend for gold-standard metrics later.
**Reasoning:** Separation of concerns: reporting service is independently testable, extensible (add `get_validation_metrics()` later), and keeps routes thin. The `_load_completed_results()` helper avoids query duplication between functions that need details_json parsing. Python-side aggregation is appropriate because ~4,327 patients is trivially small in memory and `details_json` is TEXT not JSONB.

### Decision 012: Batched SNOMED LLM categorisation (2026-03-02)
**Context:** The original `categorise_by_llm` made one LLM call per unmatched concept (322 individual API calls). This caused out-of-memory crashes — each call accumulated HTTP response objects, connection state, and DEBUG-level log data. Combined with PubMedBERT (~440MB) already in memory, this exhausted available RAM.
**Choice:** Batch 50 concepts per prompt using JSON response format.
**Alternatives rejected:**
- Keep individual calls (original) — causes OOM at 322 calls.
- Batch with line-by-line text response — fragile, line alignment can shift if LLM adds/skips lines.
**Reasoning:** JSON response format (`{"concept": "category", ...}`) maps each concept explicitly — no line-alignment risk. 322 calls → 7 calls. Falls back to individual calls if a batch fails to parse. 80% match threshold retries misses individually. `gc.collect()` between batches. **Implemented:** `src/services/snomed_categoriser.py`.

### Decision 013: Category persistence to database (2026-03-02)
**Context:** SNOMED categories were only stored in an in-memory cache. Server crash = all classification work lost = 7 LLM batch calls repeated on next run. The `clinical_entries.category` column existed since Phase 1 but was always NULL.
**Choice:** Write classified categories back to the `clinical_entries.category` column after classification.
**Alternatives rejected:**
- Memory-only cache (original) — work lost on crash, LLM calls repeated every run.
- Separate cache table — unnecessary, the column already exists on clinical_entries.
**Reasoning:** Categories are stable (e.g., "Knee pain" is always "diagnosis"). Persist once, never re-classify. `load_categories_from_db()` now reads cached categories from DB first, only classifies uncategorised concepts, then writes results back. After first run: 0 LLM calls for categorisation. **Implemented:** `src/services/pipeline.py` (`load_categories_from_db`).

### Decision 014: Per-patient session isolation in batch processing (2026-03-02)
**Context:** The original batch handler used a single DB session for the entire batch. SQLAlchemy's identity map grew with every patient — all Patient, ClinicalEntry, and AuditResult objects accumulated in memory, causing OOM on large batches.
**Choice:** Create a fresh DB session per patient; session is closed and identity map freed after each patient.
**Alternatives rejected:**
- Single session with `session.expunge_all()` — fragile, breaks relationships and lazy loading.
- Single session with periodic `session.expire_all()` — still retains objects in identity map.
**Reasoning:** Fresh session per patient guarantees constant memory usage regardless of batch size. Each session commit is atomic — if a patient fails, only that patient's session is lost. Progress is committed after every patient so polling sees real-time updates. **Implemented:** `src/api/routes/audit.py` (`_run_batch_background`).

### Decision 015: OpenAI client timeout and per-patient pipeline timeout (2026-03-02)
**Context:** No timeouts anywhere in the pipeline. OpenAI SDK defaults to a 10-minute timeout per call. A single slow LLM response could stall the entire batch for 10 minutes per call, and one patient could block indefinitely.
**Choice:** Two-level timeouts: 60s per LLM call (on the OpenAI client), 300s per patient (via `asyncio.wait_for`).
**Alternatives rejected:**
- No timeouts (original) — hangs on slow responses, no recovery.
- Server-wide timeout only — too coarse, doesn't catch per-patient stalls.
**Reasoning:** 60s per LLM call is generous (most responses take 2-5s) but prevents indefinite hangs. 300s per patient covers the full pipeline (extraction + queries + retrieval + scoring). `max_retries=2` on the OpenAI client auto-retries transient errors. Both configurable via env vars (`OPENAI_REQUEST_TIMEOUT`, `PIPELINE_PATIENT_TIMEOUT`). **Implemented:** `src/config/settings.py`, `src/ai/openai_provider.py`, `src/api/routes/audit.py`.

### Decision 016: Stale job recovery on startup (2026-03-02)
**Context:** If the server crashes mid-batch, jobs stay stuck as "pending" or "running" forever. Polling clients see stale progress. No way to clean up without manual DB intervention.
**Choice:** On every server startup, find stuck jobs and mark them "failed" with a descriptive message.
**Alternatives rejected:**
- Manual DB cleanup — requires developer intervention after every crash.
- Auto-resume interrupted jobs — too complex, risk of processing patients twice.
**Reasoning:** Simple, automatic, safe. Any job that's "pending" or "running" when the server starts is definitively stale (the background task died with the old process). Marking them "failed" lets polling clients know what happened. **Implemented:** `src/main.py` (`_recover_stale_jobs`).

### Decision 017: aiosqlite for reporting tests (2026-03-02)
**Context:** Reporting functions execute real SQL queries (aggregations, filters). Need to test actual query logic.
**Choice:** In-memory SQLite via `aiosqlite` for test fixtures.
**Alternatives rejected:**
- Mocking `session.execute()` — fragile, requires mock objects for Result/Row types, breaks when query order changes. `get_dashboard_stats()` makes 4 separate execute calls, each needing different mock return types.
- Requiring PostgreSQL for tests — adds infrastructure dependency, slows tests, unnecessary for unit-level validation.
**Reasoning:** In-memory SQLite tests real SQL queries without external dependencies. Tables are created from the same SQLAlchemy models (via `Base.metadata.create_all`), ensuring schema stays in sync. Tests run in <1 second and are robust against implementation refactors.

### Decision 018: 5-level scoring scale over binary (2026-03-10)
**Context:** Supervisor feedback requested finer-grained scoring. Binary +1/-1 doesn't capture partial compliance or distinguish unsafe care from mere omissions.
**Choice:** 5-level scale: -2 RISKY, -1 NON-COMPLIANT, 0 NOT RELEVANT, +1 PARTIAL, +2 COMPLIANT. Each score includes confidence (0.0-1.0) and cited NICE guideline text.
**Alternatives rejected:**
- 3-level scale (adherent/partial/non-adherent) — doesn't capture safety-critical distinction between non-compliance and risky non-compliance.
- Continuous 0-1 score — harder for LLMs to produce consistently, less interpretable for clinicians.
**Reasoning:** The 5 levels map naturally to clinical audit categories. The -2 (RISKY) level flags safety-critical cases that need immediate review. Confidence scores and NICE citations improve transparency and auditability. Backward compatibility is maintained — existing DB records with binary scores are detected via `"judgement" in ds` and handled correctly.

### Decision 019: Ollama via OpenAI-compatible API (2026-03-10)
**Context:** Supervisor requested local LLM option for patient data processing without external API calls.
**Choice:** Ollama with OpenAI SDK pointing to `http://localhost:11434/v1`. Registered as `"ollama"` and `"local"` (alias) in the provider factory.
**Alternatives rejected:**
- Direct Ollama REST API — would require custom HTTP client code. Ollama's OpenAI-compatible endpoint lets us reuse the openai SDK.
- LlamaCpp / vLLM — more complex setup, Ollama provides simpler one-command model management.
**Reasoning:** Ollama's OpenAI-compatible endpoint means zero new dependencies — we reuse the existing `openai` SDK. The provider pattern makes switching between OpenAI and Ollama a one-line .env change. Embeddings still use PubMedBERT (not the LLM provider).

### Decision 020: Post-retrieval relevance filter over re-indexing (2026-03-11)
**Context:** 50-patient comparison (OpenAI vs Ollama) showed 80% of disagreements caused by FAISS retriever returning off-topic guidelines. Ollama correctly detected these (scored 0 NOT RELEVANT); OpenAI masked them by giving credit anyway. Root cause: the 277K-guideline corpus includes oncology, cardiology, etc. — PubMedBERT embeddings lack domain specificity to distinguish MSK from non-MSK.
**Choice:** Two-layer post-retrieval filter in `_filter_irrelevant()`: (A) title keyword checks — exclude obviously wrong specialties and enforce topic overlap with diagnosis, (B) configurable L2 distance threshold. With fallback to best match if everything is filtered.
**Alternatives rejected:**
- Re-index with only MSK guidelines — would lose cross-specialty references (e.g., cardiovascular risk in rheumatoid arthritis). Also requires maintaining a separate filtered dataset.
- LLM-as-judge pre-filter — too expensive (1 LLM call per retrieved guideline × 5 guidelines × N diagnoses).
- Re-train embeddings — out of scope, PubMedBERT is good enough for most MSK queries.
**Reasoning:** Lightweight keyword + threshold filtering catches the obvious mismatches (cancer guidelines for back pain, chest pain guidelines for carpal tunnel) without LLM cost or data pipeline changes. Novel/rare diagnoses that don't match any topic group skip the topic filter to avoid false negatives.

### Decision 021: No LLM fine-tuning (2026-03-15)
**Context:** Supervisor asked whether any data was used for fine-tuning and what data splitting strategy was followed.
**Choice:** No fine-tuning. All LLMs (GPT-4o-mini, Mistral-Small) are used as pre-trained models with zero-shot prompting only. No train/validation/test split was applied to the data.
**Alternatives rejected:**
- Fine-tune scorer on gold-standard audits: Only 120 labelled cases exist (far below the thousands needed for meaningful fine-tuning). These 120 cases are more valuable as a held-out validation set to measure system accuracy than as training data.
- Fine-tune extractor/query agent: These agents are predominantly rule-based (84% regex categorisation, template-based query generation). LLM is only a fallback for edge cases, and results are cached permanently in the database after first classification. Fine-tuning a model for a rarely-used fallback path is not justified.
- Fine-tune a smaller model to replace the scorer LLM: Would require a large corpus of clinician-labelled audit judgements (input: patient record + guidelines, output: correct score + explanation). This labelled data does not exist. The scorer also operates on dynamically retrieved guideline context that changes per patient, making static training examples less representative.
**Reasoning:** This system is a Retrieval-Augmented Generation (RAG) pipeline, not a generative model. The core value comes from (1) high-quality retrieval via PubMedBERT + FAISS finding the right NICE guidelines, and (2) carefully engineered prompts that instruct the LLM how to evaluate adherence. All significant quality improvements came from prompt engineering (binary to 5-level scoring, confidence scores, citation requirements, missing care detection) and retrieval filtering (topic-keyword exclusion, L2 distance thresholds), not from model training. Fine-tuning would also introduce a maintenance burden: any change to the scoring rubric, guideline corpus, or output format would require re-training. With prompt engineering, these changes are immediate and zero-cost. The only scenario where fine-tuning would be justified is if, after gold-standard validation (Phase 7b), we accumulate hundreds of clinician-corrected audit scores and want to distill the scoring behaviour into a cheaper/faster model. This remains a future consideration, not a current need.

### Decision 022: Concurrent batch processing (2026-03-15)
**Context:** Auditing 50 patients took ~30 min with Ollama (sequential). Scaling to 1000 patients would take 10+ hours.
**Choice:** Added `BATCH_CONCURRENCY` setting (default 5) using `asyncio.Semaphore` + `asyncio.gather` in the batch loop. Each patient still gets its own DB session. Progress updates happen after each chunk.
**Alternatives rejected:**
- Multiprocessing: Overkill for I/O-bound LLM calls. asyncio concurrency is simpler and sufficient.
- Always-parallel with no limit: Would overwhelm Ollama (single GPU) and hit OpenAI rate limits.
**Reasoning:** Each patient is fully independent (own diagnoses, FAISS lookups, LLM calls). FAISS index is read-only, DB handles concurrent writes, LLM calls are stateless. Results are identical to sequential processing. OpenAI benefits most (5x speedup). Ollama benefits modestly (10-15% with concurrency=2) since it processes one inference at a time by default. Users can increase `OLLAMA_NUM_PARALLEL` for more Ollama parallelism.

### Decision 023: Model-based report filtering (2026-03-15)
**Context:** User wanted to audit more patients across multiple jobs and get combined reports per model, not per job.
**Choice:** Added `model` query parameter to all report, export, and evaluation endpoints. Filters by joining `audit_results.job_id` to `audit_jobs.provider`. Works alongside existing `job_id` filter.
**Reasoning:** Allows combining results from multiple batch jobs that used the same model (e.g. Job 1 + Job 3 both GPT-4.1-mini) into a single report. No schema changes needed.

### Decision 024: Deterministic evaluation ordering over random sampling (2026-03-16)
**Context:** Evaluation endpoints (`evaluate/scorer`, `evaluate/agents`) previously selected random patients. This made cross-model comparison unfair (different patients per model) and prevented resuming partial evaluation runs.
**Choice:** All evaluation endpoints now sort patients by `pat_id` (deterministic) and accept an `offset` parameter alongside `limit`. The scorer endpoint was consolidated from `POST /evaluate/scorer/{job_id}` (path param) to `POST /evaluate/scorer` with query params (`?model=&limit=&offset=`). Audit job results (`GET /audit/jobs/{job_id}/results`) now sort by `pat_id` then `AuditResult.id`.
**Alternatives rejected:**
- Random sampling with fixed seed — deterministic per-call but seed management adds complexity and doesn't support resumable evaluation.
- Requiring explicit `pat_ids` list — shifts the burden to the caller to know which patients exist.
**Reasoning:** Deterministic `pat_id` ordering ensures that `?model=gpt-4.1-mini&limit=50&offset=0` and `?model=mistral-small&limit=50&offset=0` evaluate the exact same 50 patients, enabling fair apples-to-apples comparison. The `offset` parameter enables resumable evaluation: evaluate 20 patients, then continue from offset=20 without re-evaluating. This is critical for expensive evaluation operations (each patient requires multiple LLM calls).

---

## 7. Current State Summary

**Date:** 2026-03-16
**Phase 0-7a:** COMPLETE — Full pipeline + reporting + exports
**Phase 8:** COMPLETE — Supervisor Feedback (agent renaming, 5-level scoring, Ollama)
**Phase 9:** COMPLETE — Evaluation Framework (9a Model Comparison ✅, 9b Missing Care ✅, 9d LLM-as-Judge ✅, 9e Visualizations ✅, 9c skipped until clinician labels arrive)
**Phase 10a:** COMPLETE — Retriever relevance filtering (post-retrieval title exclusion + topic matching + L2 threshold)
**Phase 10b:** COMPLETE — Comprehensive evaluation endpoints: system metrics, cross-model classification (confusion matrix, P/R/F1, AUROC, 5-class kappa), extractor metrics from DB, full agent evaluation with retriever IR (Precision@k, nDCG, MRR), 3 comparison chart SVGs
**Phase 10c:** COMPLETE — Comparison HTML report with LLM-as-Judge results (cross-model judging: each model judged by both GPT-4o-mini and mistral-small), full agent evaluation results, all saved to `exports/supervisor-report/`
**Phase 11:** COMPLETE — Docker fixes (DB_PORT override, HuggingFace cache volume, Dockerfile permissions), model upgrade (gpt-4o-mini to gpt-4.1-mini), Ollama Docker networking (host.docker.internal), concurrent batch processing (BATCH_CONCURRENCY), model-based report filtering (`?model=` param on all endpoints)
**Phase 11b:** COMPLETE — Deterministic evaluation ordering + resumable evaluation + cross-model fair comparison
**Phase 11c:** COMPLETE — Large-scale evaluation runs: scorer eval (100 patients × 4 judge combos), agent eval (50 patients × 2 models) with merged results in `data/eval_results/`
**Phase 11d:** COMPLETE — Comprehensive individual reports: individual HTML reports (`/export/html`) now include system metrics, extractor quality, LLM-as-Judge scorer quality, agent-level evaluation, and missing care opportunities sections; `use_saved_evals=true` param loads pre-computed eval data; fixed truncated patient IDs and variable shadowing bug

**What was done in Phase 9 (so far):**

*Model Comparison Service (9a):*
- `AuditJob.provider` column tracks which AI provider ran each batch
- Comparison service reads two jobs' results, matches patients by `pat_id`, matches diagnoses by `(term, index_date)`
- Computes: per-patient score diffs, per-condition adherence deltas, Cohen's kappa (inter-rater agreement), Pearson correlation
- `GET /api/v1/evaluation/compare?job_a=1&job_b=2` endpoint — ready for OpenAI vs Ollama comparison once both models have run

*Missing Care Opportunities (9b):*
- Scorer now outputs "Missing Care Opportunities" — specific NICE-recommended actions not documented in the patient record
- New `missing_care_opportunities` field in `DiagnosisScore`, parsed from LLM response, included in all outputs (JSON, CSV, HTML)
- `GET /api/v1/evaluation/missing-care` endpoint groups gaps by condition with frequency counts
- HTML report shows amber "Missing care" tags on diagnosis cards when gaps are identified

*LLM-as-Judge Evaluation (9d):*
- Evaluation service (`src/services/evaluation.py`) evaluates each pipeline agent's output quality
- Extractor: weak supervision via SNOMED rules as pseudo-ground-truth → per-category P/R/F1 + rule_match_rate (no LLM cost)
- Query generator: LLM-as-Judge rates relevance (1-5) and coverage (1-5) per diagnosis
- Retriever: LLM-as-Judge rates guideline relevance (1-5) per diagnosis
- Scorer: LLM-as-Judge rates reasoning quality, citation accuracy, score calibration (all 1-5)
- `scoring_from_stored()` enables scorer evaluation from stored `details_json` without re-running the pipeline
- `POST /api/v1/evaluation/evaluate/scorer/{job_id}` endpoint — evaluates scorer from stored batch results

*Tests:* 371/371 passing (up from 256)

**The system now has both pipeline execution and reporting analytics.** Available endpoints:
```
Health:
  GET  /health                                — Liveness check
  GET  /health/ready                          — Readiness check

Data:
  GET  /api/v1/data/stats                     — Database row counts
  POST /api/v1/data/import/patients           — Import patient CSV
  POST /api/v1/data/import/guidelines         — Import guidelines CSV

Audit (write path):
  POST /api/v1/audit/patient/{pat_id}         — Single patient audit
  POST /api/v1/audit/batch                    — Batch audit (background, supports ?limit=N&skip_audited=true)
  GET  /api/v1/audit/jobs/{job_id}            — Job progress
  GET  /api/v1/audit/jobs/{job_id}/results    — Paginated job results (sorted by pat_id then id, supports ?status=failed)
  GET  /api/v1/audit/results/{pat_id}         — All results for a patient

Reports (read path — all support ?model= and ?job_id= filtering):
  GET  /api/v1/reports/dashboard              — Summary stats
  GET  /api/v1/reports/conditions             — Per-condition breakdown
  GET  /api/v1/reports/non-adherent           — Non-adherent cases
  GET  /api/v1/reports/score-distribution     — Score histogram

Exports (shareable — all support ?model= and ?job_id= filtering):
  GET  /api/v1/reports/export/csv             — Download CSV file (one row per diagnosis)
  GET  /api/v1/reports/export/html            — Self-contained HTML report with full eval sections (open in browser, +use_saved_evals flag)
  GET  /api/v1/reports/export/comparison-html — Cross-model comparison HTML (?job_a=&job_b= or ?model_a=&model_b=, +include_scorer_eval flag)

Evaluation (Phase 9 + 10b — deterministic pat_id ordering with offset/limit):
  GET  /api/v1/evaluation/compare             — Compare two batch jobs (job_a, job_b params)
  GET  /api/v1/evaluation/missing-care        — Missing care opportunities aggregation
  POST /api/v1/evaluation/evaluate/scorer     — LLM-as-Judge scorer evaluation (?model=&limit=&offset=, deterministic pat_id sort)
  GET  /api/v1/evaluation/system-metrics      — Score class distribution, adherence rate, confidence stats
  GET  /api/v1/evaluation/cross-model-metrics — Confusion matrix, per-class P/R/F1, AUROC, 5-class kappa
  GET  /api/v1/evaluation/extractor-metrics   — Extractor SNOMED categorisation P/R/F1 (no LLM needed)
  POST /api/v1/evaluation/evaluate/agents     — Full 4-agent evaluation with retriever IR metrics (?limit=&offset=, deterministic pat_id sort)
```

**Key files (Phase 9 + 10b):**
- Comparison service: `src/services/comparison.py` (Cohen's kappa, Pearson, per-condition deltas, **+ 5×5 confusion matrix, per-class P/R/F1, AUROC, 5-class kappa**)
- Evaluation service: `src/services/evaluation.py` (LLM-as-Judge evaluation for all 4 agents, **+ evaluate_extractor_from_db, evaluate_retrieval_ir with IR metrics, run_agent_evaluation orchestrator**)
- Evaluation routes: `src/api/routes/evaluation.py` (compare + missing care + scorer evaluation **+ system-metrics + cross-model-metrics + extractor-metrics + evaluate-agents endpoints**)
- Reporting additions: `src/services/reporting.py` (`get_missing_care_summary()`, **+ `compute_system_metrics()` — class distribution, adherence rate, confidence stats**)
- Scorer changes: `src/agents/scorer.py` (new `missing_care_opportunities` field + prompt + parser)
- Migration: `migrations/versions/002_add_job_provider.py` (`provider` column on `audit_jobs`)
- Export visualizations: `src/services/export.py` (SVG chart helpers + `_collect_chart_data()` + `export_charts_to_png()`, **+ `_svg_confusion_matrix()`, `_svg_comparison_scores()`, `_svg_comparison_compliance()`, `generate_comparison_html()`, `_build_scorer_eval_section()`, `_build_agent_eval_section()`, `_kappa_label()`**, **+ `_build_system_metrics_html()`, `_build_extractor_html()`, `_build_scorer_eval_single_html()`, `_build_agent_eval_single_html()`, `_build_missing_care_html()`**)
- Reports routes: `src/api/routes/reports.py` (**+ `GET /export/comparison-html` with optional `include_scorer_eval` flag**, **+ `use_saved_evals` param on `GET /export/html`**)
- Chart export script: `scripts/export_charts.py` (CLI for saving PNG charts to local folder)
- Tests: `tests/unit/test_comparison.py` (29), `tests/unit/test_missing_care.py` (12), `tests/unit/test_evaluation.py` (28), `tests/unit/test_export.py` (36), `tests/unit/test_reporting.py` (+4 system metrics)

**Post-Phase 7a crash fixes (2026-03-02):**

*Server stability fixes (batch audit was crashing):*
- Fixed `flush()` → `commit()` — batch job status was invisible to polling clients because `flush()` doesn't commit the PostgreSQL transaction. Other sessions can't read uncommitted data.
- Per-patient session isolation — each patient gets a fresh DB session. SQLAlchemy's identity map (which tracks every ORM object) is freed after each patient, keeping memory constant regardless of batch size.
- Batched SNOMED LLM categorisation — `categorise_by_llm` was making 322 individual API calls (one per unmatched concept), causing OOM. Now batches 50 concepts per prompt using JSON response format → 7 API calls instead of 322. Falls back to individual calls on parse failure.
- Category persistence to DB — categories written to `clinical_entries.category` column after classification. Subsequent runs load from DB with 0 LLM calls. Server crash doesn't lose categorisation work.
- OpenAI client timeout (60s) + `max_retries=2` — default SDK timeout was 10 minutes. Now caps each LLM call at 60s and auto-retries transient errors.
- Per-patient pipeline timeout (300s) — `asyncio.wait_for` prevents one slow patient from stalling the entire batch.
- `_save_patient_error_and_progress()` helper — when a patient timeout/error kills its session, stores the failed AuditResult and updates job progress in a clean session.
- `_recover_stale_jobs()` on startup — finds jobs stuck as "pending" or "running" from a previous crash and marks them "failed". Runs once per server boot.
- `gc.collect()` every 10 patients + after batch completion — forces garbage collection to free memory promptly.

*Other improvements:*
- Pre-loaded PubMedBERT embedder at startup — the HTTP server was crashing because the ~440MB model loaded lazily during the first request. Now loads on startup alongside FAISS index.
- Fixed `Makefile` `run` target — added `DB_HOST=localhost` so `make run` works locally.
- Updated `README.md` — professional getting-started guide with pipeline diagram, all endpoints, usage examples.
- Added `scripts/build_index.py` — builds FAISS index from `guidelines.csv` using PubMedBERT. Previously relied on Cyprian's pre-built file; now the system can rebuild from scratch.
- Improved Swagger API docs — added `response_model`, `summary`, `Field(description=...)` to all endpoints.
- Added `GET /api/v1/data/stats` — database row counts.
- Added `?limit=N` to batch endpoint — `POST /api/v1/audit/batch?limit=50`.
- Added `GET /api/v1/audit/jobs/{job_id}/results` with pagination and `?status=` filter.
- Added `?skip_audited=true` to batch endpoint — skips patients that already have a completed audit result (enables incremental batching).

**Key files changed:**
- `src/api/routes/audit.py` — per-patient sessions, timeouts, `_save_patient_error_and_progress`, gc.collect
- `src/services/snomed_categoriser.py` — batched LLM categorisation (50 per prompt)
- `src/services/pipeline.py` — category persistence (load from DB → classify new → write back)
- `src/ai/openai_provider.py` — client timeout (60s) + max_retries (2)
- `src/config/settings.py` — `openai_request_timeout`, `pipeline_patient_timeout`
- `src/main.py` — `_recover_stale_jobs()`, PubMedBERT pre-loading, `faulthandler`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`
- `src/services/embedder.py` — `.detach()` tensor cleanup, `np.ascontiguousarray()` output
- `src/agents/retriever.py` — `encode_batch()` per diagnosis instead of individual `encode()` per query, diagnostic logging
- `src/services/vector_store.py` — `np.ascontiguousarray()` before FAISS search, auto-decompress `.csv.gz`
- `src/services/snomed_categoriser.py` — removed 'other' from categories, updated LLM prompts, added 10 new rule patterns
- `src/agents/extractor.py` — fallback default changed from 'other' to 'administrative'
- `src/agents/query.py` — entry-level dedup by (term, index_date) + term-level query cache across episodes
- `src/agents/retriever.py` — entry-level dedup by (term, index_date) + term-level retrieval cache across episodes; **Phase 10a:** post-retrieval relevance filter (`_filter_irrelevant()`) with topic keyword mapping, title exclusion, and L2 distance threshold
- `src/agents/scorer.py` — entry-level dedup by (term, index_date), eliminates duplicate entries in reports; scoring prompt rewritten for sparse coded data; regex parser fixed for bracket format `[+1]`
- `tests/unit/test_scorer.py` — added `test_parse_score_with_square_brackets`
- `src/services/export.py` — CSV and HTML report generation service
- `tests/unit/test_export.py` — 12 tests for CSV and HTML export (234 tests total)

**Blockers:** None.

**Next steps:**
1. ~~Phase 9e: Visualizations~~ ✅ — inline SVG charts in HTML reports
2. ~~PNG chart export~~ ✅ — `scripts/export_charts.py` saves charts as PNG files
3. ~~Provider-aware skip_audited~~ ✅ — `skip_audited=true` now scopes to current `AI_PROVIDER`, so switching OpenAI→Ollama won't skip patients
4. ~~Phase 10a: Retriever relevance filter~~ ✅ — post-retrieval filtering for irrelevant guidelines
5. ~~Re-run OpenAI + Ollama batches~~ ✅ — Job 1 (OpenAI, 50 patients), Job 2 (Ollama, 50 patients) completed
6. ~~Phase 10b: Comprehensive evaluation endpoints~~ ✅ — system metrics, cross-model classification, extractor metrics, full agent evaluation, comparison charts

**Multi-provider hardening (2026-03-11):**
- Health check readiness endpoint (`/health/ready`) made provider-aware — checks appropriate API key per provider (OpenAI → `openai_api_key`, Anthropic → `anthropic_api_key`, Ollama/Local → no key needed)
- HTML report patient ID truncation fixed — was showing `{pat_id[:12]}...`, now shows full UUID
- Confirmed `"local"` is an alias for `"ollama"` in AI factory — both create `OllamaProvider`
- All API endpoints verified provider-agnostic via factory pattern

7. ~~Run new evaluation endpoints against stored Job 1 & Job 2 data to collect results for supervisor report~~ ✅ — All results generated and saved to `exports/supervisor-report/`
8. ~~Generate comparison HTML report with all evaluation data~~ ✅ — `comparison-report-full.html` (38KB)
9. Phase 9c: Gold-standard metrics (when 120 clinician labels arrive)
10. Re-run batches after retriever relevance filter to validate improvement

**Deterministic evaluation ordering (2026-03-16):**
- All evaluation endpoints (`evaluate/scorer`, `evaluate/agents`) now use deterministic `pat_id` sorting instead of random patient selection
- Added `offset` parameter for resumable evaluation (evaluate patients in batches without re-evaluating)
- Removed `pat_ids` parameter from scorer endpoint; removed duplicate `POST /evaluate/scorer/{job_id}` path-param endpoint (kept only query-param version)
- `GET /audit/jobs/{job_id}/results` now returns results sorted by `pat_id` then `AuditResult.id` for stable pagination
- `_load_results_for_scorer()` in `reports.py` uses deterministic `pat_id` sorting + offset
- Cross-model fair comparison: same `offset`/`limit` produces same patients for different models
- All report/export endpoints support `?model=` query param; comparison-html supports `?model_a=`/`?model_b=`

**Key files changed (Phase 11b):**
- `src/api/routes/evaluation.py` — scorer endpoint consolidated to query params, agents endpoint uses deterministic ordering + offset
- `src/api/routes/reports.py` — `_load_results_for_scorer()` deterministic sort + offset; comparison-html supports model_a/model_b
- `src/api/routes/audit.py` — job results sorted by pat_id then id
- `src/services/evaluation.py` — `run_agent_evaluation()` uses deterministic pat_id ordering + offset

---

## 8. Known Issues / Tech Debt

- **Local PostgreSQL port conflict:** Host machine has a native PostgreSQL on port 5432, so our Docker DB uses port 5433. When running Alembic or scripts locally, must set `DB_HOST=localhost DB_PORT=5433`.
- **torch version pinned to 2.2.2:** Python 3.11 doesn't support torch 2.5.1. Will need updating if Python is upgraded.
- **SNOMED categoriser coverage at 84%:** 192 of 1,261 concepts require LLM fallback (now batched, 7 calls total). Coverage could be improved by adding more patterns, but diminishing returns — LLM handles the rest. Categories are persisted to DB after first classification. 'other' category has been eliminated — all concepts must map to one of 6 categories (diagnosis, treatment, procedure, referral, investigation, administrative).
- **faiss.normalize_L2 segfault on macOS:** `faiss.normalize_L2()` crashes when called on numpy arrays from PyTorch tensors. Worked around by using numpy normalization instead. May not affect Linux/Docker.
- **HuggingFace tokenizers parallelism segfault on macOS:** HuggingFace tokenizers use Rust-based (rayon) parallelism internally, which creates threads that conflict with uvicorn's async event loop and Python thread pools, causing segfaults. Fixed by setting `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1` before any imports in `src/main.py`. May not affect Linux/Docker.
- **PyTorch tensor → numpy memory lifecycle:** Calling `.numpy()` on a tensor without `.detach()` can cause segfaults when the tensor is garbage-collected while numpy still references it. Fixed by adding `.detach()` before `.numpy()`, explicitly `del`eting intermediate tensors, and returning `np.ascontiguousarray()` from the embedder. FAISS search also uses `np.ascontiguousarray()` to guarantee memory alignment.
- **PubMedBERT requires ~2GB RAM:** The embedding model (~440MB on disk) needs significant memory. Loaded at startup via lifespan handler so it's ready before any HTTP request arrives.
- **Embedder tests use bert-tiny model:** Real PubMedBERT (~440MB) too large for unit tests. Tests use `prajjwal1/bert-tiny` (17MB, 128-dim) — same encoding logic, different weights. Integration tests with real model needed.
- **Scorer tests use mock LLM:** Unit tests mock the AI provider. Integration tests with a real LLM needed to validate prompt quality and parsing against actual LLM outputs.
- **~~Binary scoring only~~ RESOLVED (Phase 8b):** Scoring upgraded to 5-level scale (-2 to +2) with confidence and NICE citations. Legacy binary results remain readable via backward-compat detection.
- **Sparse coded data bias:** Most patients have only diagnoses + referrals in coded SNOMED data, no explicit treatments. Scoring prompt is tuned to give benefit of the doubt (referral alone scores positively), but this may over-count adherence for patients where a referral was made for an unrelated condition. Gold-standard validation (Phase 7b) will quantify this.
- **~~Retriever returns irrelevant guidelines~~ RESOLVED (Phase 10a):** FAISS retriever returned off-topic guidelines (e.g., chest pain for carpal tunnel, diabetic foot for generic foot pain) due to PubMedBERT embedding imprecision across the 277K-guideline corpus. 80% of OpenAI vs Ollama disagreements were caused by this. Fixed with two-layer post-retrieval filter: title keyword exclusion + body-region topic matching + L2 distance threshold (default 1.2). Needs re-validation with fresh batch runs.

---

## 9. Environment & Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- An OpenAI API key (or alternative LLM provider key)
- ~2 GB RAM (for PubMedBERT model loading)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/anasraza57/clinaudit-ai.git
cd clinaudit-ai

# Copy environment template and fill in your values
cp .env.example .env
# Edit .env — at minimum, set OPENAI_API_KEY

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the database
docker compose up -d db

# Run database migrations
DB_HOST=localhost alembic upgrade head

# Import data into PostgreSQL
DB_HOST=localhost python3 scripts/import_data.py

# Build the FAISS guideline index from guidelines.csv
python3 scripts/build_index.py

# Start the app (first launch takes 30-60s to load PubMedBERT)
make run
# or: DB_HOST=localhost uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# The API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
# Health check at http://localhost:8000/health
```

### Running Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Environment Variables
See `.env.example` for all required variables with descriptions.

### Important Notes
- Database runs on port **5433** (not 5432) to avoid conflicts with local PostgreSQL.
- When running commands locally (not inside Docker), always set `DB_HOST=localhost`.
- Guidelines CSV is stored compressed (`data/guidelines.csv.gz`). The import script decompresses on the fly.
- First startup takes 30–60 seconds — the server pre-loads PubMedBERT (~440MB) and the FAISS index before accepting requests. Watch the logs for "Embedding model loaded" and "Vector store ready".
- First batch run will make ~7 LLM calls to classify ~322 SNOMED concepts. After that, categories are persisted in the DB — subsequent runs need zero LLM calls for categorisation.
- Timeout settings (configurable via env vars): `OPENAI_REQUEST_TIMEOUT=60` (seconds per LLM call), `PIPELINE_PATIENT_TIMEOUT=300` (seconds per patient in batch processing).
