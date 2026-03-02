# PROJECT BIBLE — GuidelineGuard

> **Last Updated:** 2026-03-02
> **Status:** Phase 7a COMPLETE (Reporting Endpoints) + Post-7a Crash Fixes — Next: Phase 7b (Gold-Standard Validation)

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
Patient Record → [Extractor] → [Query Generator] → [Retriever] → [Scorer] → Audit Report
```

1. **Extractor Agent** — Reads structured patient data (SNOMED-coded clinical entries), categorises each entry as a diagnosis, treatment, procedure, referral, etc.
2. **Query Agent** — Takes extracted clinical concepts and generates targeted search queries for finding relevant guidelines.
3. **Retriever Agent** — Uses semantic search (PubMedBERT embeddings + FAISS vector index) to find the most relevant NICE guideline passages for each query.
4. **Scorer Agent** — Compares documented clinical decisions against retrieved guidelines using an LLM, producing per-diagnosis adherence scores (+1 adherent / -1 non-adherent) with explanations of guidelines followed/not followed, and a final aggregate score (proportion of adherent diagnoses).

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

This project builds upon the **GuidelineGuard** framework (Shahriyear, 2024) and two MSc dissertations from Keele University:
- **Hiruni Vidanapathirana** — built the Extractor + Query agents
- **Cyprian Toroitich** — built the Retriever + Scorer agents

We are **not copying** their work. We are analysing it, taking what's good, fixing what's bad, and rebuilding the entire system as a unified, production-grade pipeline.

---

## 2. Reference Codebases Analysis

### 2A. GuidelineGuard Paper (Shahriyear, 2024)

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
│                        GuidelineGuard                            │
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
| **Guideline Scoring** | LLM with structured prompt | Per-diagnosis scoring via LLM (temperature=0), includes full clinical context + up to 2,000 chars of guideline text. Case-insensitive regex parsing of structured output. |
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
- [x] Push to GitHub (github.com/anasraza57/guideline-guard)

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
- ✅ Added `scripts/build_index.py` — rebuilds FAISS index from guidelines.csv (2026-03-02)
- ✅ Improved Swagger docs — response_model, summary, Field descriptions on all endpoints (2026-03-02)
- ✅ Added `GET /api/v1/data/stats`, `?limit=N` for batch, `GET /audit/jobs/{job_id}/results` pagination (2026-03-02)
- ✅ Learning docs updated: `05-extractor-agent-explained.md`, `09-pipeline-integration-explained.md` (2026-03-02)

### Phase 7b: Gold-Standard Validation
- [ ] Import gold-standard audit data (120 cases)
- [ ] Run system against gold-standard cases
- [ ] Compare AI scores vs human auditor scores
- [ ] Generate accuracy/agreement metrics (Cohen's kappa, etc.)
- [ ] Add `get_validation_metrics()` to reporting service + new endpoint
- [ ] Write tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 8: Polish & Documentation
- [ ] Performance optimisation (batch embeddings, caching, concurrent API calls)
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
- ✅ Push to GitHub — github.com/anasraza57/guideline-guard (2026-03-01)
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

---

## 7. Current State Summary

**Date:** 2026-03-02
**Phase 0:** COMPLETE
**Phase 1:** COMPLETE — Database, migrations, data import, vector store, 4327 patients + 1656 guidelines loaded
**Phase 2:** COMPLETE — Extractor Agent with SNOMED categoriser (batched LLM, DB persistence)
**Phase 3:** COMPLETE — Query Agent with template-based + LLM query generation
**Phase 4:** COMPLETE — Retriever Agent with PubMedBERT embeddings + FAISS search
**Phase 5:** COMPLETE — Scorer Agent with LLM-based guideline adherence scoring
**Phase 6:** COMPLETE — Pipeline Integration with REST API
**Phase 7a:** COMPLETE — Reporting Endpoints (4 analytics endpoints + service layer)

**What was done in Phase 7a:**
- Reporting service: `src/services/reporting.py` with 4 public analytics functions + 1 shared query helper
- Dashboard stats: total audited/failed, mean/median/min/max adherence score, failure rate (SQL aggregation)
- Condition breakdown: per-diagnosis adherence rates from details_json, with min_count filter and sort options
- Non-adherent cases: paginated list of score=-1 diagnoses with explanations for clinical review
- Score distribution: histogram of patient-level overall_score with configurable bins
- Report API: `src/api/routes/reports.py` with 4 GET endpoints + Pydantic response schemas
- All endpoints accept optional `?job_id=N` to scope to a specific batch run
- Added `aiosqlite==0.20.0` for async in-memory SQLite testing
- 216 unit tests passing (up from 190 in Phase 6, +26 new reporting tests)
- Learning doc: `docs/learning/10-reporting-explained.md`

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
  POST /api/v1/audit/batch                    — Batch audit (background, supports ?limit=N)
  GET  /api/v1/audit/jobs/{job_id}            — Job progress
  GET  /api/v1/audit/jobs/{job_id}/results    — Paginated job results
  GET  /api/v1/audit/results/{pat_id}         — All results for a patient

Reports (read path):
  GET  /api/v1/reports/dashboard              — Summary stats
  GET  /api/v1/reports/conditions             — Per-condition breakdown
  GET  /api/v1/reports/non-adherent           — Non-adherent cases
  GET  /api/v1/reports/score-distribution     — Score histogram
```

**Key files:**
- Reporting service: `src/services/reporting.py`
- Report API: `src/api/routes/reports.py`
- Tests: `tests/unit/test_reporting.py`

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
- Added `GET /api/v1/audit/jobs/{job_id}/results` with pagination.

**Key files changed:**
- `src/api/routes/audit.py` — per-patient sessions, timeouts, `_save_patient_error_and_progress`, gc.collect
- `src/services/snomed_categoriser.py` — batched LLM categorisation (50 per prompt)
- `src/services/pipeline.py` — category persistence (load from DB → classify new → write back)
- `src/ai/openai_provider.py` — client timeout (60s) + max_retries (2)
- `src/config/settings.py` — `openai_request_timeout`, `pipeline_patient_timeout`
- `src/main.py` — `_recover_stale_jobs()`, PubMedBERT pre-loading

**Blockers:** None.

**Next session should start with:** Phase 7b — Gold-Standard Validation
1. Import gold-standard audit data (120 cases manually scored by clinicians)
2. Run our pipeline against those 120 patients
3. Compare AI scores vs human auditor scores
4. Generate accuracy/agreement metrics (Cohen's kappa, etc.)
5. Add `get_validation_metrics()` to reporting service + new endpoint
6. Write tests
7. Update learning docs + PROJECT_BIBLE.md

---

## 8. Known Issues / Tech Debt

- **Local PostgreSQL port conflict:** Host machine has a native PostgreSQL on port 5432, so our Docker DB uses port 5433. When running Alembic or scripts locally, must set `DB_HOST=localhost DB_PORT=5433`.
- **torch version pinned to 2.2.2:** Python 3.11 doesn't support torch 2.5.1. Will need updating if Python is upgraded.
- **SNOMED categoriser coverage at 84%:** 192 of 1,261 concepts require LLM fallback (now batched, 7 calls total). Coverage could be improved by adding more patterns, but diminishing returns — LLM handles the rest. Categories are persisted to DB after first classification.
- **faiss.normalize_L2 segfault on macOS:** `faiss.normalize_L2()` crashes when called on numpy arrays from PyTorch tensors. Worked around by using numpy normalization instead. May not affect Linux/Docker.
- **PubMedBERT requires ~2GB RAM:** The embedding model (~440MB on disk) needs significant memory. Loaded at startup via lifespan handler so it's ready before any HTTP request arrives.
- **Embedder tests use bert-tiny model:** Real PubMedBERT (~440MB) too large for unit tests. Tests use `prajjwal1/bert-tiny` (17MB, 128-dim) — same encoding logic, different weights. Integration tests with real model needed.
- **Scorer tests use mock LLM:** Unit tests mock the AI provider. Integration tests with a real LLM needed to validate prompt quality and parsing against actual LLM outputs.
- **Binary scoring only:** Current scoring is +1/-1 (adherent/non-adherent). No partial adherence score. The paper uses the same binary scheme, but nuanced scoring could improve accuracy.

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
git clone https://github.com/anasraza57/guideline-guard.git
cd guideline-guard

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
