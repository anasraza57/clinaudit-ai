# PROJECT BIBLE — GuidelineGuard

> **Last Updated:** 2026-03-01
> **Status:** Phase 2 COMPLETE — Next: Phase 3 (Query Agent)

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
4. **Scorer Agent** — Compares documented clinical decisions against retrieved guidelines using an LLM, producing per-diagnosis adherence scores (+1 adherent / -1 non-adherent) and a final aggregate score with explanations.

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
- Mistral-7B → OpenAI API (with provider abstraction)
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
- The `expected_score` for test case 2 has a syntax error (missing value)
- Regex for score parsing is case-sensitive but searches lowercase content — bug
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
- Global mutable state → proper state management
- Truncated guidelines → intelligent chunking with sufficient context
- Manual test cases → automated test suite

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
| `data/guidelines.index` | 4.9 MB | Pre-built FAISS index (768-dim PubMedBERT vectors) |

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
| **Medical Coding** | Rule-based regex + LLM fallback | Two-tier SNOMED categoriser: regex patterns handle 84% of concepts, LLM classifies the remaining 16%. No FHIR server needed. |
| **Configuration** | Pydantic Settings | Type-safe, validates on startup, reads from .env files |
| **Logging** | Python `logging` + `structlog` | Structured JSON logs, correlation IDs, proper levels |
| **Containerisation** | Docker + Docker Compose | Reproducible environments, one-command setup |
| **Testing** | pytest + pytest-asyncio | Standard Python testing, async support |
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

### Phase 3: Query Agent
- [ ] Design query generation prompts
- [ ] Build Query Agent using LLM abstraction
- [ ] Generate targeted guideline search queries from extracted concepts
- [ ] Write tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 4: Retriever Agent
- [ ] Build embedding pipeline (PubMedBERT)
- [ ] Build Retriever Agent with FAISS search
- [ ] Implement intelligent result ranking and filtering
- [ ] Write tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 5: Scorer Agent
- [ ] Design scoring prompts and rubric
- [ ] Build Scorer Agent using LLM abstraction
- [ ] Implement per-diagnosis scoring with explanations
- [ ] Implement aggregate score calculation
- [ ] Write tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 6: Pipeline Integration
- [ ] Build Pipeline orchestrator (chains all 4 agents)
- [ ] Implement single-patient audit endpoint
- [ ] Implement batch audit endpoint (process multiple patients)
- [ ] Implement job tracking (async batch processing)
- [ ] Error handling and retry logic
- [ ] Write integration tests
- [ ] Update learning docs + PROJECT_BIBLE.md

### Phase 7: Validation & Reporting
- [ ] Import gold-standard audit data (120 cases)
- [ ] Run system against gold-standard cases
- [ ] Compare AI scores vs human auditor scores
- [ ] Generate accuracy/agreement metrics
- [ ] Build reporting endpoints (per-patient, aggregate, by-condition)
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
**Reasoning:** The dataset has 1,261 unique concepts with human-readable display names. Regex patterns matching medical keywords and suffixes (-itis, -ectomy, -pathy, -osis) classify 1,069 concepts instantly and for free. The remaining 192 edge cases use the LLM. Each concept is classified once and cached. **Implemented:** `src/services/snomed_categoriser.py`.

---

## 7. Current State Summary

**Date:** 2026-03-01
**Phase 0:** COMPLETE
**Phase 1:** COMPLETE — Database, migrations, data import, vector store, 4327 patients + 1656 guidelines loaded
**Phase 2:** COMPLETE — Extractor Agent with SNOMED categoriser

**What was done in Phase 2:**
- SNOMED Categoriser service: two-tier classification (rule-based 84% + LLM fallback 16%)
- Extractor Agent: groups patient entries by index_date into episodes, categorises each entry
- Data classes: CategorisedEntry, PatientEpisode, ExtractionResult with summary output
- 82 unit tests passing (up from 34 in Phase 1)
- Learning doc: `docs/learning/05-extractor-agent-explained.md`
- DB port changed from 5432→5433 (avoids conflict with local PostgreSQL)
- Data files committed to git (msk_valid_notes.csv, guidelines.csv.gz, guidelines.index)

**Blockers:** None.

**Next session should start with:** Phase 3 — Query Agent
1. Design query generation prompts for guideline search
2. Build Query Agent using AI provider abstraction
3. Generate targeted search queries from extracted diagnoses
4. Write tests
5. Update learning docs + PROJECT_BIBLE.md

---

## 8. Known Issues / Tech Debt

- **Local PostgreSQL port conflict:** Host machine has a native PostgreSQL on port 5432, so our Docker DB uses port 5433. When running Alembic or scripts locally, must set `DB_HOST=localhost DB_PORT=5433`.
- **torch version pinned to 2.2.2:** Python 3.11 doesn't support torch 2.5.1. Will need updating if Python is upgraded.
- **SNOMED categoriser coverage at 84%:** 192 of 1,261 concepts require LLM fallback. Coverage could be improved by adding more patterns, but diminishing returns — LLM handles the rest.

---

## 9. Environment & Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- An OpenAI API key (or alternative LLM provider key)

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

# Start the app
DB_HOST=localhost uvicorn src.main:app --reload

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
