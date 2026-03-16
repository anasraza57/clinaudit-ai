# ClinAuditAI

**Agentic AI for Automated Clinical Guideline Adherence Auditing in Musculoskeletal Primary Care**

## How It Works

ClinAuditAI runs a 4-agent pipeline on each patient's clinical record:

```
Patient Record -> Consultation    -> Audit Query   -> Guideline       -> Compliance      -> Audit Report
                  Insight Agent      Generator        Evidence Finder    Auditor Agent
                       |                 |                  |                 |
                       |                 |                  |                 +-- LLM scores each diagnosis
                       |                 |                  |                    on a 5-level scale (-2 to +2)
                       |                 |                  +-- FAISS + PubMedBERT find relevant NICE guidelines
                       |                 +-- Generates targeted search queries per diagnosis
                       +-- Groups SNOMED-coded entries by episode (diagnoses, treatments, referrals)
```

**Output per patient:**
- Overall adherence score (0.0 to 1.0)
- Per-diagnosis scores: **+2** Compliant, **+1** Partial, **0** Not Relevant, **-1** Non-Compliant, **-2** Risky
- Confidence scores (0.0 to 1.0) and cited NICE guideline text for each judgement
- Missing care opportunities (NICE-recommended actions not documented in the record)
- Explanations with specific guidelines followed / not followed

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Python 3.11+
- An OpenAI API key (for the LLM-based agents), or [Ollama](https://ollama.com/) for local LLM processing
- ~2 GB RAM for PubMedBERT model loading

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/anasraza57/clinaudit-ai.git
cd clinaudit-ai

# 2. Copy environment template and set your API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here

# 3. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start the database (PostgreSQL via Docker)
docker compose up -d db

# 5. Run database migrations
DB_HOST=localhost alembic upgrade head

# 6. Import patient and guideline data into PostgreSQL
DB_HOST=localhost python3 scripts/import_data.py

# 7. Download the PubMedBERT embedding model (~440 MB, one-time)
python3 -c "from transformers import AutoModel, AutoTokenizer; m='NeuML/pubmedbert-base-embeddings-matryoshka'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); print('Model downloaded.')"

# 8. Build the FAISS guideline index (encodes 1,656 guidelines with PubMedBERT)
#    Takes ~5-15 minutes on CPU. One-time cost.
python3 scripts/build_index.py

# 9. Start the application
make run
```

> **Startup note:** The first launch takes 30-60 seconds. The server loads PubMedBERT (~440 MB) and the FAISS index into memory before accepting requests. Watch the terminal for "Embedding model loaded" and "Vector store ready".

> **`DB_HOST=localhost`:** When running locally (outside Docker), the database container is reached via `localhost`. Inside Docker, services use the `db` hostname. The `.env` file defaults to `DB_HOST=db` for Docker, so prefix local commands with `DB_HOST=localhost`.

> **Config-only changes:** If you only changed `.env` variables (no code changes), `docker compose restart app` is faster than a full rebuild.

### Using Ollama (Local LLM)

For processing patient data without sending it to external APIs:

```bash
# 1. Install Ollama: https://ollama.com/
# 2. Pull a model
ollama pull mistral-small

# 3. Update .env
AI_PROVIDER=ollama
OLLAMA_MODEL=mistral-small

# For local dev (no Docker):
OLLAMA_BASE_URL=http://localhost:11434

# For Docker: Ollama runs on the host, not inside the container
OLLAMA_BASE_URL=http://host.docker.internal:11434

# 4. Start the app as normal
make run          # local
make up           # Docker
```

Switch between models by changing `AI_PROVIDER` in `.env` and restarting. Embeddings always use PubMedBERT regardless of the LLM provider.

### Verify It's Running

```bash
# Health check
curl http://localhost:8000/health

# Readiness check (DB + FAISS + PubMedBERT all loaded)
curl http://localhost:8000/health/ready

# Interactive API docs (Swagger UI)
open http://localhost:8000/docs
```

## API Endpoints

All endpoints are documented interactively at **http://localhost:8000/docs** (Swagger UI).

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness check (DB + FAISS + PubMedBERT) |

### Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/data/stats` | Database row counts (patients, entries, guidelines) |
| POST | `/api/v1/data/import/patients` | Import patient records from CSV |
| POST | `/api/v1/data/import/guidelines` | Import guidelines from CSV |

### Audit (Pipeline Execution)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/audit/patient/{pat_id}` | Audit a single patient (synchronous) |
| POST | `/api/v1/audit/batch` | Start a batch audit (`?limit=N`, `?skip_audited=true`) |
| GET | `/api/v1/audit/jobs/{job_id}` | Check batch job status and progress |
| GET | `/api/v1/audit/jobs/{job_id}/results` | Paginated results, sorted by `pat_id` then id (`?page=1&page_size=20`, `?status=failed`) |
| GET | `/api/v1/audit/results/{pat_id}` | All audit results for a specific patient |

### Reports (Analytics)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports/dashboard` | Summary stats: total audited, mean/median score, failure rate (`?job_id=N&model=X`) |
| GET | `/api/v1/reports/conditions` | Per-condition adherence breakdown (`?sort_by=adherence_rate&min_count=3&model=X`) |
| GET | `/api/v1/reports/non-adherent` | Paginated non-adherent cases for clinical review (`?job_id=N&model=X`) |
| GET | `/api/v1/reports/score-distribution` | Score histogram with configurable bins (`?bins=10&model=X`) |

### Exports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports/export/csv` | CSV file (one row per diagnosis per patient, `?model=X`) |
| GET | `/api/v1/reports/export/html` | Self-contained HTML report with inline SVG charts (`?model=X`) |
| GET | `/api/v1/reports/export/comparison-html` | Cross-model comparison report (`?job_a=1&job_b=2` or `?model_a=X&model_b=Y`, `&include_scorer_eval=true`) |

### Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/evaluation/compare` | Compare two batch jobs: Cohen's kappa, Pearson, per-condition deltas (`?job_a=1&job_b=2`) |
| GET | `/api/v1/evaluation/missing-care` | Missing care opportunities aggregated by condition (`?job_id=N&min_count=2`) |
| POST | `/api/v1/evaluation/evaluate/scorer` | LLM-as-Judge scorer quality evaluation (`?model=X&limit=10&offset=0`, deterministic `pat_id` ordering) |
| GET | `/api/v1/evaluation/system-metrics` | Score class distribution, adherence rate, confidence stats (`?job_id=N`) |
| GET | `/api/v1/evaluation/cross-model-metrics` | Confusion matrix, per-class P/R/F1, AUROC, Cohen's kappa (`?job_a=1&job_b=2`) |
| GET | `/api/v1/evaluation/extractor-metrics` | Extractor SNOMED categorisation P/R/F1 (no LLM needed, `?sample_size=500`) |
| POST | `/api/v1/evaluation/evaluate/agents` | Full 4-agent pipeline evaluation with retriever IR metrics (`?limit=5&offset=0`, deterministic `pat_id` ordering, expensive) |

All report and evaluation endpoints accept `?job_id=N` or `?model=X` to scope results to a specific batch run or model. Evaluation endpoints use deterministic `pat_id` sorting with `offset`/`limit` for resumable runs and fair cross-model comparison.

## Usage Guide

### 1. Audit a Single Patient

```bash
curl -X POST http://localhost:8000/api/v1/audit/patient/SOME_PAT_ID
```

Returns the patient's overall adherence score, per-diagnosis 5-level scores with confidence, cited NICE guideline text, and any missing care opportunities.

### 2. Run a Batch Audit

```bash
# Audit all patients
curl -X POST http://localhost:8000/api/v1/audit/batch

# Audit a subset
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"

# Skip already-audited patients (useful for incremental runs)
curl -X POST "http://localhost:8000/api/v1/audit/batch?skip_audited=true"

# Check progress
curl http://localhost:8000/api/v1/audit/jobs/1

# View results (paginated)
curl "http://localhost:8000/api/v1/audit/jobs/1/results?page=1&page_size=20"
```

### 3. View Analytics

```bash
# Dashboard summary
curl http://localhost:8000/api/v1/reports/dashboard

# Which conditions have the worst adherence?
curl "http://localhost:8000/api/v1/reports/conditions?sort_by=adherence_rate"

# Cases flagged as non-adherent (for clinical review)
curl http://localhost:8000/api/v1/reports/non-adherent

# Score distribution histogram
curl http://localhost:8000/api/v1/reports/score-distribution
```

### 4. Export Reports

```bash
# CSV (one row per diagnosis per patient, for Excel/data analysis)
curl -o audit_report.csv http://localhost:8000/api/v1/reports/export/csv

# HTML report (self-contained with inline SVG charts)
curl -o audit_report.html http://localhost:8000/api/v1/reports/export/html

# Scope to a specific job
curl -o report.html "http://localhost:8000/api/v1/reports/export/html?job_id=1"
```

The HTML report includes:
- Dashboard stats (total audited, failure rate, score statistics)
- Inline SVG charts: score distribution histogram, compliance donut chart, per-condition adherence bars
- Per-patient detail cards with 5-level badges, confidence scores, NICE citations, and missing care flags

### 5. Export Charts as PNG

For written reports (Word, LaTeX, etc.):

```bash
# Save charts from all audit data
DB_HOST=localhost python scripts/export_charts.py --output exports/charts

# Scope to a specific batch job with higher DPI for print
DB_HOST=localhost python scripts/export_charts.py --output exports/charts --job-id 1 --dpi 300
```

Produces: `score_distribution.png`, `compliance_breakdown.png`, `condition_adherence.png`.

> **Requires** the `cairo` system library: `brew install cairo` (macOS) or `apt install libcairo2-dev` (Linux).

### 6. Compare Models

Run batch audits with different LLM models, then compare:

```bash
# Step 1: Run with GPT-4o-mini
# Set AI_PROVIDER=openai in .env, start server
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"
# Note the job_id from the response (e.g. 1)

# Step 2: Run with Mistral-Small
# Set AI_PROVIDER=ollama in .env, restart server
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"
# Note the job_id (e.g. 2)

# Step 3: Compare
curl "http://localhost:8000/api/v1/evaluation/compare?job_a=1&job_b=2"

# Detailed classification metrics (confusion matrix, P/R/F1, AUROC)
curl "http://localhost:8000/api/v1/evaluation/cross-model-metrics?job_a=1&job_b=2"
```

### 7. Evaluate Scorer Quality (LLM-as-Judge)

Evaluate the quality of stored scoring results without re-running the pipeline:

```bash
# Evaluate first 10 patients for a specific model (deterministic pat_id order)
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/scorer?model=gpt-4.1-mini&limit=10&offset=0"

# Resume evaluation from where you left off
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/scorer?model=gpt-4.1-mini&limit=10&offset=10"

# Fair comparison: evaluate same 10 patients with different model
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/scorer?model=mistral-small&limit=10&offset=0"
```

Returns reasoning quality, citation accuracy, and score calibration ratings (1-5 each). Uses deterministic `pat_id` ordering so the same `offset`/`limit` evaluates the same patients across different models.

### 8. Full Agent Evaluation

Evaluate all 4 pipeline agents on a sample of patients:

```bash
# Evaluate first 20 patients with a specific model (deterministic pat_id order)
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/agents?model=gpt-4.1-mini&limit=20&offset=0"

# Resume from where you left off
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/agents?model=gpt-4.1-mini&limit=20&offset=20"

# Fair comparison: evaluate same patients with different model
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/agents?model=mistral-small&limit=20&offset=0"
```

Returns extractor P/R/F1, query relevance/coverage, retriever IR metrics (Precision@k, nDCG, MRR), and scorer quality. This is expensive as it runs the full pipeline + LLM judge calls. Uses deterministic `pat_id` ordering for reproducible and resumable evaluation. The `?model=` param dynamically selects the AI provider (gpt-* → OpenAI, else → Ollama) without changing `.env`.

### 9. Generate Comparison Report

After running audits with both models, generate a single self-contained HTML report:

```bash
# Compare by job ID
curl -o comparison.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2"

# Compare by model name (combines all jobs for each model)
curl -o comparison.html "http://localhost:8000/api/v1/reports/export/comparison-html?model_a=gpt-4.1-mini&model_b=mistral-small"

# With LLM-as-Judge scorer evaluation included (adds ~30s per job)
curl -o comparison.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2&include_scorer_eval=true"
```

The report includes: system-level metrics, SVG charts (score distribution, compliance donuts, confusion matrix), cross-model agreement (kappa, Pearson, AUROC, per-class P/R/F1), extractor quality, missing care, and per-patient comparison. Open in any browser, print-friendly, no external dependencies.

## Development

```bash
# Run all tests (371 tests)
make test

# Run tests with coverage
make test-cov

# Run a specific test file
python -m pytest tests/unit/test_scorer.py -v

# Start only the database
docker compose up -d db

# Stop all services
docker compose down

# Reset database (wipe all data and re-run migrations)
docker compose down -v && docker compose up -d db
DB_HOST=localhost alembic upgrade head

# View container logs
docker compose logs -f

# Rebuild Docker image (after code changes)
docker compose up --build -d

# Restart app only (after .env changes, no rebuild needed)
docker compose restart app
```

### Makefile Commands

Run `make help` for all available commands:

| Command | Description |
|---------|-------------|
| `make run` | Run the app locally (outside Docker) |
| `make up` | Start all services with Docker |
| `make down` | Stop all services |
| `make build` | Rebuild Docker images |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make migrate` | Run database migrations |
| `make logs` | Follow app logs |
| `make clean` | Remove containers, volumes, and cache files |
| `make build-index` | Build FAISS guideline index |

## Database

PostgreSQL runs via Docker on port **5433** (mapped from container port 5432).

| Table | Rows | Description |
|-------|------|-------------|
| `patients` | 4,327 | Anonymised MSK patients from the CrossCover trial |
| `clinical_entries` | 21,530 | SNOMED-coded clinical events (diagnoses, treatments, referrals, etc.) |
| `guidelines` | 1,656 | NICE clinical guideline documents |
| `audit_jobs` | varies | Tracks batch audit runs (model used, progress, timing) |
| `audit_results` | varies | Per-patient adherence scores with full JSON details |

### Migrations

```bash
# Apply pending migrations
DB_HOST=localhost alembic upgrade head

# Create a new migration after model changes
DB_HOST=localhost alembic revision --autogenerate -m "description"
```

## Environment Variables

Key variables (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `openai` | LLM provider: `openai`, `ollama`, or `anthropic` |
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `OLLAMA_MODEL` | `mistral-small` | Ollama model to use |
| `DB_HOST` | `db` | Database host (`db` for Docker, `localhost` for local) |
| `DB_PORT` | `5433` | Database port |
| `RETRIEVER_TOP_K` | `5` | Number of guideline chunks to retrieve per query |
| `RETRIEVER_MIN_SIMILARITY` | `1.2` | Max L2 distance threshold for guideline relevance |
| `BATCH_CONCURRENCY` | `5` | Patients processed in parallel (1-2 for Ollama, 5+ for OpenAI) |
| `PIPELINE_PATIENT_TIMEOUT` | `300` | Seconds before a single patient audit times out |
| `OPENAI_REQUEST_TIMEOUT` | `60` | Seconds before a single LLM API call times out |

## Evaluation Results

Cross-model evaluation on the same patients using deterministic `pat_id` ordering.

### Scorer Evaluation — LLM-as-Judge (100 patients, 156 diagnoses, scale 1-5)

| Metric | mistral-small + Ollama Judge | mistral-small + OpenAI Judge | gpt-4.1-mini + Ollama Judge | gpt-4.1-mini + OpenAI Judge |
|---|---|---|---|---|
| Reasoning Quality | 4.58 | 4.37 | 4.75 | 4.77 |
| Citation Accuracy | 3.81 | 3.22 | 4.56 | 4.46 |
| Score Calibration | 4.56 | 4.51 | 4.73 | 4.79 |

### Full Agent Evaluation (50 patients, 88 diagnoses)

| Metric | mistral-small | gpt-4.1-mini |
|---|---|---|
| Extractor match rate | 1.00 | 1.00 |
| Query relevance / coverage | 4.30 / 3.47 | 4.55 / 3.63 |
| Retriever precision@k / nDCG / MRR | 0.38 / 0.65 / 0.60 | 0.57 / 0.82 / 0.79 |
| Scorer reasoning / citation / calibration | 4.61 / 3.86 / 4.60 | 4.78 / 4.32 / 4.77 |

Result files stored in `data/eval_results/`.

## Project Structure

```
clinaudit-ai/
+-- src/                  # Application source code
|   +-- ai/               # LLM provider abstraction (OpenAI, Ollama)
|   +-- agents/           # 4 pipeline agents
|   |   +-- extractor.py  #   ConsultationInsightAgent: SNOMED categorisation
|   |   +-- query.py      #   AuditQueryGenerator: template + LLM queries
|   |   +-- retriever.py  #   GuidelineEvidenceFinder: PubMedBERT + FAISS
|   |   +-- scorer.py     #   ComplianceAuditorAgent: 5-level LLM scoring
|   +-- api/routes/        # FastAPI route handlers
|   |   +-- health.py     #   Health checks
|   |   +-- data.py       #   Data import + stats
|   |   +-- audit.py      #   Pipeline execution (single + batch)
|   |   +-- reports.py    #   Analytics + CSV/HTML exports
|   |   +-- evaluation.py #   Model comparison + LLM-as-Judge
|   +-- config/            # Configuration (Pydantic Settings)
|   +-- models/            # SQLAlchemy database models
|   +-- services/          # Business logic
|       +-- pipeline.py    #   Pipeline orchestrator (chains all 4 agents)
|       +-- reporting.py   #   Analytics (dashboard, conditions, missing care)
|       +-- export.py      #   CSV/HTML report generation + SVG charts
|       +-- comparison.py  #   Model comparison (Cohen's kappa, Pearson)
|       +-- evaluation.py  #   LLM-as-Judge evaluation for all agents
|       +-- embedder.py    #   PubMedBERT embedding service (singleton)
|       +-- vector_store.py #  FAISS index management
|       +-- data_import.py  #  CSV to database import
|       +-- snomed_categoriser.py  # Rule-based + LLM SNOMED classification
+-- tests/                 # Test suite (371 tests)
+-- data/                  # Data files (CSVs, FAISS index)
+-- migrations/            # Alembic database migrations
+-- scripts/               # Utility scripts
|   +-- import_data.py     #   Import CSV data into PostgreSQL
|   +-- build_index.py     #   Build FAISS index from guidelines.csv
|   +-- export_charts.py   #   Export charts as PNG files
+-- exports/               # Generated reports and evaluation results
+-- docker-compose.yml     # PostgreSQL + app services
+-- Dockerfile             # Multi-stage app build
+-- Makefile               # Common commands (make help)
+-- PROJECT_BIBLE.md       # Complete project state, decisions, and roadmap
```

## Documentation

- **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)** -- Single source of truth: architecture, decisions, roadmap, progress
- **[docs/learning/](docs/learning/)** -- Educational docs explaining every component in plain English
