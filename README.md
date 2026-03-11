# GuidelineGuard

An agentic AI framework for evaluating musculoskeletal (MSK) consultation adherence to NICE clinical guidelines in primary care.

## How It Works

GuidelineGuard runs a 4-agent pipeline on each patient's clinical record:

```
Patient Record → Consultation    → Audit Query   → Guideline       → Compliance      → Audit Report
                 Insight Agent     Generator        Evidence Finder    Auditor Agent
                      │                │                  │                 │
                      │                │                  │                 └─ LLM scores each diagnosis
                      │                │                  │                    on a 5-level scale (-2 to +2)
                      │                │                  └─ FAISS + PubMedBERT find relevant NICE guidelines
                      │                └─ Generates targeted search queries per diagnosis
                      └─ Groups SNOMED-coded entries by episode (diagnoses, treatments, referrals)
```

**Output:** Each patient gets:
- An overall adherence score (0.0–1.0)
- Per-diagnosis scores on a 5-level scale: **+2** Compliant, **+1** Partial, **0** Not Relevant, **-1** Non-Compliant, **-2** Risky
- Confidence scores (0.0–1.0) and cited NICE guideline text for each judgement
- Missing care opportunities — NICE-recommended actions not documented in the record
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
git clone https://github.com/anasraza57/guideline-guard.git
cd guideline-guard

# 2. Copy environment template and set your API key
cp .env.example .env
# Edit .env — at minimum, set: OPENAI_API_KEY=sk-your-key-here

# 3. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start the database
docker compose up -d db

# 5. Run database migrations
DB_HOST=localhost alembic upgrade head

# 6. Import data into PostgreSQL
DB_HOST=localhost python3 scripts/import_data.py

# 7. Download the PubMedBERT embedding model (~440 MB, one-time download)
python3 -c "from transformers import AutoModel, AutoTokenizer; m='NeuML/pubmedbert-base-embeddings-matryoshka'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); print('Model downloaded.')"

# 8. Build the FAISS guideline index (encodes all 1,656 guidelines with PubMedBERT)
#    Takes ~5-15 minutes on CPU — this is a one-time cost
python3 scripts/build_index.py

# 9. Start the application
make run
# (or: DB_HOST=localhost uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload)
```

> **Startup note:** The first launch takes 30–60 seconds — the server loads the PubMedBERT
> embedding model (~440 MB) and the FAISS guideline index into memory before accepting
> requests. Watch the terminal logs for "Embedding model loaded" and "Vector store ready".

> **`DB_HOST=localhost`:** When running locally, the database container is reached via
> `localhost`. Inside Docker, services communicate via the `db` hostname. The `.env` file
> defaults to `DB_HOST=db` (for Docker), so we override it for local commands.

### Using Ollama (Local LLM)

For processing patient data without sending it to external APIs:

```bash
# 1. Install Ollama (https://ollama.com/)
# 2. Pull a model
ollama pull mistral-small

# 3. Set environment variables in .env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=mistral-small

# 4. Start the app as normal — pipeline will use Ollama instead of OpenAI
make run
```

Ollama uses an OpenAI-compatible API, so the same pipeline works with both providers. Switch between them by changing `AI_PROVIDER` in `.env`. Embeddings always use PubMedBERT regardless of the LLM provider.

### Verify It's Running

```bash
# Health check
curl http://localhost:8000/health

# Open interactive API docs
open http://localhost:8000/docs
```

## API Endpoints

Once running, all endpoints are documented interactively at **http://localhost:8000/docs** (Swagger UI).

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness check (DB + FAISS + embedder) |

### Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/data/stats` | Database row counts (patients, entries, guidelines) |
| POST | `/api/v1/data/import/patients` | Import patient records from CSV |
| POST | `/api/v1/data/import/guidelines` | Import guidelines from CSV |

### Audit (Pipeline Execution)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/audit/patient/{pat_id}` | Audit a single patient |
| POST | `/api/v1/audit/batch` | Start a batch audit (`?limit=N`, `?skip_audited=true`) |
| GET | `/api/v1/audit/jobs/{job_id}` | Check batch job status and progress |
| GET | `/api/v1/audit/jobs/{job_id}/results` | Paginated results (`?status=failed`) |
| GET | `/api/v1/audit/results/{pat_id}` | All audit results for a specific patient |

### Reports (Analytics)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports/dashboard` | High-level summary (total audited, score stats, failure rate) |
| GET | `/api/v1/reports/conditions` | Per-condition adherence breakdown |
| GET | `/api/v1/reports/non-adherent` | Paginated list of non-adherent cases for clinical review |
| GET | `/api/v1/reports/score-distribution` | Score histogram (configurable bins) |

### Exports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports/export/csv` | Download CSV file (one row per diagnosis per patient) |
| GET | `/api/v1/reports/export/html` | Self-contained HTML report with inline SVG charts |
| GET | `/api/v1/reports/export/comparison-html` | Cross-model comparison report (`?job_a=1&job_b=2`) — all metrics + charts in one file |

### Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/evaluation/compare` | Compare two batch jobs (`?job_a=1&job_b=2`) — Cohen's kappa, Pearson correlation, per-condition deltas |
| GET | `/api/v1/evaluation/missing-care` | Aggregated missing care opportunities by condition |
| POST | `/api/v1/evaluation/evaluate/scorer/{job_id}` | LLM-as-Judge evaluation of scorer quality |
| GET | `/api/v1/evaluation/system-metrics` | Score class distribution, adherence rate, confidence stats (`?job_id=N`) |
| GET | `/api/v1/evaluation/cross-model-metrics` | Confusion matrix, per-class P/R/F1, AUROC, 5-class kappa (`?job_a=1&job_b=2`) |
| GET | `/api/v1/evaluation/extractor-metrics` | Extractor SNOMED categorisation P/R/F1 (no LLM needed) |
| POST | `/api/v1/evaluation/evaluate/agents` | Full 4-agent evaluation with retriever IR metrics (expensive, runs pipeline) |

All report and evaluation endpoints accept an optional `?job_id=N` query parameter to scope results to a specific batch run.

## Running Audits & Viewing Results

### 1. Audit a Single Patient

```bash
# Pick a patient ID from the database
curl -X POST http://localhost:8000/api/v1/audit/patient/SOME_PAT_ID
```

The response includes the patient's overall adherence score, per-diagnosis 5-level scores with confidence, cited NICE guideline text, and any missing care opportunities.

### 2. Run a Batch Audit

```bash
# Audit all 4,327 patients (takes a while — each patient calls the LLM)
curl -X POST http://localhost:8000/api/v1/audit/batch

# Audit a subset (e.g. 50 patients)
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"

# Skip patients that have already been audited (incremental batching)
curl -X POST "http://localhost:8000/api/v1/audit/batch?skip_audited=true"

# Check progress
curl http://localhost:8000/api/v1/audit/jobs/1

# View results (paginated)
curl "http://localhost:8000/api/v1/audit/jobs/1/results?page=1&page_size=20"
```

### 3. View Analytics

After running audits, the reporting endpoints aggregate the results:

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
# Download CSV (one row per diagnosis per patient — for Excel/data analysis)
curl -o audit_report.csv http://localhost:8000/api/v1/reports/export/csv

# Download HTML report (self-contained, includes inline SVG charts)
curl -o audit_report.html http://localhost:8000/api/v1/reports/export/html
```

The HTML report includes:
- Dashboard stats (total audited, failure rate, score statistics)
- Inline SVG charts: score distribution histogram, compliance donut chart, per-condition adherence bars
- Per-patient detail cards with 5-level badges, confidence scores, NICE citations, and missing care flags

### 5. Export Charts as PNG

For use in written reports (Word, LaTeX, etc.):

```bash
# Save charts from all audit data
DB_HOST=localhost python scripts/export_charts.py --output exports/charts

# Scope to a specific batch job, higher DPI for print
DB_HOST=localhost python scripts/export_charts.py --output exports/charts --job-id 1 --dpi 300
```

This produces three PNG files: `score_distribution.png`, `compliance_breakdown.png`, `condition_adherence.png`.

> **Note:** PNG export requires the `cairo` system library (`brew install cairo` on macOS, `apt install libcairo2-dev` on Linux).

### 6. Compare Models

Run batch audits with different LLM providers, then compare:

```bash
# Run with OpenAI (job_id=1)
# AI_PROVIDER=openai in .env
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"

# Run with Ollama (job_id=2)
# AI_PROVIDER=ollama in .env, restart server
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"

# Compare the two jobs
curl "http://localhost:8000/api/v1/evaluation/compare?job_a=1&job_b=2"
```

The comparison includes Cohen's kappa (inter-rater agreement), Pearson correlation, and per-condition adherence deltas.

### 7. Evaluate Scorer Quality

Use LLM-as-Judge to evaluate the quality of stored scoring results:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/scorer/1?limit=10"
```

### 8. Generate Comparison Report

After running audits with both providers, generate a single HTML report with all metrics:

```bash
# Self-contained HTML — open in any browser, print-friendly, no dependencies
curl -o comparison.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2"
```

The comparison report includes: system-level metrics, SVG charts (score distribution, compliance donuts, confusion matrix), cross-model agreement (kappa, Pearson, AUROC, per-class P/R/F1), extractor quality, missing care opportunities, and per-patient comparison.

Optionally include LLM-as-Judge scorer evaluation inline (adds ~30s per job):

```bash
curl -o comparison.html "http://localhost:8000/api/v1/reports/export/comparison-html?job_a=1&job_b=2&include_scorer_eval=true"
```

Or use the **Swagger UI** at http://localhost:8000/docs to explore all endpoints interactively.

## Development

```bash
# Run all tests (371 tests)
make test

# Run tests with coverage
make test-cov

# Start only the database
docker compose up -d db

# Stop all services
docker compose down

# Reset database (wipe all data and start fresh)
docker compose down -v && docker compose up -d db
DB_HOST=localhost alembic upgrade head

# View logs
docker compose logs -f
```

## Database

GuidelineGuard uses PostgreSQL (runs via Docker on port 5433).

| Table | Rows | Description |
|-------|------|-------------|
| `patients` | 4,327 | Anonymised MSK patients from the CrossCover trial |
| `clinical_entries` | 21,530 | SNOMED-coded clinical events (diagnoses, treatments, referrals, etc.) |
| `guidelines` | 1,656 | NICE clinical guideline documents |
| `audit_jobs` | — | Tracks batch audit processing runs (includes LLM provider used) |
| `audit_results` | — | Per-patient guideline adherence scores with full JSON details |

### Migrations

```bash
# Apply migrations
DB_HOST=localhost alembic upgrade head

# Create a new migration after model changes
DB_HOST=localhost alembic revision --autogenerate -m "description"
```

## Project Structure

```
guideline-guard/
├── src/                  # Application source code
│   ├── ai/               # AI/LLM provider abstraction (OpenAI, Ollama)
│   ├── agents/           # 4 pipeline agents
│   │   ├── extractor.py  #   ConsultationInsightAgent — SNOMED categorisation
│   │   ├── query.py      #   AuditQueryGenerator — template + LLM queries
│   │   ├── retriever.py  #   GuidelineEvidenceFinder — PubMedBERT + FAISS
│   │   └── scorer.py     #   ComplianceAuditorAgent — 5-level LLM scoring
│   ├── api/routes/       # FastAPI route handlers
│   │   ├── health.py     #   Health checks
│   │   ├── data.py       #   Data import + stats
│   │   ├── audit.py      #   Pipeline execution (single + batch)
│   │   ├── reports.py    #   Analytics + CSV/HTML exports
│   │   └── evaluation.py #   Model comparison + LLM-as-Judge
│   ├── config/           # Configuration management (Pydantic Settings)
│   ├── models/           # SQLAlchemy database models
│   └── services/         # Business logic
│       ├── pipeline.py   #   Pipeline orchestrator (chains all 4 agents)
│       ├── reporting.py  #   Analytics aggregation (dashboard, conditions, missing care)
│       ├── export.py     #   CSV/HTML report generation + SVG charts + PNG export
│       ├── comparison.py #   Model comparison (Cohen's kappa, Pearson, deltas)
│       ├── evaluation.py #   LLM-as-Judge evaluation for all agents
│       ├── embedder.py   #   PubMedBERT embedding service (singleton)
│       ├── vector_store.py #  FAISS index management
│       ├── data_import.py  #  CSV → database import
│       └── snomed_categoriser.py  # Rule-based + LLM SNOMED classification
├── tests/                # Test suite (371 tests)
├── data/                 # Data files — CSVs, FAISS index
├── migrations/           # Alembic database migrations
├── scripts/              # Utility scripts
│   ├── import_data.py    #   Import CSV data into PostgreSQL
│   ├── build_index.py    #   Build FAISS index from guidelines.csv
│   └── export_charts.py  #   Export charts as PNG files
├── docker-compose.yml    # PostgreSQL + app services
├── Dockerfile            # Multi-stage app build
├── Makefile              # Common commands (make help for full list)
└── PROJECT_BIBLE.md      # Complete project state, decisions, and roadmap
```

## Documentation

- **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)** — Single source of truth: analysis, architecture, decisions, roadmap, progress
- **[docs/learning/](docs/learning/)** — Educational docs explaining every concept in plain English
