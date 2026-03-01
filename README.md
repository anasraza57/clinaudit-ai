# GuidelineGuard

An agentic AI framework for evaluating musculoskeletal (MSK) consultation adherence to NICE clinical guidelines in primary care.

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Python 3.11+ (for local development)
- An OpenAI API key (or alternative LLM provider key)

### Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env — at minimum, set your OpenAI API key
#    OPENAI_API_KEY=sk-your-key-here

# 3. Start the database
docker compose up -d db

# 4. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Run database migrations
DB_HOST=localhost alembic upgrade head

# 6. Decompress guidelines (optional — import script handles .gz automatically)
gunzip -k data/guidelines.csv.gz

# 7. Import data into PostgreSQL
DB_HOST=localhost python3 scripts/import_data.py

# 7. Start the app
make up
# or run locally:
DB_HOST=localhost uvicorn src.main:app --reload

# 8. Verify it's running
curl http://localhost:8000/health
```

> **Note:** We use `DB_HOST=localhost` when running commands locally because the
> database container is accessed via `localhost` from the host machine. Inside
> Docker, services communicate via the `db` hostname.

### API Documentation
Once running, visit http://localhost:8000/docs for interactive API documentation.

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Start only the database
docker compose up -d db

# Stop all services
docker compose down

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
| `audit_jobs` | — | Tracks batch audit processing runs |
| `audit_results` | — | Per-patient guideline adherence scores |

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
│   ├── ai/               # AI/LLM provider abstraction (Strategy Pattern)
│   ├── agents/           # The 4 pipeline agents (Extractor, Query, Retriever, Scorer)
│   ├── api/routes/       # FastAPI route handlers
│   ├── config/           # Configuration management (Pydantic Settings)
│   ├── models/           # SQLAlchemy database models
│   ├── services/         # Business logic (data import, vector store)
│   └── utils/            # Shared utilities (logging)
├── tests/                # Test suite (34 tests)
├── docs/learning/        # Educational documentation
├── data/                 # Data files — CSVs, FAISS index (not in git)
├── migrations/           # Alembic database migrations
├── docker-compose.yml    # PostgreSQL + app services
├── Dockerfile            # Multi-stage app build
├── Makefile              # Common commands
└── PROJECT_BIBLE.md      # Complete project state, decisions, and roadmap
```

## Documentation

- **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)** — Single source of truth: analysis, architecture, decisions, roadmap, progress
- **[docs/learning/](docs/learning/)** — Educational docs explaining every concept in plain English:
  - `00-glossary.md` — Key terms (SNOMED, FAISS, PubMedBERT, RAG, etc.)
  - `01-project-overview.md` — Problem, solution, data, pipeline
  - `02-architecture-explained.md` — Tech stack decisions and architecture
  - `03-data-layer-explained.md` — Database, ORM, migrations, vector search
