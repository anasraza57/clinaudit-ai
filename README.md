# GuidelineGuard

An agentic AI framework for evaluating musculoskeletal (MSK) consultation adherence to NICE clinical guidelines in primary care.

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An OpenAI API key (or alternative LLM provider key)

### Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env — at minimum, set your OpenAI API key
#    OPENAI_API_KEY=sk-your-key-here

# 3. Copy data files from reference codebases
make setup-data

# 4. Start all services
make up

# 5. Verify it's running
curl http://localhost:8000/health
```

### API Documentation
Once running, visit http://localhost:8000/docs for interactive API documentation.

## Development

```bash
# Run locally (without Docker)
pip install -r requirements.txt
make run

# Run tests
make test

# Run tests with coverage
make test-cov

# View all available commands
make help
```

## Project Structure

```
GuidelineGuard/
├── src/                  # Application source code
│   ├── ai/               # AI/LLM provider abstraction
│   ├── agents/           # The 4 pipeline agents
│   ├── api/              # FastAPI routes and middleware
│   ├── config/           # Configuration management
│   ├── models/           # Database models
│   ├── repositories/     # Data access layer
│   ├── schemas/          # Request/response schemas
│   ├── services/         # Business logic
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── docs/                 # Documentation
│   └── learning/         # Educational documentation
├── data/                 # Data files (not in git)
├── migrations/           # Database migrations
└── scripts/              # Utility scripts
```

## Documentation

- **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)** — Complete project state, decisions, and roadmap
- **[docs/learning/](docs/learning/)** — Educational documentation explaining every concept
