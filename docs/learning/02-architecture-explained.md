# Architecture Explained — How the System is Designed

## Why Architecture Matters

Imagine building a house without a blueprint. You might get the kitchen done, but then realise the plumbing doesn't reach it, or the walls aren't strong enough for a second floor. Software architecture is our blueprint — it defines how all the pieces fit together before we start building.

## Our Architecture: Layered + Pipeline

Our system has two key architectural patterns working together:

### Pattern 1: The 4-Agent Pipeline

```
Patient Data
    │
    ▼
┌──────────────────┐
│  Extractor Agent  │  "What happened in this consultation?"
└────────┬─────────┘
         │ Diagnoses, Treatments, Procedures, Referrals
         ▼
┌──────────────────┐
│   Query Agent     │  "What guidelines should we look up?"
└────────┬─────────┘
         │ Search Queries
         ▼
┌──────────────────┐
│ Retriever Agent   │  "Here are the relevant guideline passages"
└────────┬─────────┘
         │ Guideline Text Chunks
         ▼
┌──────────────────┐
│  Scorer Agent     │  "Does the care match the guidelines?"
└────────┬─────────┘
         │
         ▼
   Audit Report (scores + explanations)
```

Each agent has ONE job. This is the **Single Responsibility Principle** — if we need to change how scoring works, we only touch the Scorer. If we need better guidelines retrieval, we only touch the Retriever. Nothing else breaks.

### Pattern 2: Layered Application Architecture

```
┌─────────────────────────────┐
│         API Layer            │  FastAPI routes — handles HTTP requests
│   (src/api/)                 │
├─────────────────────────────┤
│       Service Layer          │  Business logic — orchestrates the pipeline
│   (src/services/)            │
├─────────────────────────────┤
│        Agent Layer           │  The 4 agents — each does its specialised work
│   (src/agents/)              │
├─────────────────────────────┤
│    Infrastructure Layer      │  Database, AI providers, FAISS, config
│   (src/ai/, src/models/,     │
│    src/repositories/)        │
└─────────────────────────────┘
```

Each layer only talks to the layer directly below it. The API layer never directly queries the database or calls the AI. It asks the Service layer, which coordinates everything.

## Key Components Explained

### Configuration System (`src/config/`)
**What:** A single place where ALL settings live — database credentials, API keys, model names, etc.
**How:** Pydantic Settings reads from a `.env` file and validates every value on startup.
**Why this way:** Hardcoding settings (like Hiruni did with Windows paths) breaks when moving between machines. Environment variables work everywhere — on your laptop, in Docker, on a server.

### AI Abstraction Layer (`src/ai/`)
**What:** A pluggable system for swapping AI providers without changing application code.
**How:** We define an abstract interface (AIProvider) with methods like `chat()` and `embed()`. Then we implement it for OpenAI. Tomorrow we could implement it for Anthropic or a local model.
**Why this way:** Cyprian's code directly imported and called `ChatOpenAI`. If you wanted to switch to Claude, you'd have to find and change every file that uses it. With our pattern, you change ONE environment variable.

### Database Layer (`src/models/`, `src/repositories/`)
**What:** PostgreSQL stores patient data, audit results, and job tracking.
**Why not just CSV files?** CSVs can't handle multiple users querying at once, can't enforce data integrity, can't do complex queries efficiently, and don't scale.

### Vector Search (`data/` + FAISS)
**What:** The pre-built FAISS index over NICE guidelines, loaded into memory for fast similarity search.
**How:** Guidelines are converted to 768-dimensional vectors using PubMedBERT. Queries are converted the same way. FAISS finds the closest matches.

## Why We Made These Choices

| Decision | Why | Alternative Rejected |
|----------|-----|---------------------|
| FastAPI (not Flask) | Auto-docs, validation, async, type safety | Flask — too minimal, Cyprian's Flask code was fragile |
| PostgreSQL (not SQLite) | Concurrent access, production-ready | SQLite — file-based, no concurrency |
| Custom pipeline (not LangGraph) | Simpler, testable, no heavy dependency | LangGraph — overkill for a linear 4-step flow |
| Pydantic Settings (not os.getenv) | Type validation, .env support, fails fast on bad config | Scattered os.getenv() — no validation, easy to miss |
| Docker Compose | One-command setup, reproducible | Manual install — "works on my machine" problems |
