# =============================================================================
# MSK Clinical Audit System — Makefile
# =============================================================================
# Common commands for development.
#
# Usage:
#   make help        Show available commands
#   make up          Start all services
#   make down        Stop all services
#   make test        Run tests
#   make logs        Follow app logs

.PHONY: help up down build logs test lint migrate shell clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# -- Docker Commands --

up: ## Start all services (with build)
	docker compose up --build -d

down: ## Stop all services
	docker compose down

build: ## Rebuild Docker images
	docker compose build --no-cache

logs: ## Follow application logs
	docker compose logs -f app

logs-db: ## Follow database logs
	docker compose logs -f db

# -- Development --

run: ## Run the app locally (without Docker)
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# -- Database --

migrate: ## Run database migrations
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add users table")
	alembic revision --autogenerate -m "$(MSG)"

# -- Utilities --

shell: ## Open a shell in the app container
	docker compose exec app bash

clean: ## Remove all containers, volumes, and cached files
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

setup-data: ## Copy reference data files into data/ directory
	@echo "Copying data files from reference codebases..."
	cp -n "../Cleaned Data/msk_valid_notes.csv" data/ 2>/dev/null || true
	cp -n "../Cyprian/guidelines.csv" data/ 2>/dev/null || true
	cp -n "../Cyprian/guidelines.index" data/ 2>/dev/null || true
	cp -n "../Cyprian/new_guidelines.index" data/ 2>/dev/null || true
	@echo "Done. Check data/ directory."
