"""
Centralised application configuration.

All configuration is read from environment variables (via .env file).
Pydantic Settings validates types on startup — if a required variable
is missing or has the wrong type, the app fails fast with a clear error.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "GuidelineGuard"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "DEBUG"

    # ── Database ─────────────────────────────────────────────────
    db_host: str = "db"
    db_port: int = 5432
    db_name: str = "msk_audit"
    db_user: str = "msk_user"
    db_password: str = "changeme_in_production"

    @property
    def database_url(self) -> str:
        """Construct the async database URL from individual components."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL (for Alembic migrations)."""
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── AI / LLM Provider ────────────────────────────────────────
    ai_provider: Literal["openai", "anthropic", "local"] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 2048
    openai_temperature: float = 0.0

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ── Vector Search ────────────────────────────────────────────
    faiss_index_path: str = "data/guidelines.index"
    guidelines_csv_path: str = "data/guidelines.csv"

    # ── Medical Embeddings ───────────────────────────────────────
    embedding_model_name: str = "NeuML/pubmedbert-base-embeddings-matryoshka"
    embedding_dimension: int = 768

    # ── Patient Data ─────────────────────────────────────────────
    patient_data_path: str = "data/msk_valid_notes.csv"

    # ── Pipeline Settings ────────────────────────────────────────
    retriever_top_k: int = 5
    max_queries_per_diagnosis: int = 3
    scorer_max_guideline_chars: int = 2000

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures we only read the .env file once.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()
