"""Tests for configuration system."""

import os


def test_settings_loads_defaults():
    """Settings should load with sensible defaults."""
    from src.config.settings import get_settings
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.app_name == "GuidelineGuard"
    assert settings.app_port == 8000
    assert settings.ai_provider == "openai"
    assert settings.retriever_top_k == 5
    assert settings.embedding_dimension == 768

    get_settings.cache_clear()


def test_database_url_construction():
    """Database URL should be built from individual components."""
    from src.config.settings import get_settings
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.db_user in settings.database_url
    assert settings.db_name in settings.database_url
    assert "asyncpg" in settings.database_url
    assert "psycopg2" in settings.database_url_sync

    get_settings.cache_clear()


def test_is_production():
    """is_production property should reflect app_env."""
    from src.config.settings import get_settings
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.is_production is False  # test env is 'development'

    get_settings.cache_clear()
