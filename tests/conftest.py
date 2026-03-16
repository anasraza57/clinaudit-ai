"""
Shared test fixtures and configuration.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Override settings BEFORE importing anything from src
os.environ["APP_ENV"] = "development"
os.environ["APP_DEBUG"] = "true"
os.environ["AI_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key-not-real"
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5432"
os.environ["DB_NAME"] = "clinaudit_ai_test"
os.environ["DB_USER"] = "gg_user"
os.environ["DB_PASSWORD"] = "testpassword"


@pytest.fixture()
def client():
    """Create a test client for the FastAPI app."""
    # Clear cached settings so test env vars take effect
    from src.config.settings import get_settings
    get_settings.cache_clear()

    from src.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c

    # Clean up
    get_settings.cache_clear()
