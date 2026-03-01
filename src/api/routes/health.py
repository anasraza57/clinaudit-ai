"""
Health check endpoints.

These allow monitoring systems (and humans) to verify the
application is running and its dependencies are accessible.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Basic health check — confirms the API is running.

    Returns:
        Status, environment, and timestamp.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready")
async def readiness_check() -> dict:
    """
    Readiness check — confirms the API and its dependencies are ready.

    This will be expanded as we add database connections, FAISS index
    loading, etc. For now it checks basic configuration validity.
    """
    settings = get_settings()
    checks: dict[str, str] = {}

    # Check AI provider configuration
    if settings.ai_provider == "openai" and settings.openai_api_key:
        checks["ai_provider"] = "configured"
    elif settings.ai_provider == "openai":
        checks["ai_provider"] = "missing_api_key"
    else:
        checks["ai_provider"] = f"configured ({settings.ai_provider})"

    # Overall status
    all_ok = all(v != "missing_api_key" for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
