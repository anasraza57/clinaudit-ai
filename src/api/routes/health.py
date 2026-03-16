"""
Health check endpoints.

These allow monitoring systems (and humans) to verify the
application is running and its dependencies are accessible.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


# ── Response schemas ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    timestamp: str

    model_config = {"json_schema_extra": {"examples": [
        {"status": "healthy", "service": "ClinAuditAI", "environment": "development", "timestamp": "2026-03-02T12:00:00+00:00"}
    ]}}


class ReadinessCheck(BaseModel):
    ai_provider: str


class ReadinessResponse(BaseModel):
    status: str
    checks: ReadinessCheck
    timestamp: str


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> dict:
    """
    Basic liveness check — confirms the API process is running.

    Returns the service name, environment, and current timestamp.
    This endpoint does not check database or model availability
    (use `/health/ready` for that).
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness check")
async def readiness_check() -> dict:
    """
    Readiness check — confirms the API and its dependencies are ready
    to handle requests.

    Checks AI provider configuration. Returns "ready" if all checks
    pass, or "degraded" if any dependency is unavailable.
    """
    settings = get_settings()
    checks: dict[str, str] = {}

    # Check AI provider configuration — each provider has different requirements
    provider = settings.ai_provider
    _KEY_REQUIRED: dict[str, str] = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
    }

    if provider in _KEY_REQUIRED:
        key_attr = _KEY_REQUIRED[provider]
        if getattr(settings, key_attr, ""):
            checks["ai_provider"] = f"configured ({provider})"
        else:
            checks["ai_provider"] = f"missing_api_key ({provider})"
    else:
        # Providers without API keys (ollama, local)
        checks["ai_provider"] = f"configured ({provider})"

    # Overall status
    all_ok = all("missing_api_key" not in v for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
