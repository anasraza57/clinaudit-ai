"""
Application entry point.

Creates and configures the FastAPI application with all routes,
middleware, and startup/shutdown hooks.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.utils.logging import setup_logging
from src.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.
    """
    logger = logging.getLogger(__name__)
    settings = get_settings()

    logger.info(
        "Starting %s (env=%s, debug=%s)",
        settings.app_name,
        settings.app_env,
        settings.app_debug,
    )

    # -- Startup --
    # Future: initialise database connection pool
    # Future: load FAISS index into memory
    # Future: initialise AI provider

    yield

    # -- Shutdown --
    logger.info("Shutting down %s", settings.app_name)
    # Future: close database connections
    # Future: cleanup resources


def create_app() -> FastAPI:
    """
    Application factory.

    Creates a configured FastAPI instance. Using a factory function
    (instead of a global app variable) allows creating separate
    instances for testing.
    """
    # Initialise logging first — everything else depends on it
    setup_logging()

    settings = get_settings()

    app = FastAPI(
        title="GuidelineGuard API",
        description=(
            "An agentic AI framework for evaluating MSK consultation "
            "adherence to NICE clinical guidelines in primary care."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # -- CORS Middleware --
    if settings.is_production:
        # In production, restrict to known origins
        origins = []  # Configure via env var when deploying
    else:
        # In development, allow all origins
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Register Routers --
    app.include_router(health_router)
    # Future: app.include_router(audit_router, prefix="/api/v1")
    # Future: app.include_router(patients_router, prefix="/api/v1")

    return app


# The app instance used by uvicorn
app = create_app()
