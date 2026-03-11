"""
Application entry point.

Creates and configures the FastAPI application with all routes,
middleware, and startup/shutdown hooks.
"""

import faulthandler
import logging
import os
from contextlib import asynccontextmanager

# Print a Python traceback on segfault instead of silently dying
faulthandler.enable()

# Disable HuggingFace tokenizers Rust-based parallelism — it creates
# internal threads that segfault when combined with uvicorn's async
# event loop and Python thread pools on macOS.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limit PyTorch/OpenMP to 1 thread to prevent threading conflicts
# with the async event loop. CPU inference is fast enough single-threaded
# for our small batch sizes (3 queries at a time).
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.utils.logging import setup_logging
from src.api.routes.health import router as health_router
from src.api.routes.data import router as data_router
from src.api.routes.audit import router as audit_router
from src.api.routes.reports import router as reports_router
from src.api.routes.evaluation import router as evaluation_router


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
    from src.models.database import init_db, close_db
    from src.services.vector_store import get_vector_store
    from src.services.embedder import get_embedder

    try:
        await init_db()
    except Exception as e:
        logger.warning("Database not available on startup: %s", e)

    # Recover any batch jobs left stuck by a previous crash
    try:
        await _recover_stale_jobs(logger)
    except Exception as e:
        logger.warning("Stale job recovery skipped: %s", e)

    try:
        vs = get_vector_store()
        vs.load()
    except Exception as e:
        logger.warning("FAISS index not loaded on startup: %s", e)

    try:
        embedder = get_embedder()
        embedder.load()
    except Exception as e:
        logger.warning("PubMedBERT embedder not loaded on startup: %s", e)

    yield

    # -- Shutdown --
    logger.info("Shutting down %s", settings.app_name)
    await close_db()
    get_vector_store().unload()
    get_embedder().unload()


async def _recover_stale_jobs(logger) -> None:
    """
    Mark any jobs left as 'pending' or 'running' from a previous crash as 'failed'.

    If the server dies mid-batch, jobs get stuck forever. This runs once on
    startup to clean them up so they don't confuse polling clients.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select, or_
    from src.models.audit import AuditJob
    from src.models.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(AuditJob).where(
                or_(AuditJob.status == "pending", AuditJob.status == "running")
            )
        )
        stale_jobs = result.scalars().all()

        if not stale_jobs:
            return

        for job in stale_jobs:
            job.status = "failed"
            job.error_message = "Server restarted before job could complete"
            job.completed_at = datetime.now(timezone.utc)

        await session.commit()
        logger.info(
            "Recovered %d stale batch job(s) from previous crash", len(stale_jobs),
        )


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
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(audit_router, prefix="/api/v1")
    app.include_router(reports_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")

    return app


# The app instance used by uvicorn
app = create_app()
