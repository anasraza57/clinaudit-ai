"""
Database engine and session management.

Provides async database connectivity using SQLAlchemy 2.0's
async engine and session patterns.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# These are initialised lazily on first use (via init_db)
_engine = None
_session_factory = None


def get_engine():
    """Return the async engine, creating it if needed."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.app_debug and not settings.is_production,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        logger.info(
            "Database engine created (host=%s, db=%s)",
            settings.db_host,
            settings.db_name,
        )
    return _engine


def get_session_factory():
    """Return the async session factory, creating it if needed."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that yields an async database session.

    Usage in FastAPI:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Verify database connectivity on startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )
    logger.info("Database connection verified")


async def close_db() -> None:
    """Close the database engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection closed")
