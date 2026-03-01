"""
Structured logging configuration.

Sets up Python's logging with structured JSON output for production
and human-readable coloured output for development.
"""

import logging
import sys
from typing import Optional

from src.config.settings import get_settings


def setup_logging(level_override: Optional[str] = None) -> None:
    """
    Configure the root logger for the application.

    Args:
        level_override: Override the log level from settings (useful in tests).
    """
    settings = get_settings()
    level = level_override or settings.app_log_level

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set the root level
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if settings.is_production:
        # JSON format for production (machine-parseable)
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        from src.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"patient_id": "abc123"})
    """
    return logging.getLogger(name)
