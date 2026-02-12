"""Structured JSON logging for Crow's Nest.

Provides structured logging using structlog with JSON output
for production and pretty-printed output for development.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from crows_nest.core.config import GeneralSettings

_configured: bool = False


def setup_logging(settings: GeneralSettings) -> None:
    """Configure structured logging.

    Args:
        settings: General application settings including log level and format.
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return

    # Set up standard library logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.json_logs:
        # Production: JSON output
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Development: Pretty console output
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    _configured = True


def reset_logging() -> None:
    """Reset logging configuration. Useful for testing."""
    global _configured  # noqa: PLW0603
    _configured = False
    structlog.reset_defaults()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger.

    Args:
        name: Optional logger name. Defaults to the calling module.

    Returns:
        A bound structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: object) -> None:
    """Bind key-value pairs to the logging context.

    These values will be included in all subsequent log entries
    from the current context (e.g., within a request handler).

    Args:
        **kwargs: Key-value pairs to bind to the context.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the logging context.

    Args:
        *keys: Keys to remove from the context.
    """
    structlog.contextvars.unbind_contextvars(*keys)
