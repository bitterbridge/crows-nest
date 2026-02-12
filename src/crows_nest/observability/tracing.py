"""OpenTelemetry tracing for Crow's Nest.

Provides distributed tracing for thunk lifecycle events including:
- Thunk creation
- Thunk forcing (execution)
- Thunk completion/failure
- Dependency resolution
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, Tracer

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from uuid import UUID

    from crows_nest.core.config import TracingSettings

# Module-level tracer
_tracer: Tracer | None = None
_initialized: bool = False

P = ParamSpec("P")
T = TypeVar("T")


def setup_tracing(settings: TracingSettings) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        settings: Tracing configuration settings.
    """
    global _tracer, _initialized  # noqa: PLW0603

    if _initialized:
        return

    if not settings.enabled:
        _initialized = True
        return

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": "0.1.0",
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is configured
    if settings.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            # Fall back to console exporter if OTLP not available
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Use console exporter for development
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Get our tracer
    _tracer = trace.get_tracer("crows_nest", "0.1.0")
    _initialized = True


def get_tracer() -> Tracer:
    """Get the configured tracer.

    Returns:
        The OpenTelemetry tracer instance.

    Raises:
        RuntimeError: If tracing has not been initialized.
    """
    if _tracer is None:
        # Return a no-op tracer if not initialized
        return trace.get_tracer("crows_nest", "0.1.0")
    return _tracer


def reset_tracing() -> None:
    """Reset tracing state. Useful for testing."""
    global _tracer, _initialized  # noqa: PLW0603
    _tracer = None
    _initialized = False


@contextmanager
def thunk_span(
    thunk_id: UUID,
    operation: str,
    *,
    parent_id: UUID | None = None,
    capabilities: frozenset[str] | None = None,
) -> Generator[trace.Span, None, None]:
    """Create a span for thunk execution.

    Args:
        thunk_id: The thunk's unique identifier.
        operation: The operation being executed.
        parent_id: Optional parent thunk ID for context.
        capabilities: Optional capabilities granted to the thunk.

    Yields:
        The active span for the thunk execution.
    """
    tracer = get_tracer()

    attributes: dict[str, Any] = {
        "thunk.id": str(thunk_id),
        "thunk.operation": operation,
    }

    if parent_id is not None:
        attributes["thunk.parent_id"] = str(parent_id)

    if capabilities is not None:
        attributes["thunk.capabilities"] = sorted(capabilities)

    with tracer.start_as_current_span(
        f"thunk.force:{operation}",
        attributes=attributes,
    ) as span:
        yield span


def trace_thunk_created(
    thunk_id: UUID,
    operation: str,
    *,
    parent_id: UUID | None = None,
) -> None:
    """Record a thunk creation event.

    Args:
        thunk_id: The thunk's unique identifier.
        operation: The operation name.
        parent_id: Optional parent thunk ID.
    """
    tracer = get_tracer()

    attributes: dict[str, Any] = {
        "thunk.id": str(thunk_id),
        "thunk.operation": operation,
    }

    if parent_id is not None:
        attributes["thunk.parent_id"] = str(parent_id)

    with tracer.start_as_current_span("thunk.created", attributes=attributes):
        pass  # Event is recorded as a span


def trace_thunk_completed(
    thunk_id: UUID,
    operation: str,
    duration_ms: int,
    *,
    span: trace.Span | None = None,
) -> None:
    """Record a thunk completion event.

    Args:
        thunk_id: The thunk's unique identifier.
        operation: The operation name.
        duration_ms: Execution duration in milliseconds.
        span: Optional existing span to update.
    """
    if span is not None:
        span.set_attribute("thunk.duration_ms", duration_ms)
        span.set_status(Status(StatusCode.OK))
    else:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "thunk.completed",
            attributes={
                "thunk.id": str(thunk_id),
                "thunk.operation": operation,
                "thunk.duration_ms": duration_ms,
            },
        ) as new_span:
            new_span.set_status(Status(StatusCode.OK))


def trace_thunk_failed(
    thunk_id: UUID,
    operation: str,
    error_type: str,
    error_message: str,
    *,
    span: trace.Span | None = None,
) -> None:
    """Record a thunk failure event.

    Args:
        thunk_id: The thunk's unique identifier.
        operation: The operation name.
        error_type: The type of error that occurred.
        error_message: The error message.
        span: Optional existing span to update.
    """
    if span is not None:
        span.set_attribute("thunk.error.type", error_type)
        span.set_attribute("thunk.error.message", error_message)
        span.set_status(Status(StatusCode.ERROR, error_message))
    else:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "thunk.failed",
            attributes={
                "thunk.id": str(thunk_id),
                "thunk.operation": operation,
                "thunk.error.type": error_type,
                "thunk.error.message": error_message,
            },
        ) as new_span:
            new_span.set_status(Status(StatusCode.ERROR, error_message))


def traced(
    operation_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace a function.

    Args:
        operation_name: Optional custom name for the span.
            Defaults to the function name.

    Returns:
        A decorator that wraps the function with tracing.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def traced_async(
    operation_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace an async function.

    Args:
        operation_name: Optional custom name for the span.
            Defaults to the function name.

    Returns:
        A decorator that wraps the async function with tracing.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name):
                return await func(*args, **kwargs)  # type: ignore[misc, no-any-return]

        return wrapper  # type: ignore[return-value]

    return decorator
