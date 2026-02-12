"""Tests for observability (tracing and logging)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.core.config import GeneralSettings, TracingSettings
from crows_nest.observability import (
    bind_context,
    clear_context,
    get_logger,
    get_tracer,
    reset_logging,
    reset_tracing,
    setup_logging,
    setup_tracing,
    thunk_span,
    trace_thunk_completed,
    trace_thunk_created,
    trace_thunk_failed,
    traced,
    traced_async,
    unbind_context,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_observability() -> Generator[None, None, None]:
    """Reset observability state before and after each test."""
    reset_tracing()
    reset_logging()
    yield
    reset_tracing()
    reset_logging()


class TestTracingSetup:
    """Tests for tracing initialization."""

    def test_setup_tracing_disabled(self) -> None:
        """Tracing setup with disabled setting does not configure."""
        settings = TracingSettings(enabled=False)
        setup_tracing(settings)

        # Should return a no-op tracer
        tracer = get_tracer()
        assert tracer is not None

    def test_setup_tracing_enabled(self) -> None:
        """Tracing setup with enabled setting configures provider."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        tracer = get_tracer()
        assert tracer is not None

    def test_setup_tracing_idempotent(self) -> None:
        """Multiple setup calls are idempotent."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)
        setup_tracing(settings)  # Should not raise

        tracer = get_tracer()
        assert tracer is not None

    def test_reset_tracing(self) -> None:
        """Reset tracing clears state."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)
        reset_tracing()

        # After reset, should be able to setup again
        setup_tracing(settings)


class TestThunkSpan:
    """Tests for thunk span context manager."""

    def test_thunk_span_creates_span(self) -> None:
        """thunk_span creates a span with attributes."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        with thunk_span(thunk_id, "test.operation") as span:
            assert span is not None
            # Span should have attributes set

    def test_thunk_span_with_parent(self) -> None:
        """thunk_span can include parent ID."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        parent_id = uuid4()
        with thunk_span(thunk_id, "test.operation", parent_id=parent_id) as span:
            assert span is not None

    def test_thunk_span_with_capabilities(self) -> None:
        """thunk_span can include capabilities."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        caps = frozenset(["cap.a", "cap.b"])
        with thunk_span(thunk_id, "test.operation", capabilities=caps) as span:
            assert span is not None


class TestTracingEvents:
    """Tests for tracing event functions."""

    def test_trace_thunk_created(self) -> None:
        """trace_thunk_created records event."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        # Should not raise
        trace_thunk_created(thunk_id, "test.operation")

    def test_trace_thunk_created_with_parent(self) -> None:
        """trace_thunk_created can include parent ID."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        parent_id = uuid4()
        trace_thunk_created(thunk_id, "test.operation", parent_id=parent_id)

    def test_trace_thunk_completed(self) -> None:
        """trace_thunk_completed records success."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        trace_thunk_completed(thunk_id, "test.operation", duration_ms=100)

    def test_trace_thunk_completed_with_span(self) -> None:
        """trace_thunk_completed can update existing span."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        with thunk_span(thunk_id, "test.operation") as span:
            trace_thunk_completed(thunk_id, "test.operation", duration_ms=100, span=span)

    def test_trace_thunk_failed(self) -> None:
        """trace_thunk_failed records failure."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        trace_thunk_failed(thunk_id, "test.operation", "TestError", "Something went wrong")

    def test_trace_thunk_failed_with_span(self) -> None:
        """trace_thunk_failed can update existing span."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        thunk_id = uuid4()
        with thunk_span(thunk_id, "test.operation") as span:
            trace_thunk_failed(
                thunk_id,
                "test.operation",
                "TestError",
                "Something went wrong",
                span=span,
            )


class TestTracedDecorators:
    """Tests for the traced decorators."""

    def test_traced_sync_function(self) -> None:
        """@traced decorator works for sync functions."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        @traced()
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_traced_sync_with_custom_name(self) -> None:
        """@traced decorator can use custom span name."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        @traced("custom.operation")
        def my_function() -> str:
            return "hello"

        result = my_function()
        assert result == "hello"

    async def test_traced_async_function(self) -> None:
        """@traced_async decorator works for async functions."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        @traced_async()
        async def my_async_function(x: int) -> int:
            return x * 2

        result = await my_async_function(5)
        assert result == 10

    async def test_traced_async_with_custom_name(self) -> None:
        """@traced_async decorator can use custom span name."""
        settings = TracingSettings(enabled=True, otlp_endpoint="")
        setup_tracing(settings)

        @traced_async("custom.async.operation")
        async def my_async_function() -> str:
            return "hello"

        result = await my_async_function()
        assert result == "hello"


class TestLoggingSetup:
    """Tests for logging initialization."""

    def test_setup_logging_console(self) -> None:
        """Logging setup with json_logs=False uses console renderer."""
        settings = GeneralSettings(log_level="INFO", json_logs=False)
        setup_logging(settings)

        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_json(self) -> None:
        """Logging setup with json_logs=True uses JSON renderer."""
        settings = GeneralSettings(log_level="INFO", json_logs=True)
        setup_logging(settings)

        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_idempotent(self) -> None:
        """Multiple setup calls are idempotent."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)
        setup_logging(settings)  # Should not raise

    def test_setup_logging_custom_level(self) -> None:
        """Logging respects log level setting."""
        settings = GeneralSettings(log_level="DEBUG")
        setup_logging(settings)

        logger = get_logger("test")
        assert logger is not None

    def test_reset_logging(self) -> None:
        """Reset logging clears state."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)
        reset_logging()

        # After reset, should be able to setup again
        setup_logging(settings)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_without_name(self) -> None:
        """get_logger works without a name."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)

        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """get_logger works with a specific name."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)

        logger = get_logger("my.module")
        assert logger is not None


class TestLoggingContext:
    """Tests for logging context management."""

    def test_bind_context(self) -> None:
        """bind_context adds values to context."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)

        bind_context(request_id="123", user="test")
        # Context is bound (verified by not raising)

    def test_unbind_context(self) -> None:
        """unbind_context removes values from context."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)

        bind_context(request_id="123")
        unbind_context("request_id")
        # Context is unbound (verified by not raising)

    def test_clear_context(self) -> None:
        """clear_context removes all values."""
        settings = GeneralSettings(log_level="INFO")
        setup_logging(settings)

        bind_context(a="1", b="2", c="3")
        clear_context()
        # All context is cleared (verified by not raising)
