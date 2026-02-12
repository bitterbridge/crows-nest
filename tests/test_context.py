"""Tests for thunk context, cancellation, and progress reporting."""

from __future__ import annotations

import asyncio

import pytest

from crows_nest.core.context import (
    CallbackProgressReporter,
    CancellationError,
    CancellationToken,
    NullProgressReporter,
    ProgressEvent,
    QueueProgressReporter,
    ThunkContext,
)


class TestCancellationToken:
    """Tests for CancellationToken."""

    @pytest.mark.asyncio
    async def test_initial_state_not_cancelled(self):
        """Token should not be cancelled initially."""
        token = CancellationToken()
        assert token.is_cancelled is False
        assert token.reason is None

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self):
        """Cancellation should set the flag and reason."""
        token = CancellationToken()
        await token.cancel("User requested")
        assert token.is_cancelled is True
        assert token.reason == "User requested"

    @pytest.mark.asyncio
    async def test_cancel_sync(self):
        """Synchronous cancellation should work."""
        token = CancellationToken()
        token.cancel_sync("Sync cancel")
        assert token.is_cancelled is True
        assert token.reason == "Sync cancel"

    @pytest.mark.asyncio
    async def test_raise_if_cancelled(self):
        """Should raise CancellationError when cancelled."""
        token = CancellationToken()
        token.raise_if_cancelled()  # Should not raise

        await token.cancel("test")
        with pytest.raises(CancellationError) as exc_info:
            token.raise_if_cancelled()
        assert "test" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_callback(self):
        """Cancellation should invoke registered callbacks."""
        token = CancellationToken()
        callback_called = []

        token.on_cancel(lambda: callback_called.append(True))
        assert callback_called == []

        await token.cancel()
        assert callback_called == [True]

    @pytest.mark.asyncio
    async def test_cancel_callback_already_cancelled(self):
        """Callback registered after cancellation should be called immediately."""
        token = CancellationToken()
        await token.cancel()

        callback_called = []
        token.on_cancel(lambda: callback_called.append(True))
        assert callback_called == [True]

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self):
        """Multiple cancellations should have no additional effect."""
        token = CancellationToken()
        await token.cancel("first")
        await token.cancel("second")
        assert token.reason == "first"  # First reason preserved

    @pytest.mark.asyncio
    async def test_child_token(self):
        """Child token should cancel when parent cancels."""
        parent = CancellationToken()
        child = parent.child()

        assert child.is_cancelled is False
        await parent.cancel("parent cancelled")

        assert parent.is_cancelled is True
        assert child.is_cancelled is True

    @pytest.mark.asyncio
    async def test_child_token_independent_cancel(self):
        """Child can be cancelled independently of parent."""
        parent = CancellationToken()
        child = parent.child()

        child.cancel_sync("child only")

        assert child.is_cancelled is True
        assert parent.is_cancelled is False

    @pytest.mark.asyncio
    async def test_child_of_cancelled_parent(self):
        """Child of cancelled parent should be cancelled immediately."""
        parent = CancellationToken()
        await parent.cancel("already done")

        child = parent.child()
        assert child.is_cancelled is True


class TestProgressReporter:
    """Tests for progress reporters."""

    @pytest.mark.asyncio
    async def test_callback_reporter(self):
        """CallbackProgressReporter should invoke callback with events."""
        events: list[ProgressEvent] = []

        reporter = CallbackProgressReporter(events.append)
        await reporter.report_progress(0.5, "Halfway")
        await reporter.emit_partial("chunk", {"data": "test"})

        assert len(events) == 2
        assert events[0].event_type == "progress"
        assert events[0].progress == 0.5
        assert events[0].message == "Halfway"
        assert events[1].event_type == "partial"
        assert events[1].data == {"data": "test"}

    @pytest.mark.asyncio
    async def test_callback_reporter_async(self):
        """CallbackProgressReporter should handle async callbacks."""
        events: list[ProgressEvent] = []

        async def async_callback(event: ProgressEvent) -> None:
            await asyncio.sleep(0)  # Simulate async work
            events.append(event)

        reporter = CallbackProgressReporter(async_callback)
        await reporter.report_progress(1.0, "Done")

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_queue_reporter(self):
        """QueueProgressReporter should put events in queue."""
        reporter = QueueProgressReporter()

        await reporter.report_progress(0.25, "Quarter done")
        await reporter.emit_partial("row", {"id": 1})

        event1 = await reporter.queue.get()
        event2 = await reporter.queue.get()

        assert event1.progress == 0.25
        assert event2.data == {"id": 1}

    @pytest.mark.asyncio
    async def test_null_reporter(self):
        """NullProgressReporter should silently discard events."""
        reporter = NullProgressReporter()
        # Should not raise
        await reporter.report_progress(0.5)
        await reporter.emit_partial("test", {})
        await reporter.start(100)
        await reporter.complete()

    @pytest.mark.asyncio
    async def test_start_complete_helpers(self):
        """Start and complete helpers should report correct progress."""
        events: list[ProgressEvent] = []
        reporter = CallbackProgressReporter(events.append)

        await reporter.start(10)
        await reporter.complete("All done")

        assert events[0].progress == 0.0
        assert events[1].progress == 1.0
        assert events[1].message == "All done"


class TestThunkContext:
    """Tests for ThunkContext."""

    def test_create_empty(self):
        """Should create empty context."""
        ctx = ThunkContext.create()
        assert ctx.get("foo") is None
        assert ctx.cancellation is None

    def test_create_with_values(self):
        """Should create context with initial values."""
        ctx = ThunkContext.create(user_id="123", request_id="abc")
        assert ctx.get("user_id") == "123"
        assert ctx.get("request_id") == "abc"

    def test_create_with_cancellation(self):
        """Should create context with cancellation token."""
        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token)
        assert ctx.cancellation is token

    def test_create_with_progress(self):
        """Should create context with progress reporter."""
        reporter = NullProgressReporter()
        ctx = ThunkContext.create(progress=reporter)
        assert ctx.progress is reporter

    def test_background_context(self):
        """Background context should be minimal."""
        ctx = ThunkContext.background()
        assert ctx.cancellation is None
        assert isinstance(ctx.progress, NullProgressReporter)

    def test_get_with_default(self):
        """Get should return default for missing keys."""
        ctx = ThunkContext.create()
        assert ctx.get("missing", "default") == "default"

    def test_getitem(self):
        """Should support bracket access."""
        ctx = ThunkContext.create(foo="bar")
        assert ctx["foo"] == "bar"

    def test_getitem_missing_raises(self):
        """Bracket access should raise KeyError for missing keys."""
        ctx = ThunkContext.create()
        with pytest.raises(KeyError):
            _ = ctx["missing"]

    def test_contains(self):
        """Should support 'in' operator."""
        ctx = ThunkContext.create(foo="bar")
        assert "foo" in ctx
        assert "missing" not in ctx

    def test_with_value(self):
        """with_value should return new context with added value."""
        ctx1 = ThunkContext.create(a=1)
        ctx2 = ctx1.with_value("b", 2)

        assert ctx1.get("b") is None  # Original unchanged
        assert ctx2.get("a") == 1  # Inherited
        assert ctx2.get("b") == 2  # Added

    def test_with_values(self):
        """with_values should return new context with multiple values."""
        ctx1 = ThunkContext.create()
        ctx2 = ctx1.with_values(a=1, b=2, c=3)

        assert ctx2.get("a") == 1
        assert ctx2.get("b") == 2
        assert ctx2.get("c") == 3

    def test_with_cancellation(self):
        """with_cancellation should return new context with token."""
        ctx1 = ThunkContext.create()
        token = CancellationToken()
        ctx2 = ctx1.with_cancellation(token)

        assert ctx1.cancellation is None
        assert ctx2.cancellation is token

    def test_with_progress(self):
        """with_progress should return new context with reporter."""
        ctx1 = ThunkContext.create()
        reporter = NullProgressReporter()
        ctx2 = ctx1.with_progress(reporter)

        assert ctx2.progress is reporter

    def test_child_inherits_values(self):
        """Child context should inherit parent values."""
        parent = ThunkContext.create(a=1, b=2)
        child = parent.child(c=3)

        assert child.get("a") == 1  # Inherited
        assert child.get("b") == 2  # Inherited
        assert child.get("c") == 3  # Own value

    def test_child_overrides_values(self):
        """Child values should override parent values."""
        parent = ThunkContext.create(a=1)
        child = parent.child(a=99)

        assert parent.get("a") == 1  # Unchanged
        assert child.get("a") == 99  # Overridden

    def test_child_inherits_progress(self):
        """Child should inherit progress reporter from parent."""
        reporter = NullProgressReporter()
        parent = ThunkContext.create(progress=reporter)
        child = parent.child()

        assert child.progress is reporter

    def test_child_creates_linked_cancellation(self):
        """Child should have linked cancellation token."""
        parent_token = CancellationToken()
        parent = ThunkContext.create(cancellation=parent_token)
        child = parent.child()

        assert child.cancellation is not parent_token
        assert child.cancellation is not None

        parent_token.cancel_sync("parent cancel")
        assert child.is_cancelled is True

    def test_is_cancelled_property(self):
        """is_cancelled should check token state."""
        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token)

        assert ctx.is_cancelled is False
        token.cancel_sync()
        assert ctx.is_cancelled is True

    def test_is_cancelled_no_token(self):
        """is_cancelled should be False when no token."""
        ctx = ThunkContext.create()
        assert ctx.is_cancelled is False

    def test_raise_if_cancelled(self):
        """raise_if_cancelled should check token."""
        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token)

        ctx.raise_if_cancelled()  # Should not raise

        token.cancel_sync("stopped")
        with pytest.raises(CancellationError):
            ctx.raise_if_cancelled()

    def test_raise_if_cancelled_no_token(self):
        """raise_if_cancelled should not raise when no token."""
        ctx = ThunkContext.create()
        ctx.raise_if_cancelled()  # Should not raise

    def test_to_dict(self):
        """to_dict should flatten context hierarchy."""
        parent = ThunkContext.create(a=1, b=2)
        child = parent.child(b=99, c=3)

        result = child.to_dict()
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_contains_in_parent(self):
        """'in' should check parent context too."""
        parent = ThunkContext.create(parent_key="value")
        child = parent.child()

        assert "parent_key" in child
