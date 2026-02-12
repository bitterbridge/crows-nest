"""Thunk context for propagating state, cancellation, and progress.

Provides a structured way to pass cross-cutting concerns through thunk pipelines
without polluting operation signatures.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class CancellationError(Exception):
    """Raised when a thunk is cancelled."""

    def __init__(self, reason: str = "Operation cancelled") -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of thunks.

    Cancellation is cooperative - thunks must check the token periodically.
    The runtime checks before execution, and combinators check between steps.

    Example:
        token = CancellationToken()

        # In a long-running operation:
        for item in items:
            token.raise_if_cancelled()
            process(item)

        # Cancel from elsewhere:
        token.cancel("User requested stop")
    """

    _cancelled: bool = field(default=False, repr=False)
    _reason: str | None = field(default=None, repr=False)
    _callbacks: list[Callable[[], None | Awaitable[None]]] = field(default_factory=list, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """Get the cancellation reason, if cancelled."""
        return self._reason

    async def cancel(self, reason: str = "Operation cancelled") -> None:
        """Request cancellation.

        Thread-safe. Callbacks are invoked after setting the cancelled flag.

        Args:
            reason: Human-readable reason for cancellation.
        """
        async with self._lock:
            if self._cancelled:
                return  # Already cancelled
            self._cancelled = True
            self._reason = reason

        # Invoke callbacks outside lock - errors don't prevent cancellation
        for callback in self._callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: S112 # nosec B112 - intentionally suppressing callback errors
                continue

    def cancel_sync(self, reason: str = "Operation cancelled") -> None:
        """Synchronous cancellation for use in sync contexts.

        Note: Callbacks are invoked synchronously. Async callbacks will be skipped.
        """
        if self._cancelled:
            return
        self._cancelled = True
        self._reason = reason

        # Invoke sync callbacks - skip async ones in sync context
        for callback in self._callbacks:
            try:
                result = callback()
                # Skip if callback returns a coroutine (can't await in sync context)
                if asyncio.iscoroutine(result):
                    result.close()  # Clean up unawaited coroutine
            except Exception:  # noqa: S112 # nosec B112 - intentionally suppressing callback errors
                continue

    def raise_if_cancelled(self) -> None:
        """Raise CancellationError if cancelled.

        Call this periodically in long-running operations.

        Raises:
            CancellationError: If cancellation was requested.
        """
        if self._cancelled:
            raise CancellationError(self._reason or "Operation cancelled")

    def on_cancel(self, callback: Callable[[], None | Awaitable[None]]) -> None:
        """Register a callback to be invoked on cancellation.

        If already cancelled, callback is invoked immediately (sync only).

        Args:
            callback: Function to call when cancelled.
        """
        if self._cancelled:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    # In sync context, we can't await - schedule as background task
                    # Store reference to prevent garbage collection
                    task = asyncio.create_task(result)
                    task.add_done_callback(lambda _: None)  # Prevent warning
            except Exception:
                return
        else:
            self._callbacks.append(callback)

    def child(self) -> CancellationToken:
        """Create a child token that cancels when parent cancels.

        The child can be cancelled independently, but parent cancellation
        propagates to child.

        Returns:
            New CancellationToken linked to this one.
        """
        child_token = CancellationToken()
        if self._cancelled:
            child_token._cancelled = True
            child_token._reason = self._reason
        else:
            self.on_cancel(lambda: child_token.cancel_sync(self._reason or "Parent cancelled"))
        return child_token


class ProgressReporter(ABC):
    """Abstract base for reporting progress and streaming partial results.

    Implementations might write to a callback, send to a queue, or emit events.
    """

    @abstractmethod
    async def report_progress(
        self,
        progress: float,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Report progress as a fraction from 0.0 to 1.0.

        Args:
            progress: Progress fraction (0.0 = just started, 1.0 = complete).
            message: Optional human-readable status message.
            metadata: Optional additional data about current state.
        """

    @abstractmethod
    async def emit_partial(
        self,
        event: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a partial/streaming result.

        Args:
            event: Event type/name (e.g., "chunk", "row", "token").
            data: The partial data.
            metadata: Optional additional context.
        """

    async def start(self, total_steps: int | None = None) -> None:
        """Signal the start of an operation.

        Args:
            total_steps: Optional total number of steps for progress calculation.
        """
        await self.report_progress(0.0, "Starting", {"total_steps": total_steps})

    async def complete(self, message: str = "Complete") -> None:
        """Signal completion of an operation."""
        await self.report_progress(1.0, message)


@dataclass
class ProgressEvent:
    """A progress or streaming event."""

    event_type: str  # "progress" or "partial"
    progress: float | None = None
    message: str | None = None
    data: Any = None
    metadata: dict[str, Any] | None = None


class CallbackProgressReporter(ProgressReporter):
    """Progress reporter that invokes a callback for each event."""

    def __init__(
        self,
        callback: Callable[[ProgressEvent], None | Awaitable[None]],
    ) -> None:
        """Initialize with a callback function.

        Args:
            callback: Function to call with each progress event.
                      Can be sync or async.
        """
        self._callback = callback

    async def _invoke(self, event: ProgressEvent) -> None:
        result = self._callback(event)
        if asyncio.iscoroutine(result):
            await result

    async def report_progress(
        self,
        progress: float,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._invoke(
            ProgressEvent(
                event_type="progress",
                progress=progress,
                message=message,
                metadata=metadata,
            )
        )

    async def emit_partial(
        self,
        event: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._invoke(
            ProgressEvent(
                event_type="partial",
                message=event,
                data=data,
                metadata=metadata,
            )
        )


class QueueProgressReporter(ProgressReporter):
    """Progress reporter that puts events into an async queue."""

    def __init__(self, queue: asyncio.Queue[ProgressEvent] | None = None) -> None:
        """Initialize with optional queue.

        Args:
            queue: Queue to put events into. Created if not provided.
        """
        self._queue: asyncio.Queue[ProgressEvent] = queue or asyncio.Queue()

    @property
    def queue(self) -> asyncio.Queue[ProgressEvent]:
        """Get the event queue for consuming progress events."""
        return self._queue

    async def report_progress(
        self,
        progress: float,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._queue.put(
            ProgressEvent(
                event_type="progress",
                progress=progress,
                message=message,
                metadata=metadata,
            )
        )

    async def emit_partial(
        self,
        event: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._queue.put(
            ProgressEvent(
                event_type="partial",
                message=event,
                data=data,
                metadata=metadata,
            )
        )


class NullProgressReporter(ProgressReporter):
    """Progress reporter that discards all events."""

    async def report_progress(
        self,
        progress: float,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    async def emit_partial(
        self,
        event: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass


# Singleton null reporter
_NULL_REPORTER = NullProgressReporter()


@dataclass
class ThunkContext:
    """Context that flows through thunk execution pipelines.

    Carries cross-cutting concerns like cancellation, progress reporting,
    and user-defined values without polluting operation signatures.

    Context is immutable - modifications return new contexts. Child contexts
    inherit from parents but can override values.

    Example:
        # Create context with cancellation and progress
        ctx = ThunkContext.create(
            cancellation=CancellationToken(),
            progress=CallbackProgressReporter(print),
        )

        # Add custom values
        ctx = ctx.with_values(user_id="123", request_id="abc")

        # Pass to runtime
        result = await runtime.force(thunk, context=ctx)

        # In a combinator, create child context
        child_ctx = ctx.child(operation="sub_operation")
    """

    _values: dict[str, Any] = field(default_factory=dict)
    _cancellation: CancellationToken | None = field(default=None)
    _progress: ProgressReporter | None = field(default=None)
    _parent: ThunkContext | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        cancellation: CancellationToken | None = None,
        progress: ProgressReporter | None = None,
        **values: Any,
    ) -> ThunkContext:
        """Create a new context.

        Args:
            cancellation: Optional cancellation token.
            progress: Optional progress reporter.
            **values: Initial context values.

        Returns:
            New ThunkContext.
        """
        return cls(
            _values=dict(values),
            _cancellation=cancellation,
            _progress=progress,
            _parent=None,
        )

    @classmethod
    def background(cls) -> ThunkContext:
        """Create a minimal background context.

        Use this when no specific context is needed but the API requires one.

        Returns:
            Empty context with no cancellation or progress.
        """
        return cls()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context, checking parent if not found.

        Args:
            key: The key to look up.
            default: Value to return if key not found.

        Returns:
            The value, or default if not found.
        """
        if key in self._values:
            return self._values[key]
        if self._parent is not None:
            return self._parent.get(key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        """Get a value, raising KeyError if not found."""
        if key in self._values:
            return self._values[key]
        if self._parent is not None:
            return self._parent[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context or parents."""
        if key in self._values:
            return True
        if self._parent is not None:
            return key in self._parent
        return False

    @property
    def cancellation(self) -> CancellationToken | None:
        """Get the cancellation token, checking parent if needed."""
        if self._cancellation is not None:
            return self._cancellation
        if self._parent is not None:
            return self._parent.cancellation
        return None

    @property
    def progress(self) -> ProgressReporter:
        """Get the progress reporter, or null reporter if none set."""
        if self._progress is not None:
            return self._progress
        if self._parent is not None:
            return self._parent.progress
        return _NULL_REPORTER

    @property
    def is_cancelled(self) -> bool:
        """Check if the context's cancellation token is cancelled."""
        token = self.cancellation
        return token.is_cancelled if token else False

    def raise_if_cancelled(self) -> None:
        """Raise CancellationError if cancelled.

        Convenience method that checks the cancellation token.

        Raises:
            CancellationError: If cancellation was requested.
        """
        token = self.cancellation
        if token:
            token.raise_if_cancelled()

    def with_value(self, key: str, value: Any) -> ThunkContext:
        """Create new context with an additional value.

        Args:
            key: The key to set.
            value: The value to associate.

        Returns:
            New context with the value set.
        """
        new_values = {**self._values, key: value}
        return ThunkContext(
            _values=new_values,
            _cancellation=self._cancellation,
            _progress=self._progress,
            _parent=self._parent,
        )

    def with_values(self, **values: Any) -> ThunkContext:
        """Create new context with additional values.

        Args:
            **values: Key-value pairs to add.

        Returns:
            New context with the values set.
        """
        new_values = {**self._values, **values}
        return ThunkContext(
            _values=new_values,
            _cancellation=self._cancellation,
            _progress=self._progress,
            _parent=self._parent,
        )

    def with_cancellation(self, token: CancellationToken) -> ThunkContext:
        """Create new context with a different cancellation token.

        Args:
            token: The new cancellation token.

        Returns:
            New context with the token set.
        """
        return ThunkContext(
            _values=self._values,
            _cancellation=token,
            _progress=self._progress,
            _parent=self._parent,
        )

    def with_progress(self, reporter: ProgressReporter) -> ThunkContext:
        """Create new context with a different progress reporter.

        Args:
            reporter: The new progress reporter.

        Returns:
            New context with the reporter set.
        """
        return ThunkContext(
            _values=self._values,
            _cancellation=self._cancellation,
            _progress=reporter,
            _parent=self._parent,
        )

    def child(self, **values: Any) -> ThunkContext:
        """Create a child context that inherits from this one.

        Child contexts can access parent values but override them locally.
        If this context has a cancellation token, a linked child token is created.

        Args:
            **values: Values to set in the child context.

        Returns:
            New child context.
        """
        # Create linked cancellation token if parent has one
        child_cancellation = None
        if self._cancellation is not None:
            child_cancellation = self._cancellation.child()

        return ThunkContext(
            _values=dict(values),
            _cancellation=child_cancellation,
            _progress=None,  # Inherit from parent via property
            _parent=self,
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten context values to a dict, including parent values.

        Child values override parent values.

        Returns:
            Dict of all context values.
        """
        if self._parent is not None:
            result = self._parent.to_dict()
            result.update(self._values)
            return result
        return dict(self._values)
