"""Event bus for pub/sub communication.

Provides an async event bus that allows components to subscribe to events
and publish events to all interested subscribers.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from crows_nest.messaging.events import Event

    EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]

logger = logging.getLogger(__name__)


class EventBus:
    """Async event bus for pub/sub communication.

    Supports:
    - Type-based subscription (exact event type matching)
    - Wildcard patterns (e.g., "thunk.*" matches "thunk.started", "thunk.completed")
    - Both sync and async handlers
    - Handler errors don't break event delivery to other handlers

    Usage:
        bus = EventBus()

        async def on_thunk_started(event: ThunkStartedEvent):
            print(f"Thunk {event.thunk_id} started")

        bus.subscribe("thunk.started", on_thunk_started)
        bus.subscribe("thunk.*", lambda e: print(f"Thunk event: {e.event_type}"))

        await bus.publish(ThunkStartedEvent(thunk_id=uuid4(), operation="test"))
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(self, event_pattern: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event pattern.

        Args:
            event_pattern: Event type to match. Supports wildcards:
                - "thunk.started" matches only "thunk.started"
                - "thunk.*" matches "thunk.started", "thunk.completed", etc.
                - "*" matches all events
            handler: Async or sync callable that receives the event.
        """
        self._handlers[event_pattern].append(handler)
        logger.debug("Subscribed handler to pattern: %s", event_pattern)

    def unsubscribe(self, event_pattern: str, handler: EventHandler) -> bool:
        """Unsubscribe a handler from an event pattern.

        Args:
            event_pattern: The pattern the handler was subscribed to.
            handler: The handler to remove.

        Returns:
            True if the handler was found and removed, False otherwise.
        """
        handlers = self._handlers.get(event_pattern, [])
        try:
            handlers.remove(handler)
            logger.debug("Unsubscribed handler from pattern: %s", event_pattern)
            return True
        except ValueError:
            return False

    async def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: The event to publish.

        Returns:
            Number of handlers that received the event.
        """
        handlers_called = 0
        event_type = event.event_type

        # Find all matching handlers
        matching_handlers: list[EventHandler] = []
        for pattern, handlers in self._handlers.items():
            if self._matches_pattern(pattern, event_type):
                matching_handlers.extend(handlers)

        # Call all handlers
        for handler in matching_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                handlers_called += 1
            except Exception:
                # Log but don't break delivery to other handlers
                logger.exception(
                    "Handler error for event %s",
                    event_type,
                )

        logger.debug(
            "Published event %s to %d handlers",
            event_type,
            handlers_called,
        )
        return handlers_called

    def _matches_pattern(self, pattern: str, event_type: str) -> bool:
        """Check if an event type matches a subscription pattern.

        Args:
            pattern: Subscription pattern (may include wildcards).
            event_type: Event type to match against.

        Returns:
            True if the event type matches the pattern.
        """
        # Direct match
        if pattern == event_type:
            return True
        # Wildcard match using fnmatch
        return fnmatch.fnmatch(event_type, pattern)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
        logger.debug("Cleared all event subscriptions")

    def get_handler_count(self, event_pattern: str | None = None) -> int:
        """Get the number of subscribed handlers.

        Args:
            event_pattern: If provided, count handlers for this pattern only.
                If None, count all handlers.

        Returns:
            Number of subscribed handlers.
        """
        if event_pattern is not None:
            return len(self._handlers.get(event_pattern, []))
        return sum(len(handlers) for handlers in self._handlers.values())

    def get_patterns(self) -> list[str]:
        """Get all subscribed event patterns.

        Returns:
            List of event patterns with active subscriptions.
        """
        return [pattern for pattern, handlers in self._handlers.items() if handlers]


# Global event bus instance
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus.

    Returns:
        The global EventBus instance.
    """
    global _global_bus  # noqa: PLW0603
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus. Useful for testing."""
    global _global_bus  # noqa: PLW0603
    _global_bus = None


async def publish_event(event: Event) -> int:
    """Publish an event using the global event bus.

    Convenience function for publishing events.

    Args:
        event: The event to publish.

    Returns:
        Number of handlers that received the event.
    """
    return await get_event_bus().publish(event)


def subscribe_event(event_pattern: str, handler: Any) -> None:
    """Subscribe to events using the global event bus.

    Convenience function for subscribing to events.

    Args:
        event_pattern: Event type pattern to subscribe to.
        handler: Handler callable.
    """
    get_event_bus().subscribe(event_pattern, handler)


def unsubscribe_event(event_pattern: str, handler: Any) -> bool:
    """Unsubscribe from events using the global event bus.

    Convenience function for unsubscribing from events.

    Args:
        event_pattern: Event type pattern to unsubscribe from.
        handler: Handler to remove.

    Returns:
        True if handler was removed, False otherwise.
    """
    return get_event_bus().unsubscribe(event_pattern, handler)
