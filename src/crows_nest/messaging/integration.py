"""Integration between thunk runtime and messaging system.

Bridges the runtime event system to the pub/sub event bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crows_nest.core.runtime import RuntimeEvent, RuntimeEventData
from crows_nest.messaging.bus import get_event_bus
from crows_nest.messaging.events import (
    ThunkCompletedEvent,
    ThunkFailedEvent,
    ThunkStartedEvent,
)

if TYPE_CHECKING:
    from crows_nest.core.runtime import ThunkRuntime


async def _runtime_to_eventbus_listener(event_data: RuntimeEventData) -> None:
    """Convert runtime events to EventBus events and publish them.

    Args:
        event_data: The runtime event data.
    """
    bus = get_event_bus()

    if event_data.event == RuntimeEvent.THUNK_FORCING:
        started_event = ThunkStartedEvent(
            thunk_id=event_data.thunk_id,
            operation=event_data.data.get("operation", "unknown"),
            trace_id=event_data.data.get("trace_id", ""),
        )
        await bus.publish(started_event)

    elif event_data.event == RuntimeEvent.THUNK_COMPLETED:
        duration_ms = event_data.data.get("duration_ms", 0)
        completed_event = ThunkCompletedEvent(
            thunk_id=event_data.thunk_id,
            operation=event_data.data.get("operation", "unknown"),
            duration_ms=duration_ms,
            trace_id=event_data.data.get("trace_id", ""),
        )
        await bus.publish(completed_event)

    elif event_data.event == RuntimeEvent.THUNK_FAILED:
        error = event_data.data.get("error")
        error_type = "Error"
        error_message = "Unknown error"
        if error:
            error_type = type(error).__name__ if hasattr(error, "__class__") else "Error"
            error_message = str(error.message) if hasattr(error, "message") else str(error)

        failed_event = ThunkFailedEvent(
            thunk_id=event_data.thunk_id,
            operation=event_data.data.get("operation", "unknown"),
            error_type=error_type,
            error_message=error_message,
            duration_ms=event_data.data.get("duration_ms", 0),
            trace_id=event_data.data.get("trace_id", ""),
        )
        await bus.publish(failed_event)


def connect_runtime_to_eventbus(runtime: ThunkRuntime) -> None:
    """Connect a ThunkRuntime to the global EventBus.

    Registers a listener on the runtime that forwards events to the EventBus.
    This allows components to subscribe to thunk lifecycle events via the
    EventBus instead of directly on the runtime.

    Args:
        runtime: The runtime to connect.

    Example:
        runtime = ThunkRuntime(registry=get_global_registry())
        connect_runtime_to_eventbus(runtime)

        # Now components can subscribe via the event bus
        bus = get_event_bus()
        bus.subscribe("thunk.started", my_handler)
    """
    # Register for all relevant events
    runtime.on(RuntimeEvent.THUNK_FORCING, _runtime_to_eventbus_listener)
    runtime.on(RuntimeEvent.THUNK_COMPLETED, _runtime_to_eventbus_listener)
    runtime.on(RuntimeEvent.THUNK_FAILED, _runtime_to_eventbus_listener)


def disconnect_runtime_from_eventbus(runtime: ThunkRuntime) -> None:
    """Disconnect a ThunkRuntime from the global EventBus.

    Removes the listener that forwards events to the EventBus.

    Args:
        runtime: The runtime to disconnect.
    """
    runtime.off(RuntimeEvent.THUNK_FORCING, _runtime_to_eventbus_listener)
    runtime.off(RuntimeEvent.THUNK_COMPLETED, _runtime_to_eventbus_listener)
    runtime.off(RuntimeEvent.THUNK_FAILED, _runtime_to_eventbus_listener)
