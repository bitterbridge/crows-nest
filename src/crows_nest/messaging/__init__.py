"""Messaging system for async communication.

Provides pub/sub events and message queues for agent communication.
"""

# Import operations to register them with the thunk registry
from crows_nest.messaging import operations as _operations  # noqa: F401
from crows_nest.messaging.bus import (
    EventBus,
    get_event_bus,
    publish_event,
    reset_event_bus,
    subscribe_event,
    unsubscribe_event,
)
from crows_nest.messaging.events import (
    EVENT_TYPES,
    AgentSpawnedEvent,
    Event,
    QueueMessageEvent,
    ThunkCompletedEvent,
    ThunkFailedEvent,
    ThunkStartedEvent,
    deserialize_event,
)
from crows_nest.messaging.integration import (
    connect_runtime_to_eventbus,
    disconnect_runtime_from_eventbus,
)
from crows_nest.messaging.messages import (
    Message,
    MessageStatus,
    QueueStats,
)
from crows_nest.messaging.queue import (
    MessageQueue,
    get_message_queue,
    reset_message_queue,
)

__all__ = [
    "EVENT_TYPES",
    "AgentSpawnedEvent",
    "Event",
    "EventBus",
    "Message",
    "MessageQueue",
    "MessageStatus",
    "QueueMessageEvent",
    "QueueStats",
    "ThunkCompletedEvent",
    "ThunkFailedEvent",
    "ThunkStartedEvent",
    "connect_runtime_to_eventbus",
    "deserialize_event",
    "disconnect_runtime_from_eventbus",
    "get_event_bus",
    "get_message_queue",
    "publish_event",
    "reset_event_bus",
    "reset_message_queue",
    "subscribe_event",
    "unsubscribe_event",
]
