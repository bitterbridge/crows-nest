"""Event types for the messaging system.

Provides typed events for pub/sub communication between components.
Events are immutable and serializable for persistence and transmission.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base event with common metadata.

    All events inherit from this class and add domain-specific fields.

    Attributes:
        id: Unique event identifier.
        event_type: String identifying the event type (e.g., "thunk.started").
        timestamp: When the event occurred.
        trace_id: Correlation ID for distributed tracing.
        payload: Additional event-specific data.
    """

    id: UUID = Field(default_factory=uuid4)
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Create event from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            event_type=data["event_type"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if isinstance(data["timestamp"], str)
                else data["timestamp"]
            ),
            trace_id=data["trace_id"],
            payload=data.get("payload", {}),
        )


class ThunkStartedEvent(Event):
    """Emitted when a thunk begins execution.

    Attributes:
        thunk_id: ID of the thunk being executed.
        operation: Name of the operation being performed.
        parent_id: ID of the parent thunk, if any.
    """

    event_type: Literal["thunk.started"] = "thunk.started"
    thunk_id: UUID
    operation: str
    parent_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = super().to_dict()
        data["thunk_id"] = str(self.thunk_id)
        data["operation"] = self.operation
        data["parent_id"] = str(self.parent_id) if self.parent_id else None
        return data


class ThunkCompletedEvent(Event):
    """Emitted when a thunk completes successfully.

    Attributes:
        thunk_id: ID of the completed thunk.
        operation: Name of the operation that completed.
        duration_ms: Execution time in milliseconds.
        result_summary: Brief summary of the result (not full payload).
    """

    event_type: Literal["thunk.completed"] = "thunk.completed"
    thunk_id: UUID
    operation: str
    duration_ms: int
    result_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = super().to_dict()
        data["thunk_id"] = str(self.thunk_id)
        data["operation"] = self.operation
        data["duration_ms"] = self.duration_ms
        data["result_summary"] = self.result_summary
        return data


class ThunkFailedEvent(Event):
    """Emitted when a thunk fails.

    Attributes:
        thunk_id: ID of the failed thunk.
        operation: Name of the operation that failed.
        error_type: Type of error (e.g., "ValidationError").
        error_message: Human-readable error message.
        duration_ms: Execution time before failure.
    """

    event_type: Literal["thunk.failed"] = "thunk.failed"
    thunk_id: UUID
    operation: str
    error_type: str
    error_message: str
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = super().to_dict()
        data["thunk_id"] = str(self.thunk_id)
        data["operation"] = self.operation
        data["error_type"] = self.error_type
        data["error_message"] = self.error_message
        data["duration_ms"] = self.duration_ms
        return data


class AgentSpawnedEvent(Event):
    """Emitted when an agent spawns a child agent.

    Attributes:
        parent_agent_id: ID of the parent agent.
        child_agent_id: ID of the newly spawned agent.
        child_spec_name: Name of the agent spec used.
        capabilities: Capabilities granted to the child.
    """

    event_type: Literal["agent.spawned"] = "agent.spawned"
    parent_agent_id: UUID
    child_agent_id: UUID
    child_spec_name: str
    capabilities: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = super().to_dict()
        data["parent_agent_id"] = str(self.parent_agent_id)
        data["child_agent_id"] = str(self.child_agent_id)
        data["child_spec_name"] = self.child_spec_name
        data["capabilities"] = self.capabilities
        return data


class QueueMessageEvent(Event):
    """Emitted when a message is published to a queue.

    Attributes:
        message_id: ID of the published message.
        queue_name: Name of the target queue.
        publisher_id: ID of the publishing agent/thunk.
    """

    event_type: Literal["queue.message_published"] = "queue.message_published"
    message_id: UUID
    queue_name: str
    publisher_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = super().to_dict()
        data["message_id"] = str(self.message_id)
        data["queue_name"] = self.queue_name
        data["publisher_id"] = str(self.publisher_id) if self.publisher_id else None
        return data


# Event type registry for deserialization
EVENT_TYPES: dict[str, type[Event]] = {
    "thunk.started": ThunkStartedEvent,
    "thunk.completed": ThunkCompletedEvent,
    "thunk.failed": ThunkFailedEvent,
    "agent.spawned": AgentSpawnedEvent,
    "queue.message_published": QueueMessageEvent,
}


def deserialize_event(data: dict[str, Any]) -> Event:
    """Deserialize an event from a dictionary.

    Args:
        data: Dictionary containing event data.

    Returns:
        The appropriate Event subclass instance.

    Raises:
        ValueError: If event_type is unknown.
    """
    event_type = data.get("event_type")
    if event_type in EVENT_TYPES:
        event_class = EVENT_TYPES[event_type]
        return event_class.model_validate(data)
    # Fall back to base Event for unknown types
    return Event.model_validate(data)
