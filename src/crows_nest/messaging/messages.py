"""Message types for the queue system.

Provides the Message class for queue-based async communication.
Messages support acknowledgment, retry, and dead letter handling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageStatus(StrEnum):
    """Status of a queue message."""

    PENDING = "pending"  # Waiting to be consumed
    PROCESSING = "processing"  # Being processed by a consumer
    COMPLETED = "completed"  # Successfully processed and acked
    FAILED = "failed"  # Failed and will be retried
    DEAD = "dead"  # Exceeded max attempts, moved to DLQ


class Message(BaseModel):
    """A message in a queue.

    Messages are the unit of work in the queue system. They support:
    - Acknowledgment (ack) for successful processing
    - Negative acknowledgment (nack) for retry
    - Automatic retry with attempt tracking
    - Dead letter queue after max attempts exceeded

    Attributes:
        id: Unique message identifier.
        queue_name: Name of the queue this message belongs to.
        payload: The message content (JSON-serializable dict).
        status: Current message status.
        created_at: When the message was created.
        updated_at: When the message was last updated.
        attempts: Number of delivery attempts.
        max_attempts: Maximum attempts before moving to DLQ.
        error: Last error message if failed.
        trace_id: Correlation ID for distributed tracing.
        publisher_id: ID of the thunk/agent that published this message.
    """

    id: UUID = Field(default_factory=uuid4)
    queue_name: str
    payload: dict[str, Any]
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    publisher_id: UUID | None = None

    model_config = {"frozen": False}  # Messages are mutable for status updates

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": str(self.id),
            "queue_name": self.queue_name,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "trace_id": self.trace_id,
            "publisher_id": str(self.publisher_id) if self.publisher_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create message from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            queue_name=data["queue_name"],
            payload=data["payload"],
            status=MessageStatus(data["status"]),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data["created_at"], str)
                else data["created_at"]
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data["updated_at"], str)
                else data["updated_at"]
            ),
            attempts=data["attempts"],
            max_attempts=data["max_attempts"],
            error=data.get("error"),
            trace_id=data["trace_id"],
            publisher_id=(
                UUID(data["publisher_id"])
                if data.get("publisher_id") and isinstance(data["publisher_id"], str)
                else data.get("publisher_id")
            ),
        )

    def can_retry(self) -> bool:
        """Check if the message can be retried."""
        return self.attempts < self.max_attempts

    def mark_processing(self) -> None:
        """Mark message as being processed."""
        self.status = MessageStatus.PROCESSING
        self.attempts += 1
        self.updated_at = datetime.now(UTC)

    def mark_completed(self) -> None:
        """Mark message as successfully processed."""
        self.status = MessageStatus.COMPLETED
        self.updated_at = datetime.now(UTC)
        self.error = None

    def mark_failed(self, error: str) -> None:
        """Mark message as failed.

        If max attempts exceeded, moves to DEAD status.

        Args:
            error: Error message describing the failure.
        """
        self.error = error
        self.updated_at = datetime.now(UTC)
        if self.can_retry():
            self.status = MessageStatus.FAILED
        else:
            self.status = MessageStatus.DEAD

    def reset_for_retry(self) -> None:
        """Reset message for retry attempt."""
        if self.can_retry():
            self.status = MessageStatus.PENDING
            self.updated_at = datetime.now(UTC)


class QueueStats(BaseModel):
    """Statistics for a message queue.

    Attributes:
        queue_name: Name of the queue.
        pending: Number of messages waiting to be processed.
        processing: Number of messages currently being processed.
        completed: Number of successfully processed messages.
        failed: Number of failed messages awaiting retry.
        dead: Number of messages in dead letter queue.
        total_published: Total messages ever published.
        total_consumed: Total messages ever consumed.
    """

    queue_name: str
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    dead: int = 0
    total_published: int = 0
    total_consumed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "queue_name": self.queue_name,
            "pending": self.pending,
            "processing": self.processing,
            "completed": self.completed,
            "failed": self.failed,
            "dead": self.dead,
            "total_published": self.total_published,
            "total_consumed": self.total_consumed,
            "depth": self.pending + self.processing + self.failed,
        }
