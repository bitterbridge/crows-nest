"""Message queue for async task distribution.

Provides in-memory message queues with optional SQLite persistence.
Supports acknowledgment, retry, and dead letter handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import aiosqlite

from crows_nest.messaging.messages import Message, MessageStatus, QueueStats

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    MessageHandler = Callable[[Message], Awaitable[bool]] | Callable[[Message], bool]

logger = logging.getLogger(__name__)


class MessageQueue:
    """In-memory message queue with optional persistence.

    Features:
    - FIFO ordering per queue
    - Message acknowledgment (ack/nack)
    - Automatic retry with configurable max attempts
    - Dead letter queue for failed messages
    - Optional SQLite persistence

    Usage:
        queue = MessageQueue()

        # Publish a message
        msg = await queue.publish("tasks", {"action": "process", "data": "..."})

        # Consume messages
        async def handler(msg: Message) -> bool:
            # Process message
            return True  # Return True to ack, False to nack

        await queue.consume("tasks", handler)
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        default_max_attempts: int = 3,
    ) -> None:
        """Initialize the message queue.

        Args:
            persistence_path: Path to SQLite database for persistence.
                If None, queue is in-memory only.
            default_max_attempts: Default max delivery attempts for messages.
        """
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._dead_letter: dict[str, list[Message]] = defaultdict(list)
        self._processing: dict[UUID, Message] = {}
        self._stats: dict[str, QueueStats] = {}
        self._consumers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._persistence_path = persistence_path
        self._default_max_attempts = default_max_attempts
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._running = False
        self._consumer_tasks: list[asyncio.Task[None]] = []

    async def initialize(self) -> None:
        """Initialize the queue, setting up persistence if configured."""
        if self._persistence_path:
            await self._init_persistence()

    async def _init_persistence(self) -> None:
        """Initialize SQLite persistence."""
        if not self._persistence_path:
            return

        self._db = await aiosqlite.connect(self._persistence_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                queue_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                error TEXT,
                trace_id TEXT NOT NULL,
                publisher_id TEXT
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_queue_status
            ON messages(queue_name, status)
        """)
        await self._db.commit()

        # Load pending messages from database
        await self._load_persisted_messages()

    async def _load_persisted_messages(self) -> None:
        """Load messages from persistence on startup."""
        if not self._db:
            return

        async with self._db.execute(
            "SELECT * FROM messages WHERE status IN (?, ?)",
            (MessageStatus.PENDING.value, MessageStatus.FAILED.value),
        ) as cursor:
            async for row in cursor:
                msg = self._row_to_message(row)
                if msg.status == MessageStatus.DEAD:
                    self._dead_letter[msg.queue_name].append(msg)
                else:
                    msg.status = MessageStatus.PENDING
                    self._queues[msg.queue_name].append(msg)

    def _row_to_message(self, row: Any) -> Message:
        """Convert a database row to a Message."""
        return Message(
            id=UUID(row[0]),
            queue_name=row[1],
            payload=json.loads(row[2]),
            status=MessageStatus(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            attempts=row[6],
            max_attempts=row[7],
            error=row[8],
            trace_id=row[9],
            publisher_id=UUID(row[10]) if row[10] else None,
        )

    async def _persist_message(self, msg: Message) -> None:
        """Persist a message to the database."""
        if not self._db:
            return

        await self._db.execute(
            """
            INSERT OR REPLACE INTO messages
            (id, queue_name, payload, status, created_at, updated_at,
             attempts, max_attempts, error, trace_id, publisher_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(msg.id),
                msg.queue_name,
                json.dumps(msg.payload),
                msg.status.value,
                msg.created_at.isoformat(),
                msg.updated_at.isoformat(),
                msg.attempts,
                msg.max_attempts,
                msg.error,
                msg.trace_id,
                str(msg.publisher_id) if msg.publisher_id else None,
            ),
        )
        await self._db.commit()

    async def _delete_message(self, message_id: UUID) -> None:
        """Delete a message from persistence."""
        if not self._db:
            return

        await self._db.execute(
            "DELETE FROM messages WHERE id = ?",
            (str(message_id),),
        )
        await self._db.commit()

    async def publish(
        self,
        queue_name: str,
        payload: dict[str, Any],
        max_attempts: int | None = None,
        publisher_id: UUID | None = None,
        trace_id: str | None = None,
    ) -> Message:
        """Publish a message to a queue.

        Args:
            queue_name: Name of the target queue.
            payload: Message payload (JSON-serializable dict).
            max_attempts: Max delivery attempts (uses default if None).
            publisher_id: ID of the publishing agent/thunk.
            trace_id: Trace ID for correlation.

        Returns:
            The created Message.
        """
        msg = Message(
            queue_name=queue_name,
            payload=payload,
            max_attempts=max_attempts or self._default_max_attempts,
            publisher_id=publisher_id,
            trace_id=trace_id or "",
        )

        async with self._lock:
            self._queues[queue_name].append(msg)
            self._ensure_stats(queue_name)
            self._stats[queue_name].total_published += 1
            self._stats[queue_name].pending += 1

        await self._persist_message(msg)
        logger.debug("Published message %s to queue %s", msg.id, queue_name)

        return msg

    async def consume(
        self,
        queue_name: str,
        handler: MessageHandler,
        *,
        auto_ack: bool = True,
    ) -> None:
        """Start consuming messages from a queue.

        This method processes one message at a time from the queue.
        For continuous consumption, use start_consumer().

        Args:
            queue_name: Name of the queue to consume from.
            handler: Async or sync callable that processes messages.
                Should return True to ack, False to nack.
            auto_ack: If True, automatically ack on handler success.
        """
        async with self._lock:
            if not self._queues[queue_name]:
                return

            msg = self._queues[queue_name].pop(0)
            msg.mark_processing()
            self._processing[msg.id] = msg
            self._ensure_stats(queue_name)
            self._stats[queue_name].pending -= 1
            self._stats[queue_name].processing += 1

        await self._persist_message(msg)

        try:
            result = handler(msg)
            if asyncio.iscoroutine(result):
                success = await result
            else:
                success = result

            if auto_ack:
                if success:
                    await self.ack(msg.id)
                else:
                    await self.nack(msg.id, "Handler returned False")
        except Exception as e:
            logger.exception("Handler error for message %s", msg.id)
            await self.nack(msg.id, str(e))

    async def ack(self, message_id: UUID) -> bool:
        """Acknowledge successful message processing.

        Args:
            message_id: ID of the message to acknowledge.

        Returns:
            True if message was found and acknowledged.
        """
        async with self._lock:
            msg = self._processing.pop(message_id, None)
            if not msg:
                return False

            msg.mark_completed()
            self._ensure_stats(msg.queue_name)
            self._stats[msg.queue_name].processing -= 1
            self._stats[msg.queue_name].completed += 1
            self._stats[msg.queue_name].total_consumed += 1

        await self._delete_message(message_id)
        logger.debug("Acked message %s", message_id)
        return True

    async def nack(self, message_id: UUID, error: str = "") -> bool:
        """Negative acknowledge - mark message for retry or dead letter.

        Args:
            message_id: ID of the message to nack.
            error: Error message describing the failure.

        Returns:
            True if message was found and nacked.
        """
        async with self._lock:
            msg = self._processing.pop(message_id, None)
            if not msg:
                return False

            msg.mark_failed(error)
            self._ensure_stats(msg.queue_name)
            self._stats[msg.queue_name].processing -= 1

            if msg.status == MessageStatus.DEAD:
                # Move to dead letter queue
                self._dead_letter[msg.queue_name].append(msg)
                self._stats[msg.queue_name].dead += 1
                logger.warning(
                    "Message %s moved to dead letter queue after %d attempts",
                    message_id,
                    msg.attempts,
                )
            else:
                # Requeue for retry
                msg.reset_for_retry()
                self._queues[msg.queue_name].append(msg)
                self._stats[msg.queue_name].pending += 1
                self._stats[msg.queue_name].failed += 1
                logger.debug(
                    "Message %s requeued for retry (attempt %d/%d)",
                    message_id,
                    msg.attempts,
                    msg.max_attempts,
                )

        await self._persist_message(msg)
        return True

    def _ensure_stats(self, queue_name: str) -> None:
        """Ensure stats exist for a queue."""
        if queue_name not in self._stats:
            self._stats[queue_name] = QueueStats(queue_name=queue_name)

    def get_stats(self, queue_name: str) -> QueueStats:
        """Get statistics for a queue.

        Args:
            queue_name: Name of the queue.

        Returns:
            QueueStats for the queue.
        """
        self._ensure_stats(queue_name)
        return self._stats[queue_name]

    def list_queues(self) -> list[str]:
        """Get names of all queues with messages or stats.

        Returns:
            List of queue names.
        """
        queues = set(self._queues.keys())
        queues.update(self._dead_letter.keys())
        queues.update(self._stats.keys())
        return sorted(queues)

    def get_pending_count(self, queue_name: str) -> int:
        """Get the number of pending messages in a queue."""
        return len(self._queues.get(queue_name, []))

    def get_dead_letter_count(self, queue_name: str) -> int:
        """Get the number of messages in the dead letter queue."""
        return len(self._dead_letter.get(queue_name, []))

    async def get_dead_letters(self, queue_name: str) -> list[Message]:
        """Get all dead letter messages for a queue.

        Args:
            queue_name: Name of the queue.

        Returns:
            List of dead letter messages.
        """
        return list(self._dead_letter.get(queue_name, []))

    async def retry_dead_letter(self, message_id: UUID) -> bool:
        """Retry a dead letter message.

        Args:
            message_id: ID of the dead letter message.

        Returns:
            True if message was found and requeued.
        """
        async with self._lock:
            for queue_name, messages in self._dead_letter.items():
                for i, msg in enumerate(messages):
                    if msg.id == message_id:
                        messages.pop(i)
                        msg.status = MessageStatus.PENDING
                        msg.attempts = 0
                        msg.error = None
                        msg.updated_at = datetime.now(UTC)
                        self._queues[queue_name].append(msg)
                        self._stats[queue_name].dead -= 1
                        self._stats[queue_name].pending += 1
                        await self._persist_message(msg)
                        logger.info("Retried dead letter message %s", message_id)
                        return True
        return False

    async def purge(self, queue_name: str, *, include_dead_letter: bool = False) -> int:
        """Purge all messages from a queue.

        Args:
            queue_name: Name of the queue to purge.
            include_dead_letter: If True, also purge dead letter queue.

        Returns:
            Number of messages purged.
        """
        count = 0
        async with self._lock:
            if queue_name in self._queues:
                count += len(self._queues[queue_name])
                self._queues[queue_name].clear()
            if include_dead_letter and queue_name in self._dead_letter:
                count += len(self._dead_letter[queue_name])
                self._dead_letter[queue_name].clear()
            if queue_name in self._stats:
                self._stats[queue_name] = QueueStats(queue_name=queue_name)

        if self._db:
            await self._db.execute(
                "DELETE FROM messages WHERE queue_name = ?",
                (queue_name,),
            )
            await self._db.commit()

        logger.info("Purged %d messages from queue %s", count, queue_name)
        return count

    async def close(self) -> None:
        """Close the queue and release resources."""
        self._running = False
        for task in self._consumer_tasks:
            task.cancel()
        self._consumer_tasks.clear()

        if self._db:
            await self._db.close()
            self._db = None


# Global message queue instance
_global_queue: MessageQueue | None = None


async def get_message_queue() -> MessageQueue:
    """Get or create the global message queue.

    Returns:
        The global MessageQueue instance.
    """
    global _global_queue  # noqa: PLW0603
    if _global_queue is None:
        _global_queue = MessageQueue()
        await _global_queue.initialize()
    return _global_queue


async def reset_message_queue() -> None:
    """Reset the global message queue. Useful for testing."""
    global _global_queue  # noqa: PLW0603
    if _global_queue:
        await _global_queue.close()
    _global_queue = None
