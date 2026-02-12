"""Thunk operations for message queue interaction.

Provides operations for publishing and consuming messages via thunks.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from crows_nest.core.registry import thunk_operation
from crows_nest.messaging.queue import get_message_queue


@thunk_operation(
    name="queue.publish",
    description="Publish a message to a named queue.",
    required_capabilities=frozenset({"queue.publish"}),
)
async def queue_publish(
    queue_name: str,
    payload: dict[str, Any],
    max_attempts: int = 3,
    publisher_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Publish a message to a queue.

    Args:
        queue_name: Name of the target queue.
        payload: Message payload (JSON-serializable dict).
        max_attempts: Maximum delivery attempts before dead letter.
        publisher_id: ID of the publishing thunk/agent (as string).
        trace_id: Trace ID for correlation.

    Returns:
        Dict with message_id and queue_name.
    """
    queue = await get_message_queue()
    pub_id: UUID | None = None
    if publisher_id:
        pub_id = UUID(publisher_id)

    msg = await queue.publish(
        queue_name=queue_name,
        payload=payload,
        max_attempts=max_attempts,
        publisher_id=pub_id,
        trace_id=trace_id,
    )

    return {
        "message_id": str(msg.id),
        "queue_name": queue_name,
        "status": msg.status.value,
    }


@thunk_operation(
    name="queue.list",
    description="List all available queues.",
    required_capabilities=frozenset({"queue.read"}),
)
async def queue_list() -> dict[str, Any]:
    """List all queues with their message counts.

    Returns:
        Dict with list of queues and their stats.
    """
    queue = await get_message_queue()
    queues = queue.list_queues()

    queue_info = []
    for name in queues:
        stats = queue.get_stats(name)
        queue_info.append(
            {
                "name": name,
                "pending": stats.pending,
                "processing": stats.processing,
                "dead": stats.dead,
            }
        )

    return {"queues": queue_info}


@thunk_operation(
    name="queue.stats",
    description="Get detailed statistics for a queue.",
    required_capabilities=frozenset({"queue.read"}),
)
async def queue_stats(queue_name: str) -> dict[str, Any]:
    """Get detailed statistics for a queue.

    Args:
        queue_name: Name of the queue.

    Returns:
        Dict with queue statistics.
    """
    queue = await get_message_queue()
    stats = queue.get_stats(queue_name)
    return stats.to_dict()


@thunk_operation(
    name="queue.depth",
    description="Get the current depth (pending message count) of a queue.",
    required_capabilities=frozenset({"queue.read"}),
)
async def queue_depth(queue_name: str) -> dict[str, Any]:
    """Get the pending message count for a queue.

    Args:
        queue_name: Name of the queue.

    Returns:
        Dict with queue name and depth.
    """
    queue = await get_message_queue()
    depth = queue.get_pending_count(queue_name)
    return {"queue_name": queue_name, "depth": depth}


@thunk_operation(
    name="queue.dead_letters",
    description="Get dead letter messages for a queue.",
    required_capabilities=frozenset({"queue.read"}),
)
async def queue_dead_letters(queue_name: str) -> dict[str, Any]:
    """Get dead letter messages for a queue.

    Args:
        queue_name: Name of the queue.

    Returns:
        Dict with list of dead letter messages.
    """
    queue = await get_message_queue()
    messages = await queue.get_dead_letters(queue_name)

    return {
        "queue_name": queue_name,
        "count": len(messages),
        "messages": [
            {
                "id": str(msg.id),
                "payload": msg.payload,
                "attempts": msg.attempts,
                "error": msg.error,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ],
    }


@thunk_operation(
    name="queue.retry_dead_letter",
    description="Retry a dead letter message.",
    required_capabilities=frozenset({"queue.write"}),
)
async def queue_retry_dead_letter(message_id: str) -> dict[str, Any]:
    """Retry a dead letter message.

    Args:
        message_id: ID of the dead letter message to retry.

    Returns:
        Dict with success status.
    """
    queue = await get_message_queue()
    success = await queue.retry_dead_letter(UUID(message_id))

    return {
        "message_id": message_id,
        "retried": success,
    }


@thunk_operation(
    name="queue.purge",
    description="Purge all messages from a queue.",
    required_capabilities=frozenset({"queue.admin"}),
)
async def queue_purge(
    queue_name: str,
    include_dead_letter: bool = False,
) -> dict[str, Any]:
    """Purge all messages from a queue.

    Args:
        queue_name: Name of the queue to purge.
        include_dead_letter: If True, also purge dead letter queue.

    Returns:
        Dict with number of messages purged.
    """
    queue = await get_message_queue()
    count = await queue.purge(queue_name, include_dead_letter=include_dead_letter)

    return {
        "queue_name": queue_name,
        "purged": count,
        "included_dead_letter": include_dead_letter,
    }
