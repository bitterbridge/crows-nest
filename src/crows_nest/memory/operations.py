"""Thunk operations for memory system interaction.

Provides operations for remembering, recalling, and managing agent memories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from crows_nest.core.registry import thunk_operation
from crows_nest.memory.models import MemoryQuery, MemoryScope, MemoryType
from crows_nest.memory.sqlite import SQLiteMemoryStore

if TYPE_CHECKING:
    from crows_nest.memory.store import MemoryStore

# Global memory store instance (lazily initialized)
_memory_store: MemoryStore | None = None


async def get_memory_store() -> MemoryStore:
    """Get the global memory store instance.

    Returns:
        The memory store instance.
    """
    global _memory_store  # noqa: PLW0603
    if _memory_store is None:
        _memory_store = SQLiteMemoryStore()
        await _memory_store.initialize()
    return _memory_store


async def set_memory_store(store: MemoryStore) -> None:
    """Set the global memory store instance.

    Args:
        store: The memory store to use.
    """
    global _memory_store  # noqa: PLW0603
    _memory_store = store


@thunk_operation(
    name="memory.remember",
    description="Store a new memory for an agent.",
    required_capabilities=frozenset({"memory.write"}),
)
async def memory_remember(
    agent_id: str,
    content: str,
    memory_type: str = "semantic",
    scope: str = "private",
    importance: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a new memory.

    Args:
        agent_id: ID of the agent storing this memory.
        content: The memory content.
        memory_type: Type of memory (episodic, semantic, procedural).
        scope: Visibility scope (private, children, global).
        importance: Importance score (0-1).
        metadata: Additional structured data.

    Returns:
        Dict with memory ID and creation time.
    """
    store = await get_memory_store()
    entry = await store.remember(
        agent_id=UUID(agent_id),
        content=content,
        memory_type=MemoryType(memory_type),
        scope=MemoryScope(scope),
        importance=importance,
        metadata=metadata,
    )

    return {
        "memory_id": str(entry.id),
        "agent_id": agent_id,
        "created_at": entry.created_at.isoformat(),
    }


@thunk_operation(
    name="memory.recall",
    description="Retrieve memories matching a query.",
    required_capabilities=frozenset({"memory.read"}),
)
async def memory_recall(
    agent_id: str,
    text: str | None = None,
    memory_type: str | None = None,
    scope: str | None = None,
    min_importance: float = 0.0,
    limit: int = 10,
    include_expired: bool = False,
) -> dict[str, Any]:
    """Retrieve memories matching a query.

    Args:
        agent_id: ID of the agent requesting memories.
        text: Text to search for.
        memory_type: Filter by memory type.
        scope: Filter by scope.
        min_importance: Minimum importance threshold.
        limit: Maximum number of results.
        include_expired: Whether to include expired memories.

    Returns:
        Dict with list of matching memories.
    """
    store = await get_memory_store()
    query = MemoryQuery(
        text=text,
        memory_type=MemoryType(memory_type) if memory_type else None,
        scope=MemoryScope(scope) if scope else None,
        min_importance=min_importance,
        limit=limit,
        include_expired=include_expired,
    )

    entries = await store.recall(agent_id=UUID(agent_id), query=query)

    return {
        "agent_id": agent_id,
        "count": len(entries),
        "memories": [
            {
                "id": str(entry.id),
                "content": entry.content,
                "memory_type": entry.memory_type.value,
                "scope": entry.scope.value,
                "importance": entry.importance,
                "created_at": entry.created_at.isoformat(),
                "accessed_at": entry.accessed_at.isoformat(),
                "access_count": entry.access_count,
                "metadata": entry.metadata,
            }
            for entry in entries
        ],
    }


@thunk_operation(
    name="memory.get",
    description="Get a specific memory by ID.",
    required_capabilities=frozenset({"memory.read"}),
)
async def memory_get(
    agent_id: str,
    memory_id: str,
) -> dict[str, Any]:
    """Get a specific memory by ID.

    Args:
        agent_id: ID of the agent requesting the memory.
        memory_id: ID of the memory to retrieve.

    Returns:
        Dict with memory details, or error if not found.
    """
    store = await get_memory_store()
    entry = await store.get(agent_id=UUID(agent_id), memory_id=UUID(memory_id))

    if entry is None:
        return {
            "found": False,
            "memory_id": memory_id,
            "error": "Memory not found or not accessible",
        }

    return {
        "found": True,
        "memory": {
            "id": str(entry.id),
            "agent_id": str(entry.agent_id),
            "content": entry.content,
            "memory_type": entry.memory_type.value,
            "scope": entry.scope.value,
            "importance": entry.importance,
            "created_at": entry.created_at.isoformat(),
            "accessed_at": entry.accessed_at.isoformat(),
            "access_count": entry.access_count,
            "metadata": entry.metadata,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        },
    }


@thunk_operation(
    name="memory.update",
    description="Update an existing memory.",
    required_capabilities=frozenset({"memory.write"}),
)
async def memory_update(
    agent_id: str,
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update an existing memory.

    Args:
        agent_id: ID of the agent owning the memory.
        memory_id: ID of the memory to update.
        content: New content (if provided).
        importance: New importance (if provided).
        metadata: New metadata (if provided, replaces existing).

    Returns:
        Dict with updated memory details.
    """
    store = await get_memory_store()
    entry = await store.update(
        agent_id=UUID(agent_id),
        memory_id=UUID(memory_id),
        content=content,
        importance=importance,
        metadata=metadata,
    )

    if entry is None:
        return {
            "updated": False,
            "memory_id": memory_id,
            "error": "Memory not found",
        }

    return {
        "updated": True,
        "memory": {
            "id": str(entry.id),
            "content": entry.content,
            "importance": entry.importance,
            "metadata": entry.metadata,
        },
    }


@thunk_operation(
    name="memory.forget",
    description="Delete a memory.",
    required_capabilities=frozenset({"memory.write"}),
)
async def memory_forget(
    agent_id: str,
    memory_id: str,
) -> dict[str, Any]:
    """Delete a memory.

    Args:
        agent_id: ID of the agent owning the memory.
        memory_id: ID of the memory to delete.

    Returns:
        Dict with deletion status.
    """
    store = await get_memory_store()
    deleted = await store.forget(agent_id=UUID(agent_id), memory_id=UUID(memory_id))

    return {
        "deleted": deleted,
        "memory_id": memory_id,
    }


@thunk_operation(
    name="memory.forget_all",
    description="Delete multiple memories.",
    required_capabilities=frozenset({"memory.admin"}),
)
async def memory_forget_all(
    agent_id: str,
    memory_type: str | None = None,
    older_than_seconds: float | None = None,
) -> dict[str, Any]:
    """Delete multiple memories.

    Args:
        agent_id: ID of the agent owning the memories.
        memory_type: Only delete memories of this type (optional).
        older_than_seconds: Only delete memories older than this many seconds.

    Returns:
        Dict with number of memories deleted.
    """
    store = await get_memory_store()
    count = await store.forget_all(
        agent_id=UUID(agent_id),
        memory_type=MemoryType(memory_type) if memory_type else None,
        older_than=older_than_seconds,
    )

    return {
        "agent_id": agent_id,
        "deleted_count": count,
    }


@thunk_operation(
    name="memory.decay",
    description="Remove memories with low decay scores.",
    required_capabilities=frozenset({"memory.admin"}),
)
async def memory_decay(
    agent_id: str,
    threshold: float = 0.1,
    half_life_seconds: float = 86400.0,
) -> dict[str, Any]:
    """Remove memories with decay score below threshold.

    Args:
        agent_id: ID of the agent.
        threshold: Minimum decay score to keep.
        half_life_seconds: Decay half-life for score calculation.

    Returns:
        Dict with number of memories removed.
    """
    store = await get_memory_store()
    count = await store.decay(
        agent_id=UUID(agent_id),
        threshold=threshold,
        half_life_seconds=half_life_seconds,
    )

    return {
        "agent_id": agent_id,
        "decayed_count": count,
        "threshold": threshold,
        "half_life_seconds": half_life_seconds,
    }


@thunk_operation(
    name="memory.stats",
    description="Get statistics about an agent's memories.",
    required_capabilities=frozenset({"memory.read"}),
)
async def memory_stats(agent_id: str) -> dict[str, Any]:
    """Get statistics about an agent's memories.

    Args:
        agent_id: ID of the agent.

    Returns:
        Dict with memory statistics.
    """
    store = await get_memory_store()
    stats = await store.get_stats(agent_id=UUID(agent_id))

    return stats.to_dict()


@thunk_operation(
    name="memory.export",
    description="Export all memories for an agent.",
    required_capabilities=frozenset({"memory.admin"}),
)
async def memory_export(
    agent_id: str,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    """Export all memories for an agent.

    Args:
        agent_id: ID of the agent.
        include_embeddings: Whether to include vector embeddings.

    Returns:
        Dict with list of memory dictionaries.
    """
    store = await get_memory_store()
    memories = await store.export_memories(
        agent_id=UUID(agent_id),
        include_embeddings=include_embeddings,
    )

    return {
        "agent_id": agent_id,
        "count": len(memories),
        "memories": memories,
    }


@thunk_operation(
    name="memory.import",
    description="Import memories for an agent.",
    required_capabilities=frozenset({"memory.admin"}),
)
async def memory_import(
    agent_id: str,
    memories: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import memories for an agent.

    Args:
        agent_id: ID of the agent.
        memories: List of memory dictionaries.
        overwrite: If True, overwrite existing memories with same ID.

    Returns:
        Dict with number of memories imported.
    """
    store = await get_memory_store()
    count = await store.import_memories(
        agent_id=UUID(agent_id),
        memories=memories,
        overwrite=overwrite,
    )

    return {
        "agent_id": agent_id,
        "imported_count": count,
        "overwrite": overwrite,
    }
