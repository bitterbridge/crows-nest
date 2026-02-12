"""Memory store protocol and base implementation.

Defines the interface for memory storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from crows_nest.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryScope,
    MemoryStats,
    MemoryType,
)

if TYPE_CHECKING:
    from uuid import UUID


class MemoryStore(ABC):
    """Abstract base class for memory storage.

    Implementations provide persistence for agent memories with
    support for different scopes, types, and query patterns.

    Usage:
        store = SQLiteMemoryStore(path="memories.db")
        await store.initialize()

        # Remember something
        entry = await store.remember(
            agent_id=agent.id,
            content="The user prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )

        # Recall memories
        memories = await store.recall(
            agent_id=agent.id,
            query=MemoryQuery(text="user preferences"),
        )

        # Forget a memory
        await store.forget(agent_id=agent.id, memory_id=entry.id)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (create tables, indexes, etc.)."""

    @abstractmethod
    async def close(self) -> None:
        """Close the store and release resources."""

    @abstractmethod
    async def remember(
        self,
        agent_id: UUID,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        scope: MemoryScope = MemoryScope.PRIVATE,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryEntry:
        """Store a new memory.

        Args:
            agent_id: ID of the agent storing this memory.
            content: The memory content.
            memory_type: Type of memory (episodic, semantic, procedural).
            scope: Visibility scope for the memory.
            importance: Importance score (0-1).
            metadata: Additional structured data.
            embedding: Optional vector embedding for semantic search.

        Returns:
            The created MemoryEntry.
        """

    @abstractmethod
    async def recall(
        self,
        agent_id: UUID,
        query: MemoryQuery,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching a query.

        Args:
            agent_id: ID of the agent requesting memories.
            query: Query parameters for filtering.

        Returns:
            List of matching memories, ordered by relevance.
        """

    @abstractmethod
    async def get(
        self,
        agent_id: UUID,
        memory_id: UUID,
    ) -> MemoryEntry | None:
        """Get a specific memory by ID.

        Args:
            agent_id: ID of the agent requesting the memory.
            memory_id: ID of the memory to retrieve.

        Returns:
            The memory entry, or None if not found or not accessible.
        """

    @abstractmethod
    async def update(
        self,
        agent_id: UUID,
        memory_id: UUID,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        """Update an existing memory.

        Args:
            agent_id: ID of the agent owning the memory.
            memory_id: ID of the memory to update.
            content: New content (if provided).
            importance: New importance (if provided).
            metadata: New metadata (if provided, replaces existing).

        Returns:
            The updated memory, or None if not found.
        """

    @abstractmethod
    async def forget(
        self,
        agent_id: UUID,
        memory_id: UUID,
    ) -> bool:
        """Delete a memory.

        Args:
            agent_id: ID of the agent owning the memory.
            memory_id: ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """

    @abstractmethod
    async def forget_all(
        self,
        agent_id: UUID,
        memory_type: MemoryType | None = None,
        older_than: float | None = None,
    ) -> int:
        """Delete multiple memories.

        Args:
            agent_id: ID of the agent owning the memories.
            memory_type: Only delete memories of this type (optional).
            older_than: Only delete memories older than this many seconds.

        Returns:
            Number of memories deleted.
        """

    @abstractmethod
    async def decay(
        self,
        agent_id: UUID,
        threshold: float = 0.1,
        half_life_seconds: float = 86400.0,
    ) -> int:
        """Remove memories with decay score below threshold.

        Args:
            agent_id: ID of the agent.
            threshold: Minimum decay score to keep.
            half_life_seconds: Decay half-life for score calculation.

        Returns:
            Number of memories removed.
        """

    @abstractmethod
    async def get_stats(self, agent_id: UUID) -> MemoryStats:
        """Get statistics about an agent's memories.

        Args:
            agent_id: ID of the agent.

        Returns:
            MemoryStats for the agent.
        """

    @abstractmethod
    async def export_memories(
        self,
        agent_id: UUID,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Export all memories for an agent.

        Args:
            agent_id: ID of the agent.
            include_embeddings: Whether to include vector embeddings.

        Returns:
            List of memory dictionaries.
        """

    @abstractmethod
    async def import_memories(
        self,
        agent_id: UUID,
        memories: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> int:
        """Import memories for an agent.

        Args:
            agent_id: ID of the agent.
            memories: List of memory dictionaries.
            overwrite: If True, overwrite existing memories with same ID.

        Returns:
            Number of memories imported.
        """

    async def search_semantic(
        self,
        _agent_id: UUID,
        _embedding: list[float],
        _limit: int = 10,
        _threshold: float = 0.7,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories by vector similarity.

        Default implementation returns empty list (no vector support).
        Override in backends that support vector search.

        Args:
            _agent_id: ID of the agent.
            _embedding: Query embedding vector.
            _limit: Maximum results.
            _threshold: Minimum similarity score.

        Returns:
            List of (memory, similarity_score) tuples.
        """
        return []

    def can_access(
        self,
        requesting_agent: UUID,
        memory: MemoryEntry,
        agent_ancestors: list[UUID] | None = None,
    ) -> bool:
        """Check if an agent can access a memory.

        Args:
            requesting_agent: Agent requesting access.
            memory: The memory to check.
            agent_ancestors: Ancestors of the requesting agent (for scope checks).

        Returns:
            True if the agent can access the memory.
        """
        # Owner can always access
        if memory.agent_id == requesting_agent:
            return True

        # Check scope
        if memory.scope == MemoryScope.GLOBAL:
            return True

        # Requester must be a descendant of the memory owner for CHILDREN scope
        return (
            memory.scope == MemoryScope.CHILDREN
            and agent_ancestors is not None
            and memory.agent_id in agent_ancestors
        )
