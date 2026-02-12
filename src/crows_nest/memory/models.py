"""Memory models and types.

Core data structures for agent memory system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class MemoryType(StrEnum):
    """Type of memory content."""

    EPISODIC = "episodic"  # What happened (events, conversations)
    SEMANTIC = "semantic"  # What is true (facts, knowledge)
    PROCEDURAL = "procedural"  # How to do things (learned patterns)


class MemoryScope(StrEnum):
    """Visibility scope for memories."""

    PRIVATE = "private"  # Only this agent can access
    CHILDREN = "children"  # This agent and its descendants
    GLOBAL = "global"  # All agents can access


@dataclass
class MemoryEntry:
    """A single memory entry.

    Attributes:
        id: Unique identifier for this memory.
        agent_id: ID of the agent that owns this memory.
        memory_type: Type of memory (episodic, semantic, procedural).
        scope: Visibility scope for this memory.
        content: The actual memory content.
        metadata: Additional structured data.
        embedding: Vector embedding for semantic search (optional).
        created_at: When this memory was created.
        accessed_at: When this memory was last accessed.
        access_count: Number of times this memory was accessed.
        importance: Importance score (0-1), affects decay.
        expires_at: Optional expiration time.
    """

    agent_id: UUID
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    scope: MemoryScope = MemoryScope.PRIVATE
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    importance: float = 0.5
    expires_at: datetime | None = None

    def touch(self) -> None:
        """Update access time and increment counter."""
        self.accessed_at = datetime.now(UTC)
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of memory in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()

    @property
    def staleness_seconds(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.now(UTC) - self.accessed_at).total_seconds()

    def decay_score(self, half_life_seconds: float = 86400.0) -> float:
        """Calculate decay score based on staleness and importance.

        Uses exponential decay with importance as a modifier.
        Higher importance = slower decay.

        Args:
            half_life_seconds: Time for score to decay by half (default: 1 day).

        Returns:
            Decay score between 0 and 1.
        """
        staleness = self.staleness_seconds
        # Importance extends effective half-life
        effective_half_life = half_life_seconds * (1 + self.importance)
        decay = math.exp(-0.693 * staleness / effective_half_life)
        return decay * self.importance

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            memory_type=MemoryType(data.get("memory_type", "semantic")),
            scope=MemoryScope(data.get("scope", "private")),
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now(UTC))
            ),
            accessed_at=(
                datetime.fromisoformat(data["accessed_at"])
                if isinstance(data.get("accessed_at"), str)
                else data.get("accessed_at", datetime.now(UTC))
            ),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
        )


@dataclass
class MemoryQuery:
    """Query parameters for memory retrieval.

    Attributes:
        text: Text to search for (semantic or keyword).
        agent_id: Filter by agent (required for scoped queries).
        memory_type: Filter by memory type.
        scope: Filter by scope.
        min_importance: Minimum importance threshold.
        time_range: Filter by creation time range.
        include_expired: Whether to include expired memories.
        limit: Maximum number of results.
        offset: Number of results to skip.
    """

    text: str | None = None
    agent_id: UUID | None = None
    memory_type: MemoryType | None = None
    scope: MemoryScope | None = None
    min_importance: float = 0.0
    time_range: tuple[datetime, datetime] | None = None
    include_expired: bool = False
    limit: int = 10
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "memory_type": self.memory_type.value if self.memory_type else None,
            "scope": self.scope.value if self.scope else None,
            "min_importance": self.min_importance,
            "time_range": (
                [self.time_range[0].isoformat(), self.time_range[1].isoformat()]
                if self.time_range
                else None
            ),
            "include_expired": self.include_expired,
            "limit": self.limit,
            "offset": self.offset,
        }


@dataclass
class MemoryStats:
    """Statistics about an agent's memory.

    Attributes:
        agent_id: The agent these stats are for.
        total_count: Total number of memories.
        by_type: Count by memory type.
        by_scope: Count by scope.
        avg_importance: Average importance score.
        oldest: Timestamp of oldest memory.
        newest: Timestamp of newest memory.
        total_size_bytes: Approximate storage size.
    """

    agent_id: UUID
    total_count: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_scope: dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    oldest: datetime | None = None
    newest: datetime | None = None
    total_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": str(self.agent_id),
            "total_count": self.total_count,
            "by_type": self.by_type,
            "by_scope": self.by_scope,
            "avg_importance": self.avg_importance,
            "oldest": self.oldest.isoformat() if self.oldest else None,
            "newest": self.newest.isoformat() if self.newest else None,
            "total_size_bytes": self.total_size_bytes,
        }
