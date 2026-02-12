"""Agent hierarchy and lineage tracking.

Manages parent-child relationships between agents, enabling the
agent-creates-agent pattern with controlled capability inheritance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import aiosqlite

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AgentHierarchyError(Exception):
    """Base exception for agent hierarchy errors."""


class AgentNotFoundError(AgentHierarchyError):
    """Agent not found in registry."""

    def __init__(self, agent_id: UUID) -> None:
        self.agent_id = agent_id
        super().__init__(f"Agent not found: {agent_id}")


class MaxDepthExceededError(AgentHierarchyError):
    """Maximum hierarchy depth exceeded."""

    def __init__(self, max_depth: int, current_depth: int) -> None:
        self.max_depth = max_depth
        self.current_depth = current_depth
        super().__init__(f"Maximum depth {max_depth} exceeded (current: {current_depth})")


class MaxChildrenExceededError(AgentHierarchyError):
    """Maximum children count exceeded."""

    def __init__(self, max_children: int, parent_id: UUID) -> None:
        self.max_children = max_children
        self.parent_id = parent_id
        super().__init__(f"Maximum children {max_children} exceeded for agent {parent_id}")


@dataclass
class AgentLineage:
    """Tracks agent family relationships.

    Attributes:
        agent_id: Unique identifier for this agent.
        parent_id: ID of parent agent, None if root agent.
        children: List of child agent IDs.
        depth: Depth in hierarchy (root = 0).
        created_at: When this agent was created.
        terminated_at: When this agent was terminated, None if active.
        metadata: Additional agent metadata.
    """

    agent_id: UUID
    parent_id: UUID | None = None
    children: list[UUID] = field(default_factory=list)
    depth: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    terminated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if agent is still active."""
        return self.terminated_at is None

    @property
    def is_root(self) -> bool:
        """Check if this is a root agent (no parent)."""
        return self.parent_id is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "children": [str(c) for c in self.children],
            "depth": self.depth,
            "created_at": self.created_at.isoformat(),
            "terminated_at": (self.terminated_at.isoformat() if self.terminated_at else None),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentLineage:
        """Create from dictionary."""
        return cls(
            agent_id=UUID(data["agent_id"]),
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
            children=[UUID(c) for c in data.get("children", [])],
            depth=data.get("depth", 0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now(UTC))
            ),
            terminated_at=(
                datetime.fromisoformat(data["terminated_at"]) if data.get("terminated_at") else None
            ),
            metadata=data.get("metadata", {}),
        )


class AgentRegistry:
    """Tracks all agents and their relationships.

    Provides methods for registering agents, querying lineage,
    and managing agent lifecycle.

    Usage:
        registry = AgentRegistry()
        await registry.initialize()

        # Register a root agent
        root = await registry.register(agent_id=uuid4())

        # Spawn a child
        child = await registry.register(agent_id=uuid4(), parent_id=root.agent_id)

        # Get lineage
        lineage = await registry.get_lineage(child.agent_id)
        ancestors = await registry.get_ancestors(child.agent_id)

        # Terminate
        await registry.terminate(root.agent_id, cascade=True)
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        max_depth: int = 5,
        max_children: int = 10,
    ) -> None:
        """Initialize the agent registry.

        Args:
            persistence_path: Path to SQLite database for persistence.
            max_depth: Maximum allowed hierarchy depth.
            max_children: Maximum children per agent.
        """
        self._agents: dict[UUID, AgentLineage] = {}
        self._persistence_path = persistence_path
        self._db: aiosqlite.Connection | None = None
        self.max_depth = max_depth
        self.max_children = max_children

    async def initialize(self) -> None:
        """Initialize the registry, setting up persistence if configured."""
        if self._persistence_path:
            await self._init_persistence()

    async def _init_persistence(self) -> None:
        """Initialize SQLite persistence."""
        if not self._persistence_path:
            return

        self._db = await aiosqlite.connect(self._persistence_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS agent_lineage (
                agent_id TEXT PRIMARY KEY,
                parent_id TEXT,
                children TEXT NOT NULL,
                depth INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                terminated_at TEXT,
                metadata TEXT NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_lineage_parent
            ON agent_lineage(parent_id)
        """)
        await self._db.commit()

        # Load existing agents
        await self._load_agents()

    async def _load_agents(self) -> None:
        """Load agents from persistence."""
        if not self._db:
            return

        async with self._db.execute("SELECT * FROM agent_lineage") as cursor:
            async for row in cursor:
                lineage = AgentLineage(
                    agent_id=UUID(row[0]),
                    parent_id=UUID(row[1]) if row[1] else None,
                    children=[UUID(c) for c in json.loads(row[2])],
                    depth=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    terminated_at=(datetime.fromisoformat(row[5]) if row[5] else None),
                    metadata=json.loads(row[6]),
                )
                self._agents[lineage.agent_id] = lineage

    async def _persist_agent(self, lineage: AgentLineage) -> None:
        """Persist an agent to the database."""
        if not self._db:
            return

        await self._db.execute(
            """
            INSERT OR REPLACE INTO agent_lineage
            (agent_id, parent_id, children, depth, created_at, terminated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(lineage.agent_id),
                str(lineage.parent_id) if lineage.parent_id else None,
                json.dumps([str(c) for c in lineage.children]),
                lineage.depth,
                lineage.created_at.isoformat(),
                lineage.terminated_at.isoformat() if lineage.terminated_at else None,
                json.dumps(lineage.metadata),
            ),
        )
        await self._db.commit()

    async def register(
        self,
        agent_id: UUID | None = None,
        parent_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentLineage:
        """Register a new agent in the hierarchy.

        Args:
            agent_id: ID for the new agent, generated if not provided.
            parent_id: ID of parent agent, None for root agents.
            metadata: Additional metadata for the agent.

        Returns:
            The created AgentLineage.

        Raises:
            AgentNotFoundError: If parent_id doesn't exist.
            MaxDepthExceededError: If max hierarchy depth would be exceeded.
            MaxChildrenExceededError: If parent has too many children.
        """
        if agent_id is None:
            agent_id = uuid4()

        depth = 0
        if parent_id is not None:
            parent = self._agents.get(parent_id)
            if parent is None:
                raise AgentNotFoundError(parent_id)

            if not parent.is_active:
                raise AgentHierarchyError(f"Parent agent {parent_id} is terminated")

            depth = parent.depth + 1
            if depth > self.max_depth:
                raise MaxDepthExceededError(self.max_depth, depth)

            if len(parent.children) >= self.max_children:
                raise MaxChildrenExceededError(self.max_children, parent_id)

        lineage = AgentLineage(
            agent_id=agent_id,
            parent_id=parent_id,
            depth=depth,
            metadata=metadata or {},
        )

        self._agents[agent_id] = lineage

        # Update parent's children list
        if parent_id is not None:
            parent = self._agents[parent_id]
            parent.children.append(agent_id)
            await self._persist_agent(parent)

        await self._persist_agent(lineage)
        logger.info(
            "Registered agent %s (parent=%s, depth=%d)",
            agent_id,
            parent_id,
            depth,
        )

        return lineage

    async def get_lineage(self, agent_id: UUID) -> AgentLineage:
        """Get lineage for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            The agent's lineage.

        Raises:
            AgentNotFoundError: If agent doesn't exist.
        """
        lineage = self._agents.get(agent_id)
        if lineage is None:
            raise AgentNotFoundError(agent_id)
        return lineage

    def get_lineage_sync(self, agent_id: UUID) -> AgentLineage | None:
        """Get lineage synchronously (no persistence lookup)."""
        return self._agents.get(agent_id)

    async def get_children(self, agent_id: UUID) -> list[AgentLineage]:
        """Get all children of an agent.

        Args:
            agent_id: ID of the parent agent.

        Returns:
            List of child lineages.

        Raises:
            AgentNotFoundError: If agent doesn't exist.
        """
        lineage = await self.get_lineage(agent_id)
        return [self._agents[c] for c in lineage.children if c in self._agents]

    async def get_ancestors(self, agent_id: UUID) -> list[AgentLineage]:
        """Get all ancestors of an agent (parent, grandparent, etc.).

        Args:
            agent_id: ID of the agent.

        Returns:
            List of ancestor lineages, ordered from parent to root.

        Raises:
            AgentNotFoundError: If agent doesn't exist.
        """
        lineage = await self.get_lineage(agent_id)
        ancestors = []

        current_id = lineage.parent_id
        while current_id is not None:
            ancestor = self._agents.get(current_id)
            if ancestor is None:
                break
            ancestors.append(ancestor)
            current_id = ancestor.parent_id

        return ancestors

    async def get_descendants(self, agent_id: UUID) -> list[AgentLineage]:
        """Get all descendants of an agent (children, grandchildren, etc.).

        Args:
            agent_id: ID of the agent.

        Returns:
            List of descendant lineages in breadth-first order.

        Raises:
            AgentNotFoundError: If agent doesn't exist.
        """
        lineage = await self.get_lineage(agent_id)
        descendants = []
        queue = list(lineage.children)

        while queue:
            child_id = queue.pop(0)
            child = self._agents.get(child_id)
            if child:
                descendants.append(child)
                queue.extend(child.children)

        return descendants

    async def terminate(
        self,
        agent_id: UUID,
        cascade: bool = True,
    ) -> list[UUID]:
        """Terminate an agent.

        Args:
            agent_id: ID of the agent to terminate.
            cascade: If True, also terminate all descendants.
                If False, orphan children (set their parent to None).

        Returns:
            List of terminated agent IDs.

        Raises:
            AgentNotFoundError: If agent doesn't exist.
        """
        lineage = await self.get_lineage(agent_id)
        terminated: list[UUID] = []

        if cascade:
            # Terminate all descendants first (depth-first)
            descendants = await self.get_descendants(agent_id)
            for desc in reversed(descendants):
                if desc.is_active:
                    desc.terminated_at = datetime.now(UTC)
                    terminated.append(desc.agent_id)
                    await self._persist_agent(desc)

        else:
            # Orphan children
            for child_id in lineage.children:
                child = self._agents.get(child_id)
                if child:
                    child.parent_id = None
                    child.depth = 0
                    await self._persist_agent(child)
                    # Recalculate depths for this subtree
                    await self._recalculate_depths(child_id)

        # Terminate the agent itself
        lineage.terminated_at = datetime.now(UTC)
        terminated.append(agent_id)
        await self._persist_agent(lineage)

        # Remove from parent's children list
        if lineage.parent_id is not None:
            parent = self._agents.get(lineage.parent_id)
            if parent and agent_id in parent.children:
                parent.children.remove(agent_id)
                await self._persist_agent(parent)

        logger.info(
            "Terminated agent %s (cascade=%s, total=%d)",
            agent_id,
            cascade,
            len(terminated),
        )

        return terminated

    async def _recalculate_depths(self, agent_id: UUID) -> None:
        """Recalculate depths for an agent and all descendants."""
        lineage = self._agents.get(agent_id)
        if not lineage:
            return

        # BFS to update depths
        queue = [(agent_id, lineage.depth)]
        while queue:
            current_id, expected_depth = queue.pop(0)
            current = self._agents.get(current_id)
            if current and current.depth != expected_depth:
                current.depth = expected_depth
                await self._persist_agent(current)

            if current:
                for child_id in current.children:
                    queue.append((child_id, expected_depth + 1))

    def list_agents(
        self,
        include_terminated: bool = False,
    ) -> list[AgentLineage]:
        """List all registered agents.

        Args:
            include_terminated: If True, include terminated agents.

        Returns:
            List of agent lineages.
        """
        agents = list(self._agents.values())
        if not include_terminated:
            agents = [a for a in agents if a.is_active]
        return agents

    def list_root_agents(self) -> list[AgentLineage]:
        """List all root agents (no parent)."""
        return [a for a in self._agents.values() if a.is_root and a.is_active]

    async def close(self) -> None:
        """Close the registry and release resources."""
        if self._db:
            await self._db.close()
            self._db = None


# Global registry instance
_global_registry: AgentRegistry | None = None


async def get_agent_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = AgentRegistry()
        await _global_registry.initialize()
    return _global_registry


async def reset_agent_registry() -> None:
    """Reset the global agent registry. Useful for testing."""
    global _global_registry  # noqa: PLW0603
    if _global_registry:
        await _global_registry.close()
    _global_registry = None
