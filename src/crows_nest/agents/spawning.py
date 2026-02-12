"""Agent spawning operations.

Thunk operations for creating, managing, and terminating agent hierarchies.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from crows_nest.agents.budget import (
    AgentResources,
    CapabilityBudget,
    ResourceQuota,
)
from crows_nest.agents.hierarchy import (
    get_agent_registry,
)
from crows_nest.core.registry import thunk_operation
from crows_nest.messaging import (
    AgentSpawnedEvent,
    get_event_bus,
)

# In-memory store for agent resources (would be persisted in production)
_agent_resources: dict[UUID, AgentResources] = {}


async def get_agent_resources(agent_id: UUID) -> AgentResources | None:
    """Get resources for an agent."""
    return _agent_resources.get(agent_id)


async def set_agent_resources(agent_id: UUID, resources: AgentResources) -> None:
    """Set resources for an agent."""
    _agent_resources[agent_id] = resources


async def remove_agent_resources(agent_id: UUID) -> None:
    """Remove resources for an agent."""
    _agent_resources.pop(agent_id, None)


def reset_agent_resources() -> None:
    """Reset all agent resources. For testing."""
    _agent_resources.clear()


@thunk_operation(
    name="agent.hierarchy.spawn",
    description="Spawn a child agent with hierarchy tracking and resource allocation.",
    required_capabilities=frozenset({"agent.spawn"}),
)
async def spawn_agent(
    parent_id: str,
    capabilities: list[str],
    metadata: dict[str, Any] | None = None,
    max_children: int | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Spawn a child agent with allocated capabilities.

    Args:
        parent_id: ID of the parent agent.
        capabilities: Capabilities to grant the child.
        metadata: Optional metadata for the child agent.
        max_children: Override max children for child's quota.
        max_depth: Override max depth for child's quota.

    Returns:
        Dict with child_id, capabilities, and quota info.

    Raises:
        ValueError: If parent not found or resources insufficient.
    """
    parent_uuid = UUID(parent_id)
    registry = await get_agent_registry()

    # Get parent resources
    parent_resources = await get_agent_resources(parent_uuid)
    if parent_resources is None:
        raise ValueError(f"Parent agent {parent_id} not found or has no resources")

    # Prepare capabilities
    requested_caps = frozenset(capabilities)

    # Create child quota if custom values provided
    child_quota = None
    if max_children is not None or max_depth is not None:
        child_quota = parent_resources.quota.create_child_quota(
            max_children=max_children,
            max_depth=max_depth,
        )

    # Spawn the child (this validates and allocates resources)
    child_id = UUID(int=0)  # Will be replaced
    child_resources = parent_resources.spawn_child(
        child_id=child_id,
        capabilities=requested_caps,
        child_quota=child_quota,
    )

    # Register in hierarchy
    child_lineage = await registry.register(
        parent_id=parent_uuid,
        metadata=metadata or {},
    )
    child_id = child_lineage.agent_id

    # Update child resources with actual ID
    child_resources.agent_id = child_id
    # Re-register the capability allocation with correct child ID
    parent_resources.capabilities.deallocate(UUID(int=0))
    parent_resources.capabilities.allocate(child_id, requested_caps)

    await set_agent_resources(child_id, child_resources)

    # Emit spawn event
    bus = get_event_bus()
    await bus.publish(
        AgentSpawnedEvent(
            parent_agent_id=parent_uuid,
            child_agent_id=child_id,
            child_spec_name=metadata.get("spec_name", "unknown") if metadata else "unknown",
            capabilities=list(requested_caps),
        )
    )

    return {
        "child_id": str(child_id),
        "parent_id": parent_id,
        "capabilities": sorted(requested_caps),
        "depth": child_lineage.depth,
        "quota": child_resources.quota.to_dict(),
    }


@thunk_operation(
    name="agent.hierarchy.terminate",
    description="Terminate an agent and optionally its children.",
    required_capabilities=frozenset({"agent.terminate"}),
)
async def terminate_agent(
    agent_id: str,
    cascade: bool = True,
) -> dict[str, Any]:
    """Terminate an agent.

    Args:
        agent_id: ID of the agent to terminate.
        cascade: If True, also terminate all descendants.

    Returns:
        Dict with terminated agent IDs.
    """
    agent_uuid = UUID(agent_id)
    registry = await get_agent_registry()

    # Get lineage to find parent
    lineage = await registry.get_lineage(agent_uuid)
    parent_id = lineage.parent_id

    # Terminate in registry
    terminated_ids = await registry.terminate(agent_uuid, cascade=cascade)

    # Clean up resources for terminated agents
    for tid in terminated_ids:
        await remove_agent_resources(tid)

    # Update parent's spawn count if applicable
    if parent_id:
        parent_resources = await get_agent_resources(parent_id)
        if parent_resources:
            parent_resources.terminate_child(agent_uuid)

    return {
        "agent_id": agent_id,
        "cascade": cascade,
        "terminated": [str(t) for t in terminated_ids],
        "count": len(terminated_ids),
    }


@thunk_operation(
    name="agent.hierarchy.list_children",
    description="List all children of an agent.",
    required_capabilities=frozenset({"agent.read"}),
)
async def list_children(agent_id: str) -> dict[str, Any]:
    """List all children of an agent.

    Args:
        agent_id: ID of the parent agent.

    Returns:
        Dict with list of child info.
    """
    agent_uuid = UUID(agent_id)
    registry = await get_agent_registry()

    children = await registry.get_children(agent_uuid)

    return {
        "agent_id": agent_id,
        "children": [
            {
                "child_id": str(c.agent_id),
                "depth": c.depth,
                "created_at": c.created_at.isoformat(),
                "is_active": c.is_active,
            }
            for c in children
        ],
        "count": len(children),
    }


@thunk_operation(
    name="agent.hierarchy.get_lineage",
    description="Get full lineage tree for an agent.",
    required_capabilities=frozenset({"agent.read"}),
)
async def get_lineage(agent_id: str) -> dict[str, Any]:
    """Get full lineage information for an agent.

    Args:
        agent_id: ID of the agent.

    Returns:
        Dict with lineage info including ancestors and descendants.
    """
    agent_uuid = UUID(agent_id)
    registry = await get_agent_registry()

    lineage = await registry.get_lineage(agent_uuid)
    ancestors = await registry.get_ancestors(agent_uuid)
    descendants = await registry.get_descendants(agent_uuid)

    return {
        "agent_id": agent_id,
        "parent_id": str(lineage.parent_id) if lineage.parent_id else None,
        "depth": lineage.depth,
        "created_at": lineage.created_at.isoformat(),
        "is_active": lineage.is_active,
        "ancestors": [{"agent_id": str(a.agent_id), "depth": a.depth} for a in ancestors],
        "descendants": [{"agent_id": str(d.agent_id), "depth": d.depth} for d in descendants],
        "direct_children": [str(c) for c in lineage.children],
    }


@thunk_operation(
    name="agent.hierarchy.get_resources",
    description="Get capability budget and quota for an agent.",
    required_capabilities=frozenset({"agent.read"}),
)
async def get_resources(agent_id: str) -> dict[str, Any]:
    """Get resource information for an agent.

    Args:
        agent_id: ID of the agent.

    Returns:
        Dict with capability budget and quota info.
    """
    agent_uuid = UUID(agent_id)

    resources = await get_agent_resources(agent_uuid)
    if resources is None:
        return {
            "agent_id": agent_id,
            "error": "No resources found for agent",
        }

    return {
        "agent_id": agent_id,
        "capabilities": {
            "total": sorted(resources.capabilities.total),
            "available": sorted(resources.capabilities.available),
            "allocated_to_children": {
                str(k): sorted(v) for k, v in resources.capabilities.allocated.items()
            },
        },
        "quota": resources.quota.to_dict(),
    }


@thunk_operation(
    name="agent.hierarchy.register_root",
    description="Register a new root agent with initial capabilities.",
    required_capabilities=frozenset({"agent.admin"}),
)
async def register_root_agent(
    capabilities: list[str],
    metadata: dict[str, Any] | None = None,
    max_children: int = 10,
    max_depth: int = 5,
    max_tokens: int = 100000,
) -> dict[str, Any]:
    """Register a new root agent with initial capabilities.

    This is typically called to create the initial orchestrator agent.

    Args:
        capabilities: Initial capabilities for the root agent.
        metadata: Optional metadata for the agent.
        max_children: Maximum children the agent can spawn.
        max_depth: Maximum hierarchy depth.
        max_tokens: Maximum total tokens for descendants.

    Returns:
        Dict with agent ID and resource info.
    """
    registry = await get_agent_registry()

    # Register in hierarchy
    lineage = await registry.register(metadata=metadata or {})

    # Create resources
    resources = AgentResources(
        agent_id=lineage.agent_id,
        capabilities=CapabilityBudget(total=frozenset(capabilities)),
        quota=ResourceQuota(
            max_children=max_children,
            max_depth=max_depth,
            max_total_tokens=max_tokens,
        ),
    )
    await set_agent_resources(lineage.agent_id, resources)

    return {
        "agent_id": str(lineage.agent_id),
        "capabilities": sorted(capabilities),
        "quota": resources.quota.to_dict(),
        "is_root": True,
    }
