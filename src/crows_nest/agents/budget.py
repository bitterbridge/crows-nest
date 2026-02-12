"""Capability budget and resource quota management.

Controls capability inheritance and resource limits for agent spawning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


class BudgetError(Exception):
    """Base exception for budget-related errors."""


class InsufficientCapabilitiesError(BudgetError):
    """Requested capabilities exceed available budget."""

    def __init__(
        self,
        requested: frozenset[str],
        available: frozenset[str],
    ) -> None:
        self.requested = requested
        self.available = available
        missing = requested - available
        super().__init__(
            f"Insufficient capabilities: {sorted(missing)} not available. "
            f"Available: {sorted(available)}"
        )


class CapabilityNotAllowedError(BudgetError):
    """Capability is not in the parent's total set."""

    def __init__(self, capability: str, parent_capabilities: frozenset[str]) -> None:
        self.capability = capability
        self.parent_capabilities = parent_capabilities
        super().__init__(
            f"Capability '{capability}' not allowed. Parent has: {sorted(parent_capabilities)}"
        )


class QuotaExceededError(BudgetError):
    """Resource quota exceeded."""

    def __init__(self, resource: str, limit: int, current: int) -> None:
        self.resource = resource
        self.limit = limit
        self.current = current
        super().__init__(f"Quota exceeded for {resource}: limit={limit}, current={current}")


@dataclass
class CapabilityBudget:
    """Tracks capability allocation from parent to children.

    A parent agent has a set of total capabilities. When spawning children,
    it can allocate subsets of these capabilities. Children cannot have
    capabilities that the parent doesn't have.

    Attributes:
        total: All capabilities available to this agent.
        allocated: Capabilities allocated to each child.
        reserved: Capabilities reserved (not allocatable to children).
    """

    total: frozenset[str]
    allocated: dict[UUID, frozenset[str]] = field(default_factory=dict)
    reserved: frozenset[str] = field(default_factory=frozenset)

    @property
    def available(self) -> frozenset[str]:
        """Get capabilities available for allocation.

        Note: This returns total - reserved, not accounting for allocated.
        Multiple children can share the same capabilities.
        """
        return self.total - self.reserved

    @property
    def all_allocated(self) -> frozenset[str]:
        """Get union of all allocated capabilities."""
        result: set[str] = set()
        for caps in self.allocated.values():
            result.update(caps)
        return frozenset(result)

    def can_allocate(self, capabilities: frozenset[str]) -> bool:
        """Check if capabilities can be allocated.

        Args:
            capabilities: Capabilities to check.

        Returns:
            True if all capabilities are available for allocation.
        """
        return capabilities <= self.available

    def allocate(
        self,
        child_id: UUID,
        capabilities: frozenset[str],
    ) -> frozenset[str]:
        """Allocate capabilities to a child agent.

        Args:
            child_id: ID of the child agent.
            capabilities: Capabilities to allocate.

        Returns:
            The allocated capabilities.

        Raises:
            InsufficientCapabilitiesError: If capabilities exceed available.
        """
        if not self.can_allocate(capabilities):
            raise InsufficientCapabilitiesError(capabilities, self.available)

        self.allocated[child_id] = capabilities
        return capabilities

    def deallocate(self, child_id: UUID) -> frozenset[str] | None:
        """Deallocate capabilities from a child agent.

        Args:
            child_id: ID of the child agent.

        Returns:
            The deallocated capabilities, or None if child wasn't allocated.
        """
        return self.allocated.pop(child_id, None)

    def get_child_capabilities(self, child_id: UUID) -> frozenset[str]:
        """Get capabilities allocated to a specific child."""
        return self.allocated.get(child_id, frozenset())

    def reserve(self, capabilities: frozenset[str]) -> None:
        """Reserve capabilities (prevent allocation to children).

        Args:
            capabilities: Capabilities to reserve.

        Raises:
            CapabilityNotAllowedError: If any capability isn't in total.
        """
        for cap in capabilities:
            if cap not in self.total:
                raise CapabilityNotAllowedError(cap, self.total)
        self.reserved = self.reserved | capabilities

    def unreserve(self, capabilities: frozenset[str]) -> None:
        """Unreserve capabilities (allow allocation to children)."""
        self.reserved = self.reserved - capabilities

    def create_child_budget(
        self,
        child_id: UUID,
        capabilities: frozenset[str],
    ) -> CapabilityBudget:
        """Create a budget for a child with allocated capabilities.

        The child's budget will have the allocated capabilities as its total,
        allowing further subdivision to grandchildren.

        Args:
            child_id: ID of the child agent.
            capabilities: Capabilities to give the child.

        Returns:
            A new CapabilityBudget for the child.

        Raises:
            InsufficientCapabilitiesError: If capabilities exceed available.
        """
        allocated = self.allocate(child_id, capabilities)
        return CapabilityBudget(total=allocated)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": sorted(self.total),
            "allocated": {str(k): sorted(v) for k, v in self.allocated.items()},
            "reserved": sorted(self.reserved),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityBudget:
        """Create from dictionary."""
        return cls(
            total=frozenset(data.get("total", [])),
            allocated={UUID(k): frozenset(v) for k, v in data.get("allocated", {}).items()},
            reserved=frozenset(data.get("reserved", [])),
        )


@dataclass
class ResourceQuota:
    """Limits on agent spawning and resource usage.

    Enforces constraints on how many children an agent can spawn,
    hierarchy depth, and token budgets.

    Attributes:
        max_children: Maximum number of direct children.
        max_depth: Maximum hierarchy depth from this agent.
        max_tokens_per_child: Token limit per child agent.
        max_total_tokens: Total token limit for all descendants.
        children_spawned: Current count of spawned children.
        total_tokens_used: Current total token usage.
    """

    max_children: int = 10
    max_depth: int = 5
    max_tokens_per_child: int = 10000
    max_total_tokens: int = 100000
    children_spawned: int = 0
    total_tokens_used: int = 0

    def can_spawn(self) -> bool:
        """Check if another child can be spawned."""
        return self.children_spawned < self.max_children

    def check_spawn(self) -> None:
        """Check if spawning is allowed, raise if not.

        Raises:
            QuotaExceededError: If max children reached.
        """
        if not self.can_spawn():
            raise QuotaExceededError("children", self.max_children, self.children_spawned)

    def record_spawn(self) -> None:
        """Record that a child was spawned."""
        self.children_spawned += 1

    def record_termination(self) -> None:
        """Record that a child was terminated."""
        if self.children_spawned > 0:
            self.children_spawned -= 1

    def can_use_tokens(self, tokens: int) -> bool:
        """Check if token usage is within limits."""
        return (self.total_tokens_used + tokens) <= self.max_total_tokens

    def check_tokens(self, tokens: int) -> None:
        """Check if token usage is allowed, raise if not.

        Args:
            tokens: Number of tokens to use.

        Raises:
            QuotaExceededError: If token limit would be exceeded.
        """
        if not self.can_use_tokens(tokens):
            raise QuotaExceededError(
                "tokens",
                self.max_total_tokens,
                self.total_tokens_used + tokens,
            )

    def use_tokens(self, tokens: int) -> None:
        """Record token usage.

        Args:
            tokens: Number of tokens used.

        Raises:
            QuotaExceededError: If token limit would be exceeded.
        """
        self.check_tokens(tokens)
        self.total_tokens_used += tokens

    def create_child_quota(
        self,
        max_children: int | None = None,
        max_depth: int | None = None,
        max_tokens: int | None = None,
    ) -> ResourceQuota:
        """Create a quota for a child agent.

        Child quotas are constrained by parent quotas.

        Args:
            max_children: Max children for the child (default: same as parent).
            max_depth: Max depth for the child (default: parent - 1).
            max_tokens: Max total tokens for the child (default: max_tokens_per_child).

        Returns:
            A new ResourceQuota for the child.
        """
        return ResourceQuota(
            max_children=min(
                max_children or self.max_children,
                self.max_children,
            ),
            max_depth=min(
                max_depth or self.max_depth - 1,
                self.max_depth - 1,
            ),
            max_tokens_per_child=min(
                max_tokens or self.max_tokens_per_child,
                self.max_tokens_per_child,
            ),
            max_total_tokens=min(
                max_tokens or self.max_tokens_per_child,
                self.max_total_tokens - self.total_tokens_used,
            ),
        )

    @property
    def remaining_children(self) -> int:
        """Get remaining spawn capacity."""
        return max(0, self.max_children - self.children_spawned)

    @property
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_total_tokens - self.total_tokens_used)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_children": self.max_children,
            "max_depth": self.max_depth,
            "max_tokens_per_child": self.max_tokens_per_child,
            "max_total_tokens": self.max_total_tokens,
            "children_spawned": self.children_spawned,
            "total_tokens_used": self.total_tokens_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceQuota:
        """Create from dictionary."""
        return cls(
            max_children=data.get("max_children", 10),
            max_depth=data.get("max_depth", 5),
            max_tokens_per_child=data.get("max_tokens_per_child", 10000),
            max_total_tokens=data.get("max_total_tokens", 100000),
            children_spawned=data.get("children_spawned", 0),
            total_tokens_used=data.get("total_tokens_used", 0),
        )


@dataclass
class AgentResources:
    """Combined capability budget and resource quota for an agent.

    This is the complete resource state for an agent, combining
    capabilities and quotas.

    Attributes:
        agent_id: ID of the agent these resources belong to.
        capabilities: Capability budget for the agent.
        quota: Resource quota for the agent.
    """

    agent_id: UUID
    capabilities: CapabilityBudget
    quota: ResourceQuota = field(default_factory=ResourceQuota)

    def can_spawn_with(
        self,
        capabilities: frozenset[str],
        tokens: int = 0,
    ) -> bool:
        """Check if a child with given capabilities can be spawned.

        Args:
            capabilities: Capabilities for the child.
            tokens: Estimated token usage for the child.

        Returns:
            True if spawn is allowed.
        """
        return (
            self.quota.can_spawn()
            and self.capabilities.can_allocate(capabilities)
            and self.quota.can_use_tokens(tokens)
        )

    def spawn_child(
        self,
        child_id: UUID,
        capabilities: frozenset[str],
        child_quota: ResourceQuota | None = None,
    ) -> AgentResources:
        """Create resources for a child agent.

        Args:
            child_id: ID for the child agent.
            capabilities: Capabilities to allocate to the child.
            child_quota: Optional custom quota for the child.

        Returns:
            AgentResources for the child.

        Raises:
            QuotaExceededError: If spawn quota exceeded.
            InsufficientCapabilitiesError: If capabilities not available.
        """
        self.quota.check_spawn()

        child_cap_budget = self.capabilities.create_child_budget(child_id, capabilities)
        child_res_quota = child_quota or self.quota.create_child_quota()

        self.quota.record_spawn()

        return AgentResources(
            agent_id=child_id,
            capabilities=child_cap_budget,
            quota=child_res_quota,
        )

    def terminate_child(self, child_id: UUID) -> None:
        """Handle child termination.

        Args:
            child_id: ID of the terminated child.
        """
        self.capabilities.deallocate(child_id)
        self.quota.record_termination()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "capabilities": self.capabilities.to_dict(),
            "quota": self.quota.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResources:
        """Create from dictionary."""
        return cls(
            agent_id=UUID(data["agent_id"]),
            capabilities=CapabilityBudget.from_dict(data["capabilities"]),
            quota=ResourceQuota.from_dict(data["quota"]),
        )
