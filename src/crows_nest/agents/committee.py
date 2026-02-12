"""Hiring committee pattern for agent spawning decisions.

Implements multi-agent consensus for spawn decisions, where multiple
evaluator agents vote on whether to approve a spawn request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class SpawnVote(StrEnum):
    """Vote outcome for a spawn request."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class SpawnDecisionStatus(StrEnum):
    """Overall decision status."""

    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class SpawnRequest:
    """A request to spawn a child agent.

    Attributes:
        id: Unique request identifier.
        parent_id: ID of the requesting parent agent.
        requested_capabilities: Capabilities requested for the child.
        justification: Reason for spawning the child.
        metadata: Additional request metadata.
        created_at: When the request was created.
    """

    parent_id: UUID
    requested_capabilities: frozenset[str]
    justification: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "parent_id": str(self.parent_id),
            "requested_capabilities": sorted(self.requested_capabilities),
            "justification": self.justification,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EvaluatorVote:
    """A single evaluator's vote on a spawn request.

    Attributes:
        evaluator_id: ID of the evaluating agent.
        vote: The vote outcome.
        reasoning: Explanation for the vote.
        timestamp: When the vote was cast.
    """

    evaluator_id: UUID
    vote: SpawnVote
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluator_id": str(self.evaluator_id),
            "vote": self.vote.value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SpawnDecision:
    """The outcome of a hiring committee evaluation.

    Attributes:
        request: The original spawn request.
        status: Overall decision status.
        votes: Individual evaluator votes.
        approval_rate: Percentage of approvals.
        decided_at: When the decision was finalized.
        granted_capabilities: Capabilities approved (may differ from requested).
    """

    request: SpawnRequest
    status: SpawnDecisionStatus = SpawnDecisionStatus.PENDING
    votes: list[EvaluatorVote] = field(default_factory=list)
    decided_at: datetime | None = None
    granted_capabilities: frozenset[str] | None = None

    @property
    def approval_rate(self) -> float:
        """Calculate approval rate (excluding abstentions)."""
        voting_votes = [v for v in self.votes if v.vote != SpawnVote.ABSTAIN]
        if not voting_votes:
            return 0.0
        approvals = sum(1 for v in voting_votes if v.vote == SpawnVote.APPROVE)
        return approvals / len(voting_votes)

    @property
    def is_approved(self) -> bool:
        """Check if the request was approved."""
        return self.status == SpawnDecisionStatus.APPROVED

    def add_vote(self, vote: EvaluatorVote) -> None:
        """Add an evaluator's vote."""
        self.votes.append(vote)

    def finalize(self, threshold: float) -> None:
        """Finalize the decision based on votes and threshold.

        Args:
            threshold: Required approval rate (0-1).
        """
        if self.approval_rate >= threshold:
            self.status = SpawnDecisionStatus.APPROVED
            self.granted_capabilities = self.request.requested_capabilities
        else:
            self.status = SpawnDecisionStatus.REJECTED
            self.granted_capabilities = None
        self.decided_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request": self.request.to_dict(),
            "status": self.status.value,
            "votes": [v.to_dict() for v in self.votes],
            "approval_rate": self.approval_rate,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "granted_capabilities": (
                sorted(self.granted_capabilities) if self.granted_capabilities else None
            ),
        }


@dataclass
class EvaluatorSpec:
    """Specification for an evaluator agent.

    Attributes:
        evaluator_id: ID of the evaluator agent.
        name: Human-readable name.
        criteria: What this evaluator checks for.
        weight: Vote weight (for weighted voting).
    """

    evaluator_id: UUID
    name: str = "Evaluator"
    criteria: str = "General evaluation"
    weight: float = 1.0


class HiringCommittee:
    """Multi-agent committee for evaluating spawn requests.

    The hiring committee pattern allows multiple agents to vote on
    whether a spawn request should be approved. This provides:
    - Checks and balances on agent proliferation
    - Different perspectives (security, resources, necessity)
    - Audit trail for spawn decisions

    Usage:
        committee = HiringCommittee(
            evaluators=[security_eval, resource_eval, necessity_eval],
            consensus_threshold=0.6,
        )

        request = SpawnRequest(
            parent_id=parent.agent_id,
            requested_capabilities=frozenset(["web.search"]),
            justification="Need to search for information",
        )

        decision = await committee.evaluate(request)
        if decision.is_approved:
            # Proceed with spawn
            ...
    """

    def __init__(
        self,
        evaluators: list[EvaluatorSpec] | None = None,
        consensus_threshold: float = 0.6,
        require_quorum: bool = True,
        quorum_fraction: float = 0.5,
    ) -> None:
        """Initialize the hiring committee.

        Args:
            evaluators: List of evaluator specifications.
            consensus_threshold: Required approval rate (0-1).
            require_quorum: Whether a minimum number of votes is required.
            quorum_fraction: Fraction of evaluators needed for quorum.
        """
        self.evaluators = evaluators or []
        self.consensus_threshold = consensus_threshold
        self.require_quorum = require_quorum
        self.quorum_fraction = quorum_fraction

    def add_evaluator(self, evaluator: EvaluatorSpec) -> None:
        """Add an evaluator to the committee."""
        self.evaluators.append(evaluator)

    def remove_evaluator(self, evaluator_id: UUID) -> bool:
        """Remove an evaluator from the committee."""
        for i, e in enumerate(self.evaluators):
            if e.evaluator_id == evaluator_id:
                self.evaluators.pop(i)
                return True
        return False

    @property
    def quorum_size(self) -> int:
        """Minimum number of votes needed for quorum."""
        return max(1, int(len(self.evaluators) * self.quorum_fraction))

    def has_quorum(self, votes: list[EvaluatorVote]) -> bool:
        """Check if enough votes have been cast for quorum."""
        non_abstain = sum(1 for v in votes if v.vote != SpawnVote.ABSTAIN)
        return non_abstain >= self.quorum_size

    async def evaluate(
        self,
        request: SpawnRequest,
        votes: list[EvaluatorVote] | None = None,
    ) -> SpawnDecision:
        """Evaluate a spawn request.

        In a real implementation, this would query each evaluator agent
        for their vote. For now, it accepts pre-collected votes.

        Args:
            request: The spawn request to evaluate.
            votes: Pre-collected votes (for testing/sync scenarios).

        Returns:
            The spawn decision.
        """
        decision = SpawnDecision(request=request)

        if votes:
            for vote in votes:
                decision.add_vote(vote)

        # Check quorum
        if self.require_quorum and not self.has_quorum(decision.votes):
            decision.status = SpawnDecisionStatus.PENDING
            return decision

        # Finalize decision
        decision.finalize(self.consensus_threshold)
        return decision

    async def evaluate_with_auto_approve(
        self,
        request: SpawnRequest,
    ) -> SpawnDecision:
        """Evaluate with automatic approval if no evaluators configured.

        Useful for development/testing when committee isn't set up.

        Args:
            request: The spawn request to evaluate.

        Returns:
            The spawn decision (auto-approved if no evaluators).
        """
        if not self.evaluators:
            # Auto-approve if no committee configured
            decision = SpawnDecision(request=request)
            decision.status = SpawnDecisionStatus.APPROVED
            decision.granted_capabilities = request.requested_capabilities
            decision.decided_at = datetime.now(UTC)
            return decision

        return await self.evaluate(request)

    def to_dict(self) -> dict[str, Any]:
        """Convert committee configuration to dictionary."""
        return {
            "evaluators": [
                {
                    "evaluator_id": str(e.evaluator_id),
                    "name": e.name,
                    "criteria": e.criteria,
                    "weight": e.weight,
                }
                for e in self.evaluators
            ],
            "consensus_threshold": self.consensus_threshold,
            "require_quorum": self.require_quorum,
            "quorum_fraction": self.quorum_fraction,
            "quorum_size": self.quorum_size,
        }


# Global committee instance (can be configured at startup)
_global_committee: HiringCommittee | None = None


def get_hiring_committee() -> HiringCommittee:
    """Get or create the global hiring committee."""
    global _global_committee  # noqa: PLW0603
    if _global_committee is None:
        _global_committee = HiringCommittee()
    return _global_committee


def set_hiring_committee(committee: HiringCommittee) -> None:
    """Set the global hiring committee."""
    global _global_committee  # noqa: PLW0603
    _global_committee = committee


def reset_hiring_committee() -> None:
    """Reset the global hiring committee."""
    global _global_committee  # noqa: PLW0603
    _global_committee = None
