"""Plugin approval system.

Provides the user.approve operation and approval handling infrastructure.
Different UIs (CLI, web, API) can implement the ApprovalHandler protocol
to provide their own approval mechanisms.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from crows_nest.core.registry import thunk_operation
from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    ApprovalScope,
    PluginArtifact,
    RiskLevel,
    VerificationReport,
)
from crows_nest.plugins.storage import get_artifact_storage


@dataclass
class ApprovalRequest:
    """A request for user approval.

    Attributes:
        id: Unique request identifier.
        request_type: Type of request (e.g., "plugin.load").
        description: What's being requested.
        artifact: The plugin artifact (if applicable).
        verification: Verification report (if available).
        risk_level: Assessed risk level.
        context: Additional context.
        created_at: When the request was created.
        expires_at: When the request expires.
    """

    id: UUID
    request_type: str
    description: str
    artifact: PluginArtifact | None
    verification: VerificationReport | None
    risk_level: RiskLevel
    context: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for display."""
        return {
            "id": str(self.id),
            "request_type": self.request_type,
            "description": self.description,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "risk_level": self.risk_level.value,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@runtime_checkable
class ApprovalHandler(Protocol):
    """Protocol for approval handlers.

    Different UIs implement this to provide their approval mechanism.
    """

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Request approval from the user.

        Args:
            request: The approval request.

        Returns:
            The user's decision.

        Raises:
            asyncio.TimeoutError: If the request times out.
        """
        ...


class AutoApprovalHandler:
    """Approval handler that automatically approves/denies.

    Useful for testing and automated pipelines with predefined policies.
    """

    def __init__(
        self,
        auto_approve: bool = False,
        approve_low_risk: bool = False,
        decided_by: str = "auto",
    ) -> None:
        """Initialize auto-approval handler.

        Args:
            auto_approve: If True, approve everything.
            approve_low_risk: If True, only approve LOW risk items.
            decided_by: Who to attribute decisions to.
        """
        self.auto_approve = auto_approve
        self.approve_low_risk = approve_low_risk
        self.decided_by = decided_by

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Auto-approve or deny based on configuration."""
        should_approve = self.auto_approve or (
            self.approve_low_risk and request.risk_level == RiskLevel.LOW
        )

        if should_approve:
            reason = "Auto-approved"
            if self.approve_low_risk:
                reason = "Auto-approved (low risk)"
        else:
            reason = "Auto-denied by policy"

        return ApprovalDecision(
            id=uuid4(),
            artifact_id=request.artifact.id if request.artifact else UUID(int=0),
            artifact_hash=request.artifact.content_hash() if request.artifact else "",
            verification_id=request.verification.id if request.verification else None,
            approved=should_approve,
            reason=reason,
            conditions=(),
            decided_by=self.decided_by,
            decided_at=datetime.now(UTC),
            decision_context={
                "request_id": str(request.id),
                "auto_approved": True,
            },
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )


class QueuedApprovalHandler:
    """Approval handler that queues requests for later processing.

    Requests are stored in a queue and can be processed by an external
    system (CLI, web dashboard, etc.).
    """

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        """Initialize queued handler.

        Args:
            timeout_seconds: How long to wait for approval.
        """
        self.timeout_seconds = timeout_seconds
        self._pending: dict[UUID, ApprovalRequest] = {}
        self._responses: dict[UUID, ApprovalDecision] = {}
        self._events: dict[UUID, asyncio.Event] = {}

    @property
    def pending_requests(self) -> list[ApprovalRequest]:
        """Get list of pending approval requests."""
        return list(self._pending.values())

    def get_request(self, request_id: UUID) -> ApprovalRequest | None:
        """Get a specific pending request."""
        return self._pending.get(request_id)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Queue request and wait for response."""
        self._pending[request.id] = request
        self._events[request.id] = asyncio.Event()

        try:
            # Wait for response or timeout
            await asyncio.wait_for(
                self._events[request.id].wait(),
                timeout=self.timeout_seconds,
            )

            decision = self._responses.pop(request.id)
            return decision

        except TimeoutError:
            # Create a denial for timeout
            return ApprovalDecision(
                id=uuid4(),
                artifact_id=request.artifact.id if request.artifact else UUID(int=0),
                artifact_hash=request.artifact.content_hash() if request.artifact else "",
                verification_id=request.verification.id if request.verification else None,
                approved=False,
                reason="Request timed out",
                conditions=(),
                decided_by="system",
                decided_at=datetime.now(UTC),
                decision_context={"timeout": True},
                scope=ApprovalScope.ONCE,
                expires_at=None,
            )

        finally:
            self._pending.pop(request.id, None)
            self._events.pop(request.id, None)

    def respond(
        self,
        request_id: UUID,
        approved: bool,
        reason: str,
        decided_by: str,
        conditions: list[str] | None = None,
        scope: ApprovalScope = ApprovalScope.SESSION,
        expires_in_hours: float | None = None,
    ) -> bool:
        """Respond to a pending request.

        Args:
            request_id: ID of the request to respond to.
            approved: Whether to approve.
            reason: Reason for the decision.
            decided_by: Who made the decision.
            conditions: Any conditions on approval.
            scope: Scope of the approval.
            expires_in_hours: Hours until approval expires.

        Returns:
            True if request was found and responded to.
        """
        request = self._pending.get(request_id)
        if request is None:
            return False

        expires_at = None
        if expires_in_hours is not None:
            expires_at = datetime.now(UTC) + timedelta(hours=expires_in_hours)

        decision = ApprovalDecision(
            id=uuid4(),
            artifact_id=request.artifact.id if request.artifact else UUID(int=0),
            artifact_hash=request.artifact.content_hash() if request.artifact else "",
            verification_id=request.verification.id if request.verification else None,
            approved=approved,
            reason=reason,
            conditions=tuple(conditions or []),
            decided_by=decided_by,
            decided_at=datetime.now(UTC),
            decision_context={"request_id": str(request_id)},
            scope=scope,
            expires_at=expires_at,
        )

        self._responses[request_id] = decision
        self._events[request_id].set()
        return True


# Global approval handler
_approval_handler: ApprovalHandler | None = None


def get_approval_handler() -> ApprovalHandler:
    """Get the global approval handler.

    Returns:
        The approval handler.

    Raises:
        RuntimeError: If no handler is configured.
    """
    if _approval_handler is None:
        msg = "Approval handler not configured. Call configure_approval_handler first."
        raise RuntimeError(msg)
    return _approval_handler


def configure_approval_handler(handler: ApprovalHandler) -> None:
    """Configure the global approval handler.

    Args:
        handler: The handler to use.
    """
    global _approval_handler  # noqa: PLW0603
    _approval_handler = handler


def reset_approval_handler() -> None:
    """Reset the global approval handler (for testing)."""
    global _approval_handler  # noqa: PLW0603
    _approval_handler = None


def check_cached_approval(
    artifact: PluginArtifact,
    _verification: VerificationReport | None = None,
) -> ApprovalDecision | None:
    """Check for a valid cached approval.

    Args:
        artifact: The artifact to check.
        verification: Optional verification report.

    Returns:
        Valid cached approval if found, None otherwise.
    """
    storage = get_artifact_storage()
    approval = storage.load_approval_for_artifact(artifact.id)

    if approval is None:
        return None

    # Check if approval matches this artifact
    if not approval.matches_artifact(artifact):
        return None

    return approval


@thunk_operation(
    name="user.approve",
    description="Request human approval for an action.",
    required_capabilities=frozenset({"user.prompt"}),
)
async def user_approve(
    request_type: str,
    description: str,
    artifact_id: str | None = None,
    artifact_dict: dict[str, Any] | None = None,
    verification_id: str | None = None,
    risk_level: str = "medium",
    context: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    check_cache: bool = True,
) -> dict[str, Any]:
    """Request human approval for an action.

    Presents the request to the user and waits for a response.
    Can use cached approvals for previously approved artifacts.

    Args:
        request_type: Type of request (e.g., "plugin.load").
        description: What's being requested.
        artifact_id: UUID of artifact in storage (optional).
        artifact_dict: Artifact as dict (alternative to artifact_id).
        verification_id: UUID of verification report (optional).
        risk_level: Risk level (low, medium, high, critical).
        context: Additional context for the user.
        timeout_seconds: How long to wait (None = use handler default).
        check_cache: Whether to check for cached approvals.

    Returns:
        Dict with approval decision and metadata.
    """
    # Load artifact
    artifact: PluginArtifact | None = None
    if artifact_id:
        storage = get_artifact_storage()
        artifact = storage.load_artifact(UUID(artifact_id))
        if artifact is None:
            return {
                "success": False,
                "approved": False,
                "error": f"Artifact not found: {artifact_id}",
                "error_type": "not_found",
            }
    elif artifact_dict:
        try:
            artifact = PluginArtifact.from_dict(artifact_dict)
        except (KeyError, ValueError, TypeError) as e:
            return {
                "success": False,
                "approved": False,
                "error": f"Invalid artifact dict: {e}",
                "error_type": "invalid_input",
            }

    # Load verification report
    verification: VerificationReport | None = None
    if verification_id:
        storage = get_artifact_storage()
        verification = storage.load_verification(UUID(verification_id))

    # Check for cached approval
    if check_cache and artifact:
        cached = check_cached_approval(artifact, verification)
        if cached is not None:
            return {
                "success": True,
                "approved": cached.approved,
                "decision": cached.to_dict(),
                "decision_id": str(cached.id),
                "cached": True,
            }

    # Get the approval handler
    try:
        handler = get_approval_handler()
    except RuntimeError as e:
        return {
            "success": False,
            "approved": False,
            "error": str(e),
            "error_type": "configuration_error",
        }

    # Create the approval request
    request = ApprovalRequest(
        id=uuid4(),
        request_type=request_type,
        description=description,
        artifact=artifact,
        verification=verification,
        risk_level=RiskLevel(risk_level),
        context=context or {},
        expires_at=(
            datetime.now(UTC) + timedelta(seconds=timeout_seconds)
            if timeout_seconds
            else None
        ),
    )

    try:
        decision = await handler.request_approval(request)

        # Save the decision
        storage = get_artifact_storage()
        storage.save_approval(decision)

        return {
            "success": True,
            "approved": decision.approved,
            "decision": decision.to_dict(),
            "decision_id": str(decision.id),
            "cached": False,
        }

    except TimeoutError:
        return {
            "success": False,
            "approved": False,
            "error": "Approval request timed out",
            "error_type": "timeout",
        }

    except Exception as e:
        return {
            "success": False,
            "approved": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
