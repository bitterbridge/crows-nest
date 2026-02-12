"""Tests for plugin approval system."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.plugins.approval import (
    ApprovalRequest,
    AutoApprovalHandler,
    QueuedApprovalHandler,
    check_cached_approval,
    configure_approval_handler,
    get_approval_handler,
    reset_approval_handler,
    user_approve,
)
from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    ApprovalScope,
    PluginArtifact,
    RiskLevel,
)
from crows_nest.plugins.storage import get_artifact_storage, reset_artifact_storage

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_handlers() -> Generator[None, None, None]:
    """Reset approval handler and storage before and after each test."""
    reset_approval_handler()
    reset_artifact_storage()
    yield
    reset_approval_handler()
    reset_artifact_storage()


def make_artifact(name: str = "test_plugin") -> PluginArtifact:
    """Helper to create a plugin artifact."""
    return PluginArtifact(
        id=uuid4(),
        name=name,
        description="Test plugin",
        source_code="x = 1",
        operations=(),
        requested_capabilities=frozenset(),
        generated_by=uuid4(),
        generated_at=datetime.now(tz=UTC),
        generation_context={},
    )


def make_request(
    artifact: PluginArtifact | None = None,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> ApprovalRequest:
    """Helper to create an approval request."""
    return ApprovalRequest(
        id=uuid4(),
        request_type="plugin.load",
        description="Load test plugin",
        artifact=artifact,
        verification=None,
        risk_level=risk_level,
        context={},
    )


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_create_request(self) -> None:
        """Can create an approval request."""
        artifact = make_artifact()
        request = make_request(artifact)

        assert request.request_type == "plugin.load"
        assert request.artifact == artifact
        assert request.risk_level == RiskLevel.MEDIUM

    def test_request_to_dict(self) -> None:
        """Request serializes to dict."""
        artifact = make_artifact()
        request = make_request(artifact)

        data = request.to_dict()

        assert data["id"] == str(request.id)
        assert data["request_type"] == "plugin.load"
        assert data["risk_level"] == "medium"
        assert data["artifact"] is not None
        assert data["artifact"]["name"] == "test_plugin"

    def test_request_without_artifact(self) -> None:
        """Request can be created without artifact."""
        request = ApprovalRequest(
            id=uuid4(),
            request_type="custom.action",
            description="Do something",
            artifact=None,
            verification=None,
            risk_level=RiskLevel.LOW,
            context={"key": "value"},
        )

        data = request.to_dict()
        assert data["artifact"] is None
        assert data["context"] == {"key": "value"}


class TestAutoApprovalHandler:
    """Tests for AutoApprovalHandler."""

    @pytest.mark.asyncio
    async def test_auto_deny_by_default(self) -> None:
        """Handler denies by default."""
        handler = AutoApprovalHandler()
        request = make_request()

        decision = await handler.request_approval(request)

        assert decision.approved is False
        assert "denied" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_auto_approve_all(self) -> None:
        """Handler can auto-approve everything."""
        handler = AutoApprovalHandler(auto_approve=True)
        request = make_request()

        decision = await handler.request_approval(request)

        assert decision.approved is True
        assert "auto-approved" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_approve_low_risk_only(self) -> None:
        """Handler can approve only low risk items."""
        handler = AutoApprovalHandler(approve_low_risk=True)

        # Low risk should be approved
        low_request = make_request(risk_level=RiskLevel.LOW)
        low_decision = await handler.request_approval(low_request)
        assert low_decision.approved is True

        # Medium risk should be denied
        medium_request = make_request(risk_level=RiskLevel.MEDIUM)
        medium_decision = await handler.request_approval(medium_request)
        assert medium_decision.approved is False

    @pytest.mark.asyncio
    async def test_custom_decided_by(self) -> None:
        """Handler uses custom decided_by field."""
        handler = AutoApprovalHandler(auto_approve=True, decided_by="test_system")
        request = make_request()

        decision = await handler.request_approval(request)

        assert decision.decided_by == "test_system"

    @pytest.mark.asyncio
    async def test_decision_has_artifact_info(self) -> None:
        """Decision includes artifact information."""
        handler = AutoApprovalHandler(auto_approve=True)
        artifact = make_artifact()
        request = make_request(artifact)

        decision = await handler.request_approval(request)

        assert decision.artifact_id == artifact.id
        assert decision.artifact_hash == artifact.content_hash()


class TestQueuedApprovalHandler:
    """Tests for QueuedApprovalHandler."""

    def test_queue_starts_empty(self) -> None:
        """New handler has no pending requests."""
        handler = QueuedApprovalHandler()
        assert len(handler.pending_requests) == 0

    @pytest.mark.asyncio
    async def test_request_queued(self) -> None:
        """Request is added to queue."""
        handler = QueuedApprovalHandler(timeout_seconds=0.1)
        request = make_request()

        # Start the request (will timeout)
        task = asyncio.create_task(handler.request_approval(request))

        # Check it's in the queue
        await asyncio.sleep(0.01)
        assert len(handler.pending_requests) == 1
        assert handler.get_request(request.id) == request

        # Wait for timeout
        await task

    @pytest.mark.asyncio
    async def test_timeout_denies(self) -> None:
        """Request that times out is denied."""
        handler = QueuedApprovalHandler(timeout_seconds=0.05)
        request = make_request()

        decision = await handler.request_approval(request)

        assert decision.approved is False
        assert "timed out" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_respond_approves(self) -> None:
        """Can respond to approve a request."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        # Start request
        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        # Respond
        success = handler.respond(
            request_id=request.id,
            approved=True,
            reason="Looks good",
            decided_by="tester",
        )
        assert success is True

        # Get result
        decision = await task
        assert decision.approved is True
        assert decision.reason == "Looks good"
        assert decision.decided_by == "tester"

    @pytest.mark.asyncio
    async def test_respond_denies(self) -> None:
        """Can respond to deny a request."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        handler.respond(
            request_id=request.id,
            approved=False,
            reason="Too risky",
            decided_by="tester",
        )

        decision = await task
        assert decision.approved is False
        assert decision.reason == "Too risky"

    @pytest.mark.asyncio
    async def test_respond_with_conditions(self) -> None:
        """Can add conditions to approval."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        handler.respond(
            request_id=request.id,
            approved=True,
            reason="Approved with limits",
            decided_by="tester",
            conditions=["no network", "read only"],
        )

        decision = await task
        assert decision.conditions == ("no network", "read only")

    @pytest.mark.asyncio
    async def test_respond_with_scope(self) -> None:
        """Can set approval scope."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        handler.respond(
            request_id=request.id,
            approved=True,
            reason="Permanent approval",
            decided_by="tester",
            scope=ApprovalScope.PERMANENT,
        )

        decision = await task
        assert decision.scope == ApprovalScope.PERMANENT

    @pytest.mark.asyncio
    async def test_respond_with_expiry(self) -> None:
        """Can set approval expiry."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        handler.respond(
            request_id=request.id,
            approved=True,
            reason="Temporary approval",
            decided_by="tester",
            expires_in_hours=24.0,
        )

        decision = await task
        assert decision.expires_at is not None

    def test_respond_to_unknown_request(self) -> None:
        """Responding to unknown request returns False."""
        handler = QueuedApprovalHandler()
        success = handler.respond(
            request_id=uuid4(),
            approved=True,
            reason="Attempt",
            decided_by="tester",
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_request_removed_after_response(self) -> None:
        """Request is removed from queue after response."""
        handler = QueuedApprovalHandler(timeout_seconds=1.0)
        request = make_request()

        task = asyncio.create_task(handler.request_approval(request))
        await asyncio.sleep(0.01)

        handler.respond(
            request_id=request.id,
            approved=True,
            reason="Done",
            decided_by="tester",
        )

        await task

        assert len(handler.pending_requests) == 0
        assert handler.get_request(request.id) is None


class TestGlobalHandler:
    """Tests for global handler management."""

    def test_no_handler_raises(self) -> None:
        """Getting handler without configuration raises."""
        with pytest.raises(RuntimeError, match="not configured"):
            get_approval_handler()

    def test_configure_handler(self) -> None:
        """Can configure global handler."""
        handler = AutoApprovalHandler()
        configure_approval_handler(handler)

        assert get_approval_handler() is handler

    def test_reset_handler(self) -> None:
        """Can reset global handler."""
        configure_approval_handler(AutoApprovalHandler())
        reset_approval_handler()

        with pytest.raises(RuntimeError):
            get_approval_handler()


class TestCachedApproval:
    """Tests for approval caching."""

    def test_no_cached_approval(self) -> None:
        """Returns None when no approval cached."""
        artifact = make_artifact()
        result = check_cached_approval(artifact)
        assert result is None

    def test_cached_approval_found(self) -> None:
        """Returns cached approval when available."""
        artifact = make_artifact()

        # Save an approval
        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        approval = ApprovalDecision(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            verification_id=None,
            approved=True,
            reason="Previously approved",
            conditions=(),
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        storage.save_approval(approval)

        result = check_cached_approval(artifact)
        assert result is not None
        assert result.approved is True

    def test_cached_approval_hash_mismatch(self) -> None:
        """Returns None when artifact hash doesn't match."""
        artifact = make_artifact()
        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        # Save approval with different hash
        approval = ApprovalDecision(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash="different_hash",  # Wrong hash
            verification_id=None,
            approved=True,
            reason="Old approval",
            conditions=(),
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.HASH_PERMANENT,  # Hash-based
            expires_at=None,
        )
        storage.save_approval(approval)

        result = check_cached_approval(artifact)
        assert result is None


class TestUserApproveOperation:
    """Tests for user.approve thunk operation."""

    @pytest.mark.asyncio
    async def test_no_handler_error(self) -> None:
        """Returns error when no handler configured."""
        result = await user_approve(
            request_type="test",
            description="Test request",
        )

        assert result["success"] is False
        assert result["error_type"] == "configuration_error"

    @pytest.mark.asyncio
    async def test_artifact_not_found(self) -> None:
        """Returns error for missing artifact."""
        configure_approval_handler(AutoApprovalHandler())

        result = await user_approve(
            request_type="test",
            description="Test",
            artifact_id=str(uuid4()),
        )

        assert result["success"] is False
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_invalid_artifact_dict(self) -> None:
        """Returns error for invalid artifact dict."""
        configure_approval_handler(AutoApprovalHandler())

        result = await user_approve(
            request_type="test",
            description="Test",
            artifact_dict={"invalid": "data"},
        )

        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_approve_with_auto_handler(self) -> None:
        """Auto-approves when handler allows."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=True))

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
        )

        assert result["success"] is True
        assert result["approved"] is True
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_deny_with_auto_handler(self) -> None:
        """Auto-denies when handler denies."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=False))

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
        )

        assert result["success"] is True
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_approve_with_artifact_id(self) -> None:
        """Can approve using artifact ID from storage."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=True))

        artifact = make_artifact()
        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
            artifact_id=str(artifact.id),
        )

        assert result["success"] is True
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_approve_with_artifact_dict(self) -> None:
        """Can approve using artifact dict."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=True))

        artifact = make_artifact()

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
            artifact_dict=artifact.to_dict(),
        )

        assert result["success"] is True
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_uses_cached_approval(self) -> None:
        """Uses cached approval when available."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=False))

        artifact = make_artifact()
        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        # Create cached approval
        approval = ApprovalDecision(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            verification_id=None,
            approved=True,
            reason="Cached",
            conditions=(),
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        storage.save_approval(approval)

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
            artifact_id=str(artifact.id),
            check_cache=True,
        )

        assert result["success"] is True
        assert result["approved"] is True
        assert result["cached"] is True

    @pytest.mark.asyncio
    async def test_skip_cache(self) -> None:
        """Can skip cache check."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=False))

        artifact = make_artifact()
        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        # Create cached approval
        approval = ApprovalDecision(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            verification_id=None,
            approved=True,
            reason="Cached",
            conditions=(),
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        storage.save_approval(approval)

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
            artifact_id=str(artifact.id),
            check_cache=False,  # Skip cache
        )

        # Should use auto-handler which denies
        assert result["approved"] is False
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_decision_saved(self) -> None:
        """Decision is saved to storage."""
        configure_approval_handler(AutoApprovalHandler(auto_approve=True))

        result = await user_approve(
            request_type="plugin.load",
            description="Load plugin",
        )

        # Decision ID should be available
        decision_id = result["decision_id"]
        assert decision_id is not None

    @pytest.mark.asyncio
    async def test_risk_level_passed(self) -> None:
        """Risk level is passed to handler."""
        configure_approval_handler(AutoApprovalHandler(approve_low_risk=True))

        # Low risk should approve
        low_result = await user_approve(
            request_type="test",
            description="Low risk",
            risk_level="low",
        )
        assert low_result["approved"] is True

        # High risk should deny
        high_result = await user_approve(
            request_type="test",
            description="High risk",
            risk_level="high",
        )
        assert high_result["approved"] is False
