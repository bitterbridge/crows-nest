"""Tests for plugin loading."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.core.registry import get_global_registry
from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    ApprovalScope,
    LoadScope,
    PluginArtifact,
)
from crows_nest.plugins.loading import (
    load_artifact,
    plugin_load,
    unload_plugin,
)
from crows_nest.plugins.storage import get_artifact_storage, reset_artifact_storage

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_state() -> Generator[None, None, None]:
    """Reset state before and after each test."""
    reset_artifact_storage()
    registry = get_global_registry()
    registry.clear()
    # Clean up any dynamic modules
    modules_to_remove = [k for k in sys.modules if k.startswith("crows_nest_dynamic_")]
    for mod in modules_to_remove:
        del sys.modules[mod]
    yield
    reset_artifact_storage()
    registry.clear()
    modules_to_remove = [k for k in sys.modules if k.startswith("crows_nest_dynamic_")]
    for mod in modules_to_remove:
        del sys.modules[mod]


def make_artifact(
    source_code: str,
    name: str = "test_plugin",
    capabilities: frozenset[str] | None = None,
) -> PluginArtifact:
    """Helper to create a plugin artifact."""
    return PluginArtifact(
        id=uuid4(),
        name=name,
        description="Test plugin",
        source_code=source_code,
        operations=(),
        requested_capabilities=capabilities or frozenset(),
        generated_by=uuid4(),
        generated_at=datetime.now(tz=UTC),
        generation_context={},
    )


def make_approval(
    artifact: PluginArtifact,
    approved: bool = True,
    scope: ApprovalScope = ApprovalScope.SESSION,
    expired: bool = False,
) -> ApprovalDecision:
    """Helper to create an approval decision."""
    from datetime import timedelta

    expires_at = None
    if expired:
        expires_at = datetime.now(tz=UTC) - timedelta(hours=1)

    return ApprovalDecision(
        id=uuid4(),
        artifact_id=artifact.id,
        artifact_hash=artifact.content_hash(),
        verification_id=None,
        approved=approved,
        reason="Test approval",
        conditions=(),
        decided_by="tester",
        decided_at=datetime.now(tz=UTC),
        decision_context={},
        scope=scope,
        expires_at=expires_at,
    )


class TestLoadArtifact:
    """Tests for load_artifact function."""

    def test_load_simple_plugin(self) -> None:
        """Can load a simple plugin that defines a variable."""
        source = "x = 42"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is True
        assert result.error is None
        assert result.artifact_id == artifact.id
        assert result.approval_id == approval.id

    def test_load_plugin_with_function(self) -> None:
        """Can load a plugin that defines a function."""
        source = '''
def add(a, b):
    return a + b
'''
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is True
        assert result.error is None

    def test_load_plugin_with_operation(self) -> None:
        """Loading a plugin registers its operations."""
        source = '''
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="test.greet",
    description="Say hello",
    required_capabilities=frozenset(),
)
async def greet(name: str = "World") -> dict:
    return {"message": f"Hello, {name}!"}
'''
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is True
        assert "test.greet" in result.operations_registered

        # Verify operation is in registry
        registry = get_global_registry()
        assert "test.greet" in registry

    def test_load_denied_approval_fails(self) -> None:
        """Cannot load with a denied approval."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact, approved=False)

        result = load_artifact(artifact, approval)

        assert result.loaded is False
        assert "denied" in result.error.lower()

    def test_load_expired_approval_fails(self) -> None:
        """Cannot load with an expired approval."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact, expired=True)

        result = load_artifact(artifact, approval)

        assert result.loaded is False
        assert "expired" in result.error.lower()

    def test_load_wrong_artifact_fails(self) -> None:
        """Cannot load with approval for different artifact."""
        source = "x = 1"
        artifact = make_artifact(source)

        # Create approval for different artifact
        other_artifact = make_artifact("y = 2")
        approval = make_approval(other_artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is False
        assert result.error is not None

    def test_load_modified_artifact_fails(self) -> None:
        """Cannot load artifact modified after approval (hash-based scope)."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact, scope=ApprovalScope.HASH_PERMANENT)

        # Create modified artifact with same ID but different content
        modified = PluginArtifact(
            id=artifact.id,
            name=artifact.name,
            description=artifact.description,
            source_code="x = 2",  # Different!
            operations=artifact.operations,
            requested_capabilities=artifact.requested_capabilities,
            generated_by=artifact.generated_by,
            generated_at=artifact.generated_at,
            generation_context=artifact.generation_context,
        )

        result = load_artifact(modified, approval)

        assert result.loaded is False
        assert "modified" in result.error.lower()

    def test_load_with_syntax_error_fails(self) -> None:
        """Plugin with syntax error fails to load."""
        source = "def broken(:\n    pass"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is False
        assert "Execution error" in result.error or "SyntaxError" in result.error

    def test_load_with_runtime_error_fails(self) -> None:
        """Plugin that raises at import time fails to load."""
        source = "raise RuntimeError('oops')"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)

        assert result.loaded is False
        assert "RuntimeError" in result.error

    def test_load_scope_session(self) -> None:
        """Can specify session scope."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval, scope=LoadScope.SESSION)

        assert result.loaded is True
        assert result.scope == LoadScope.SESSION

    def test_load_scope_persistent(self) -> None:
        """Can specify persistent scope."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval, scope=LoadScope.PERSISTENT)

        assert result.loaded is True
        assert result.scope == LoadScope.PERSISTENT

    def test_rollback_on_error(self) -> None:
        """Operations are unregistered if load fails after partial execution."""
        # This plugin registers an operation then fails
        source = '''
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="test.will_fail",
    description="Will fail",
    required_capabilities=frozenset(),
)
async def will_fail() -> dict:
    return {}

# Now raise an error
raise RuntimeError("Intentional failure")
'''
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        # Get operations before
        registry = get_global_registry()
        ops_before = set(registry.list_operations())

        result = load_artifact(artifact, approval)

        # Should fail
        assert result.loaded is False

        # Operation should be unregistered
        ops_after = set(registry.list_operations())
        assert ops_after == ops_before
        assert "test.will_fail" not in registry


class TestPluginLoadOperation:
    """Tests for plugin.load thunk operation."""

    @pytest.mark.asyncio
    async def test_load_by_artifact_id(self) -> None:
        """Can load using artifact ID from storage."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        storage = get_artifact_storage()
        storage.save_artifact(artifact)
        storage.save_approval(approval)

        result = await plugin_load(
            artifact_id=str(artifact.id),
            approval_id=str(approval.id),
        )

        assert result["success"] is True
        assert result["loaded"] is True

    @pytest.mark.asyncio
    async def test_load_by_artifact_dict(self) -> None:
        """Can load using artifact and approval dicts."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = await plugin_load(
            artifact_dict=artifact.to_dict(),
            approval_dict=approval.to_dict(),
        )

        assert result["success"] is True
        assert result["loaded"] is True

    @pytest.mark.asyncio
    async def test_load_finds_approval_automatically(self) -> None:
        """Finds approval for artifact if not specified."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        storage = get_artifact_storage()
        storage.save_artifact(artifact)
        storage.save_approval(approval)

        result = await plugin_load(artifact_id=str(artifact.id))

        assert result["success"] is True
        assert result["loaded"] is True

    @pytest.mark.asyncio
    async def test_missing_artifact_error(self) -> None:
        """Returns error for missing artifact."""
        result = await plugin_load(artifact_id=str(uuid4()))

        assert result["success"] is False
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_missing_approval_error(self) -> None:
        """Returns error when no approval found."""
        source = "x = 1"
        artifact = make_artifact(source)

        storage = get_artifact_storage()
        storage.save_artifact(artifact)

        result = await plugin_load(artifact_id=str(artifact.id))

        assert result["success"] is False
        assert result["error_type"] == "not_approved"

    @pytest.mark.asyncio
    async def test_invalid_artifact_dict_error(self) -> None:
        """Returns error for invalid artifact dict."""
        result = await plugin_load(artifact_dict={"invalid": "data"})

        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_invalid_approval_dict_error(self) -> None:
        """Returns error for invalid approval dict."""
        source = "x = 1"
        artifact = make_artifact(source)

        result = await plugin_load(
            artifact_dict=artifact.to_dict(),
            approval_dict={"invalid": "data"},
        )

        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_missing_input_error(self) -> None:
        """Returns error when neither artifact_id nor artifact_dict provided."""
        result = await plugin_load()

        assert result["success"] is False
        assert result["error_type"] == "missing_input"

    @pytest.mark.asyncio
    async def test_invalid_scope_error(self) -> None:
        """Returns error for invalid scope."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = await plugin_load(
            artifact_dict=artifact.to_dict(),
            approval_dict=approval.to_dict(),
            scope="invalid",
        )

        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_result_saved_to_storage(self) -> None:
        """Load result is saved to storage."""
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        storage = get_artifact_storage()
        storage.save_artifact(artifact)
        storage.save_approval(approval)

        result = await plugin_load(
            artifact_id=str(artifact.id),
            approval_id=str(approval.id),
        )

        # Result should be saved
        result_id = result["result_id"]
        assert result_id is not None

    @pytest.mark.asyncio
    async def test_load_with_operations(self) -> None:
        """Operations are reported in result."""
        source = '''
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="test.loaded_op",
    description="A loaded operation",
    required_capabilities=frozenset(),
)
async def loaded_op() -> dict:
    return {"status": "ok"}
'''
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = await plugin_load(
            artifact_dict=artifact.to_dict(),
            approval_dict=approval.to_dict(),
        )

        assert result["success"] is True
        assert result["loaded"] is True
        assert "test.loaded_op" in result["operations_registered"]


class TestUnloadPlugin:
    """Tests for unload_plugin function."""

    def test_unload_loaded_module(self) -> None:
        """Can unload a loaded module."""
        # First load a plugin
        source = "x = 1"
        artifact = make_artifact(source)
        approval = make_approval(artifact)

        result = load_artifact(artifact, approval)
        assert result.loaded is True

        # Find the module name
        module_name = f"crows_nest_dynamic_{artifact.id.hex[:8]}"
        assert module_name in sys.modules

        # Unload it
        success = unload_plugin(module_name)

        assert success is True
        assert module_name not in sys.modules

    def test_unload_unknown_module(self) -> None:
        """Returns False for unknown module."""
        success = unload_plugin("nonexistent_module")
        assert success is False
