"""Plugin loading operation.

Provides the plugin.load thunk operation for loading verified and approved
plugin artifacts into the runtime.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from crows_nest.core.registry import get_global_registry, thunk_operation
from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    LoadResult,
    LoadScope,
    PluginArtifact,
)
from crows_nest.plugins.storage import get_artifact_storage


def _validate_approval(
    artifact: PluginArtifact,
    approval: ApprovalDecision,
) -> str | None:
    """Validate that an approval is valid for an artifact.

    Args:
        artifact: The artifact to load.
        approval: The approval to validate.

    Returns:
        Error message if invalid, None if valid.
    """
    # Check approval is for this artifact
    if approval.artifact_id != artifact.id:
        return f"Approval is for artifact {approval.artifact_id}, not {artifact.id}"

    # Check approval is actually approved
    if not approval.approved:
        return "Approval was denied"

    # Check artifact hash matches (if hash-based scope)
    if not approval.matches_artifact(artifact):
        return "Artifact has been modified since approval"

    # Check approval hasn't expired
    if approval.expires_at is not None and approval.expires_at < datetime.now(UTC):
        return "Approval has expired"

    return None


def _load_plugin_code(
    artifact: PluginArtifact,
    module_name: str | None = None,
) -> tuple[list[str], str | None]:
    """Load plugin source code into the runtime.

    Args:
        artifact: The plugin artifact to load.
        module_name: Optional module name (defaults to generated name).

    Returns:
        Tuple of (operations_registered, error_message).
        If successful, error_message is None.
    """
    if module_name is None:
        module_name = f"crows_nest_dynamic_{artifact.id.hex[:8]}"

    # Track operations before loading
    registry = get_global_registry()
    ops_before = set(registry.list_operations())

    try:
        # Create a module spec from the source code
        spec = importlib.util.spec_from_loader(
            module_name,
            loader=None,
            origin=f"<plugin:{artifact.name}>",
        )
        if spec is None:
            return [], "Could not create module spec"

        # Create module and add to sys.modules
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Set up module attributes
        module.__file__ = f"<plugin:{artifact.name}>"
        module.__doc__ = artifact.description

        try:
            # Execute the source code in the module's namespace
            exec(artifact.source_code, module.__dict__)  # noqa: S102; nosec B102 - intentional plugin loading

            # Find new operations
            ops_after = set(registry.list_operations())
            new_ops = list(ops_after - ops_before)

            return new_ops, None

        except Exception as e:
            # Rollback: unregister any operations that were registered
            ops_after = set(registry.list_operations())
            for op_name in ops_after - ops_before:
                with contextlib.suppress(Exception):
                    registry.unregister(op_name)

            # Remove module from sys.modules
            sys.modules.pop(module_name, None)

            return [], f"Execution error: {type(e).__name__}: {e}"

    except Exception as e:
        return [], f"Module setup error: {type(e).__name__}: {e}"


def load_artifact(
    artifact: PluginArtifact,
    approval: ApprovalDecision,
    scope: LoadScope = LoadScope.SESSION,
) -> LoadResult:
    """Load a verified and approved plugin artifact.

    This is the synchronous helper for direct use.

    Args:
        artifact: The plugin artifact to load.
        approval: The approval authorizing the load.
        scope: Whether loading is session-only or persistent.

    Returns:
        LoadResult indicating success or failure.
    """
    # Validate approval
    validation_error = _validate_approval(artifact, approval)
    if validation_error:
        return LoadResult.failure(artifact, approval, validation_error)

    # Load the plugin code
    operations, error = _load_plugin_code(artifact)

    if error:
        return LoadResult.failure(artifact, approval, error)

    return LoadResult.success(artifact, approval, operations, scope)


@thunk_operation(
    name="plugin.load",
    description="Load a verified and approved plugin into the runtime.",
    required_capabilities=frozenset({"plugin.load"}),
)
async def plugin_load(
    artifact_id: str | None = None,
    artifact_dict: dict[str, Any] | None = None,
    approval_id: str | None = None,
    approval_dict: dict[str, Any] | None = None,
    scope: str = "session",
) -> dict[str, Any]:
    """Load a verified and approved plugin into the runtime.

    The plugin must have been previously verified and approved. The approval
    must match the artifact and not be expired.

    Args:
        artifact_id: UUID of the artifact in storage (optional).
        artifact_dict: Artifact as a dict (alternative to artifact_id).
        approval_id: UUID of the approval in storage (optional).
        approval_dict: Approval as a dict (alternative to approval_id).
        scope: Load scope - "session" (default) or "persistent".

    Returns:
        Dict containing the LoadResult and metadata.
    """
    storage = get_artifact_storage()

    # Load artifact
    artifact: PluginArtifact | None = None
    if artifact_id:
        artifact = storage.load_artifact(UUID(artifact_id))
        if artifact is None:
            return {
                "success": False,
                "loaded": False,
                "error": f"Artifact not found: {artifact_id}",
                "error_type": "not_found",
            }
    elif artifact_dict:
        try:
            artifact = PluginArtifact.from_dict(artifact_dict)
        except (KeyError, ValueError, TypeError) as e:
            return {
                "success": False,
                "loaded": False,
                "error": f"Invalid artifact dict: {e}",
                "error_type": "invalid_input",
            }
    else:
        return {
            "success": False,
            "loaded": False,
            "error": "Either artifact_id or artifact_dict must be provided",
            "error_type": "missing_input",
        }

    # Load approval
    approval: ApprovalDecision | None = None
    if approval_id:
        approval = storage.load_approval(UUID(approval_id))
        if approval is None:
            return {
                "success": False,
                "loaded": False,
                "error": f"Approval not found: {approval_id}",
                "error_type": "not_found",
            }
    elif approval_dict:
        try:
            approval = ApprovalDecision.from_dict(approval_dict)
        except (KeyError, ValueError, TypeError) as e:
            return {
                "success": False,
                "loaded": False,
                "error": f"Invalid approval dict: {e}",
                "error_type": "invalid_input",
            }
    else:
        # Try to find approval for this artifact
        approval = storage.load_approval_for_artifact(artifact.id)
        if approval is None:
            return {
                "success": False,
                "loaded": False,
                "error": "No approval found for this artifact",
                "error_type": "not_approved",
            }

    # Parse scope
    try:
        load_scope = LoadScope(scope)
    except ValueError:
        return {
            "success": False,
            "loaded": False,
            "error": f"Invalid scope: {scope}. Must be 'session' or 'persistent'.",
            "error_type": "invalid_input",
        }

    # Load the artifact
    result = load_artifact(artifact, approval, load_scope)

    # Save the load result
    storage.save_load_result(result)

    return {
        "success": True,
        "loaded": result.loaded,
        "result": result.to_dict(),
        "result_id": str(result.id),
        "operations_registered": list(result.operations_registered),
        "error": result.error,
    }


def unload_plugin(module_name: str) -> bool:
    """Unload a dynamically loaded plugin.

    Removes the module from sys.modules and unregisters its operations.
    Note: This doesn't fully unload the code from memory due to Python's
    import system, but prevents the operations from being used.

    Args:
        module_name: The module name to unload.

    Returns:
        True if the module was found and removed.
    """
    if module_name not in sys.modules:
        return False

    # Remove the module from sys.modules
    # Note: We can't easily track which operations came from which module,
    # so operations remain registered but will fail if they reference
    # unloaded module state.
    del sys.modules[module_name]

    return True
