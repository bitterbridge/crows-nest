"""Plugin verification operation.

Provides the sandbox.verify thunk operation for verifying plugin artifacts
in an isolated sandbox environment.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from crows_nest.core.registry import thunk_operation
from crows_nest.plugins.artifacts import PluginArtifact
from crows_nest.plugins.sandbox import PluginVerifier, SandboxConfig, TestCase
from crows_nest.plugins.storage import get_artifact_storage


@thunk_operation(
    name="sandbox.verify",
    description="Execute a plugin in a sandbox and analyze its behavior.",
    required_capabilities=frozenset({"sandbox.run"}),
)
async def sandbox_verify(
    artifact_id: str | None = None,
    artifact_dict: dict[str, Any] | None = None,
    test_cases: list[dict[str, Any]] | None = None,
    timeout_seconds: float = 30.0,
    allow_file_io: bool = False,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Execute plugin in sandbox and analyze behavior.

    The sandbox provides:
    - Isolated execution context
    - Restricted imports and builtins
    - Resource limits
    - Violation detection

    Args:
        artifact_id: UUID of an artifact in storage (optional).
        artifact_dict: Artifact as a dict (alternative to artifact_id).
        test_cases: Test cases to run. Each test case is a dict with:
            - name: Test name
            - operation: Operation to test
            - input_data: Dict of input arguments
            - expected_output: Expected return value (optional)
            - should_succeed: Whether it should succeed (default True)
        timeout_seconds: Maximum execution time.
        allow_file_io: Whether to allow file operations.
        allow_network: Whether to allow network operations.

    Returns:
        Dict containing the VerificationReport as a dict, or error info.
    """
    # Get the artifact
    artifact: PluginArtifact | None = None

    if artifact_id:
        storage = get_artifact_storage()
        artifact = storage.load_artifact(UUID(artifact_id))
        if artifact is None:
            return {
                "success": False,
                "error": f"Artifact not found: {artifact_id}",
                "error_type": "not_found",
            }
    elif artifact_dict:
        try:
            artifact = PluginArtifact.from_dict(artifact_dict)
        except (KeyError, ValueError, TypeError) as e:
            return {
                "success": False,
                "error": f"Invalid artifact dict: {e}",
                "error_type": "invalid_input",
            }
    else:
        return {
            "success": False,
            "error": "Either artifact_id or artifact_dict must be provided",
            "error_type": "missing_input",
        }

    # Convert test case dicts to TestCase objects
    parsed_tests: list[TestCase] = []
    if test_cases:
        for tc in test_cases:
            parsed_tests.append(
                TestCase(
                    name=tc.get("name", "unnamed"),
                    operation=tc["operation"],
                    input_data=tc.get("input_data", {}),
                    expected_output=tc.get("expected_output"),
                    should_succeed=tc.get("should_succeed", True),
                )
            )

    # Configure and run verification
    config = SandboxConfig(
        timeout_seconds=timeout_seconds,
        allow_file_io=allow_file_io,
        allow_network=allow_network,
    )
    verifier = PluginVerifier(config)

    try:
        report = verifier.verify(artifact, parsed_tests)

        # Save the verification report
        storage = get_artifact_storage()
        storage.save_verification(report)

        return {
            "success": True,
            "report": report.to_dict(),
            "report_id": str(report.id),
            "passed": report.passed,
            "risk_level": report.risk_level.value,
            "summary": report.summary,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def verify_artifact(
    artifact: PluginArtifact,
    test_cases: list[TestCase] | None = None,
    config: SandboxConfig | None = None,
) -> dict[str, Any]:
    """Synchronous helper to verify an artifact.

    Args:
        artifact: The plugin artifact to verify.
        test_cases: Optional test cases.
        config: Optional sandbox configuration.

    Returns:
        Dict with verification results.
    """
    verifier = PluginVerifier(config)
    report = verifier.verify(artifact, test_cases)
    return {
        "report": report,
        "passed": report.passed,
        "risk_level": report.risk_level,
        "summary": report.summary,
    }
