"""Plugin artifact models for dynamic plugin generation.

A PluginArtifact represents generated plugin code that hasn't been loaded yet.
It goes through verification and approval before being loaded into the registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping


class RiskLevel(Enum):
    """Risk level for a plugin or operation."""

    LOW = "low"  # Simple data transformation, no I/O
    MEDIUM = "medium"  # File I/O within working directory
    HIGH = "high"  # Network, shell, or broad file access
    CRITICAL = "critical"  # Requests dangerous capabilities


class ApprovalScope(Enum):
    """Scope of an approval decision."""

    ONCE = "once"  # This execution only
    SESSION = "session"  # Until restart
    PERMANENT = "permanent"  # Persisted approval
    HASH_PERMANENT = "hash_permanent"  # Permanent for this exact code hash


class LoadScope(Enum):
    """Scope for loading a plugin."""

    SESSION = "session"  # Current runtime only
    PERSISTENT = "persistent"  # Saved and reloaded on restart


class ViolationType(Enum):
    """Type of security violation detected."""

    NETWORK = "network"  # Attempted network access
    FILESYSTEM = "filesystem"  # Unauthorized file access
    RESOURCE = "resource"  # Resource limit exceeded
    CAPABILITY = "capability"  # Attempted to use unauthorized capability
    CODE = "code"  # Dangerous code pattern detected


class Severity(Enum):
    """Severity of a violation."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Specification of an operation parameter.

    Attributes:
        name: Parameter name.
        type_annotation: Type annotation as string (e.g., "str", "dict[str, Any]").
        description: Human-readable description.
        required: Whether the parameter is required.
        default: Default value if not required.
    """

    name: str
    type_annotation: str
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            name=data["name"],
            type_annotation=data["type_annotation"],
            description=data["description"],
            required=data.get("required", True),
            default=data.get("default"),
        )


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Specification of an operation the plugin will register.

    Attributes:
        name: Operation name (e.g., "json.parse").
        description: Human-readable description.
        parameters: Parameter specifications.
        required_capabilities: Capabilities needed to execute.
        return_type: Return type annotation as string.
    """

    name: str
    description: str
    parameters: tuple[ParameterSpec, ...]
    required_capabilities: frozenset[str]
    return_type: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "required_capabilities": sorted(self.required_capabilities),
            "return_type": self.return_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=tuple(ParameterSpec.from_dict(p) for p in data["parameters"]),
            required_capabilities=frozenset(data["required_capabilities"]),
            return_type=data["return_type"],
        )


@dataclass(frozen=True, slots=True)
class PluginArtifact:
    """Generated plugin code that hasn't been loaded yet.

    This is the output of plugin.generate - it contains the source code
    and metadata but has not been executed or registered.

    Attributes:
        id: Unique identifier for this artifact.
        name: Plugin name (e.g., "json_parser").
        description: What the plugin does.
        source_code: The actual Python code.
        operations: Operations this plugin will register.
        requested_capabilities: Capabilities the plugin needs.
        generated_by: Agent ID that requested generation.
        generated_at: When the plugin was generated.
        generation_context: Why it was generated (for audit).
    """

    id: UUID
    name: str
    description: str
    source_code: str
    operations: tuple[OperationSpec, ...]
    requested_capabilities: frozenset[str]
    generated_by: UUID
    generated_at: datetime
    generation_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        source_code: str,
        operations: list[OperationSpec],
        requested_capabilities: set[str],
        generated_by: UUID,
        generation_context: dict[str, Any] | None = None,
    ) -> Self:
        """Create a new plugin artifact.

        Args:
            name: Plugin name.
            description: What the plugin does.
            source_code: The Python source code.
            operations: Operations it will register.
            requested_capabilities: Capabilities it needs.
            generated_by: Agent that requested generation.
            generation_context: Why it was generated.

        Returns:
            New PluginArtifact instance.
        """
        return cls(
            id=uuid4(),
            name=name,
            description=description,
            source_code=source_code,
            operations=tuple(operations),
            requested_capabilities=frozenset(requested_capabilities),
            generated_by=generated_by,
            generated_at=datetime.now(UTC),
            generation_context=generation_context or {},
        )

    def content_hash(self) -> str:
        """Compute SHA-256 hash of the source code.

        Used to verify the code hasn't changed between approval and load.
        """
        return hashlib.sha256(self.source_code.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "source_code": self.source_code,
            "operations": [op.to_dict() for op in self.operations],
            "requested_capabilities": sorted(self.requested_capabilities),
            "generated_by": str(self.generated_by),
            "generated_at": self.generated_at.isoformat(),
            "generation_context": self.generation_context,
            "content_hash": self.content_hash(),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data["description"],
            source_code=data["source_code"],
            operations=tuple(OperationSpec.from_dict(op) for op in data["operations"]),
            requested_capabilities=frozenset(data["requested_capabilities"]),
            generated_by=UUID(data["generated_by"]),
            generated_at=datetime.fromisoformat(data["generated_at"]),
            generation_context=data.get("generation_context", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, slots=True)
class TestResult:
    """Result of a single test case during verification.

    Attributes:
        test_name: Name of the test.
        passed: Whether the test passed.
        input_data: Input provided to the operation.
        expected_output: What was expected.
        actual_output: What was returned.
        error: Error message if the test failed.
        duration_ms: How long the test took.
    """

    test_name: str
    passed: bool
    input_data: dict[str, Any]
    expected_output: Any
    actual_output: Any
    error: str | None
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            test_name=data["test_name"],
            passed=data["passed"],
            input_data=data["input_data"],
            expected_output=data["expected_output"],
            actual_output=data["actual_output"],
            error=data.get("error"),
            duration_ms=data["duration_ms"],
        )


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource usage during sandbox execution.

    Attributes:
        cpu_time_ms: CPU time used in milliseconds.
        memory_peak_mb: Peak memory usage in megabytes.
        file_descriptors: Number of file descriptors opened.
        files_created: Number of files created.
        files_modified: Number of files modified.
    """

    cpu_time_ms: int
    memory_peak_mb: float
    file_descriptors: int
    files_created: int
    files_modified: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "cpu_time_ms": self.cpu_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "file_descriptors": self.file_descriptors,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            cpu_time_ms=data["cpu_time_ms"],
            memory_peak_mb=data["memory_peak_mb"],
            file_descriptors=data["file_descriptors"],
            files_created=data["files_created"],
            files_modified=data["files_modified"],
        )


@dataclass(frozen=True, slots=True)
class Violation:
    """Security or resource violation detected during verification.

    Attributes:
        type: Type of violation.
        severity: How severe the violation is.
        description: Human-readable description.
        evidence: Details about what triggered the violation.
    """

    type: ViolationType
    severity: Severity
    description: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            type=ViolationType(data["type"]),
            severity=Severity(data["severity"]),
            description=data["description"],
            evidence=data["evidence"],
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Results from sandbox verification.

    Attributes:
        id: Unique identifier for this report.
        artifact_id: ID of the artifact that was verified.
        artifact_hash: Content hash at time of verification.
        sandbox_id: Identifier for the sandbox instance.
        executed_at: When verification was performed.
        duration_ms: Total verification time.
        loaded_successfully: Whether the plugin loaded in sandbox.
        operations_registered: Operations that were registered.
        test_results: Results of test cases.
        resource_usage: Resource usage during execution.
        violations: Any violations detected.
        passed: Overall pass/fail verdict.
        risk_level: Assessed risk level.
        summary: Human-readable summary.
    """

    id: UUID
    artifact_id: UUID
    artifact_hash: str
    sandbox_id: str
    executed_at: datetime
    duration_ms: int
    loaded_successfully: bool
    operations_registered: tuple[str, ...]
    test_results: tuple[TestResult, ...]
    resource_usage: ResourceUsage
    violations: tuple[Violation, ...]
    passed: bool
    risk_level: RiskLevel
    summary: str

    @classmethod
    def create(
        cls,
        artifact: PluginArtifact,
        sandbox_id: str,
        duration_ms: int,
        loaded_successfully: bool,
        operations_registered: list[str],
        test_results: list[TestResult],
        resource_usage: ResourceUsage,
        violations: list[Violation],
    ) -> Self:
        """Create a verification report.

        Automatically computes passed and risk_level based on results.
        """
        # Determine if passed
        passed = (
            loaded_successfully
            and all(t.passed for t in test_results)
            and not any(v.severity == Severity.CRITICAL for v in violations)
        )

        # Determine risk level
        if any(v.severity == Severity.CRITICAL for v in violations):
            risk_level = RiskLevel.CRITICAL
        elif any(v.severity == Severity.ERROR for v in violations):
            risk_level = RiskLevel.HIGH
        elif any(v.type in (ViolationType.NETWORK, ViolationType.FILESYSTEM) for v in violations):
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Generate summary
        test_pass_count = sum(1 for t in test_results if t.passed)
        test_total = len(test_results)
        violation_count = len(violations)

        if passed:
            summary = f"Passed: {test_pass_count}/{test_total} tests, {violation_count} violations"
        else:
            failures = [t.test_name for t in test_results if not t.passed]
            summary = f"Failed: {failures[:3]}{'...' if len(failures) > 3 else ''}"

        return cls(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            sandbox_id=sandbox_id,
            executed_at=datetime.now(UTC),
            duration_ms=duration_ms,
            loaded_successfully=loaded_successfully,
            operations_registered=tuple(operations_registered),
            test_results=tuple(test_results),
            resource_usage=resource_usage,
            violations=tuple(violations),
            passed=passed,
            risk_level=risk_level,
            summary=summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "artifact_id": str(self.artifact_id),
            "artifact_hash": self.artifact_hash,
            "sandbox_id": self.sandbox_id,
            "executed_at": self.executed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "loaded_successfully": self.loaded_successfully,
            "operations_registered": list(self.operations_registered),
            "test_results": [t.to_dict() for t in self.test_results],
            "resource_usage": self.resource_usage.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "passed": self.passed,
            "risk_level": self.risk_level.value,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            id=UUID(data["id"]),
            artifact_id=UUID(data["artifact_id"]),
            artifact_hash=data["artifact_hash"],
            sandbox_id=data["sandbox_id"],
            executed_at=datetime.fromisoformat(data["executed_at"]),
            duration_ms=data["duration_ms"],
            loaded_successfully=data["loaded_successfully"],
            operations_registered=tuple(data["operations_registered"]),
            test_results=tuple(TestResult.from_dict(t) for t in data["test_results"]),
            resource_usage=ResourceUsage.from_dict(data["resource_usage"]),
            violations=tuple(Violation.from_dict(v) for v in data["violations"]),
            passed=data["passed"],
            risk_level=RiskLevel(data["risk_level"]),
            summary=data["summary"],
        )


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Human decision on plugin approval.

    Attributes:
        id: Unique identifier for this decision.
        artifact_id: ID of the artifact being approved.
        artifact_hash: Content hash at time of approval.
        verification_id: ID of the verification report (if any).
        approved: Whether the plugin was approved.
        reason: Human explanation for the decision.
        conditions: Any conditions on the approval.
        decided_by: User identifier.
        decided_at: When the decision was made.
        decision_context: What info was shown to user.
        scope: How broadly the approval applies.
        expires_at: Optional expiration time.
    """

    id: UUID
    artifact_id: UUID
    artifact_hash: str
    verification_id: UUID | None
    approved: bool
    reason: str
    conditions: tuple[str, ...]
    decided_by: str
    decided_at: datetime
    decision_context: dict[str, Any]
    scope: ApprovalScope
    expires_at: datetime | None

    @classmethod
    def create(
        cls,
        artifact: PluginArtifact,
        verification: VerificationReport | None,
        approved: bool,
        reason: str,
        decided_by: str,
        conditions: list[str] | None = None,
        scope: ApprovalScope = ApprovalScope.SESSION,
        expires_at: datetime | None = None,
        decision_context: dict[str, Any] | None = None,
    ) -> Self:
        """Create an approval decision."""
        return cls(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            verification_id=verification.id if verification else None,
            approved=approved,
            reason=reason,
            conditions=tuple(conditions or []),
            decided_by=decided_by,
            decided_at=datetime.now(UTC),
            decision_context=decision_context or {},
            scope=scope,
            expires_at=expires_at,
        )

    def is_valid(self) -> bool:
        """Check if the approval is still valid (not expired)."""
        if not self.approved:
            return False
        return not (self.expires_at and datetime.now(UTC) > self.expires_at)

    def matches_artifact(self, artifact: PluginArtifact) -> bool:
        """Check if this approval is for the given artifact."""
        if self.artifact_id != artifact.id:
            return False
        if self.scope == ApprovalScope.HASH_PERMANENT:
            return self.artifact_hash == artifact.content_hash()
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "artifact_id": str(self.artifact_id),
            "artifact_hash": self.artifact_hash,
            "verification_id": str(self.verification_id) if self.verification_id else None,
            "approved": self.approved,
            "reason": self.reason,
            "conditions": list(self.conditions),
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat(),
            "decision_context": self.decision_context,
            "scope": self.scope.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            id=UUID(data["id"]),
            artifact_id=UUID(data["artifact_id"]),
            artifact_hash=data["artifact_hash"],
            verification_id=UUID(data["verification_id"]) if data.get("verification_id") else None,
            approved=data["approved"],
            reason=data["reason"],
            conditions=tuple(data["conditions"]),
            decided_by=data["decided_by"],
            decided_at=datetime.fromisoformat(data["decided_at"]),
            decision_context=data.get("decision_context", {}),
            scope=ApprovalScope(data["scope"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Result of loading a plugin.

    Attributes:
        id: Unique identifier for this load instance.
        artifact_id: ID of the artifact that was loaded.
        approval_id: ID of the approval that authorized loading.
        loaded: Whether loading succeeded.
        error: Error message if loading failed.
        operations_registered: Operations that were registered.
        capabilities_granted: Capabilities that were granted.
        loaded_at: When the plugin was loaded.
        scope: Whether this is session-only or persistent.
    """

    id: UUID
    artifact_id: UUID
    approval_id: UUID
    loaded: bool
    error: str | None
    operations_registered: tuple[str, ...]
    capabilities_granted: frozenset[str]
    loaded_at: datetime
    scope: LoadScope

    @classmethod
    def success(
        cls,
        artifact: PluginArtifact,
        approval: ApprovalDecision,
        operations_registered: list[str],
        scope: LoadScope = LoadScope.SESSION,
    ) -> Self:
        """Create a successful load result."""
        return cls(
            id=uuid4(),
            artifact_id=artifact.id,
            approval_id=approval.id,
            loaded=True,
            error=None,
            operations_registered=tuple(operations_registered),
            capabilities_granted=artifact.requested_capabilities,
            loaded_at=datetime.now(UTC),
            scope=scope,
        )

    @classmethod
    def failure(
        cls,
        artifact: PluginArtifact,
        approval: ApprovalDecision,
        error: str,
    ) -> Self:
        """Create a failed load result."""
        return cls(
            id=uuid4(),
            artifact_id=artifact.id,
            approval_id=approval.id,
            loaded=False,
            error=error,
            operations_registered=(),
            capabilities_granted=frozenset(),
            loaded_at=datetime.now(UTC),
            scope=LoadScope.SESSION,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "artifact_id": str(self.artifact_id),
            "approval_id": str(self.approval_id),
            "loaded": self.loaded,
            "error": self.error,
            "operations_registered": list(self.operations_registered),
            "capabilities_granted": sorted(self.capabilities_granted),
            "loaded_at": self.loaded_at.isoformat(),
            "scope": self.scope.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            id=UUID(data["id"]),
            artifact_id=UUID(data["artifact_id"]),
            approval_id=UUID(data["approval_id"]),
            loaded=data["loaded"],
            error=data.get("error"),
            operations_registered=tuple(data["operations_registered"]),
            capabilities_granted=frozenset(data["capabilities_granted"]),
            loaded_at=datetime.fromisoformat(data["loaded_at"]),
            scope=LoadScope(data["scope"]),
        )
