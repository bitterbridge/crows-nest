"""Tests for plugin artifact models and storage."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    ApprovalScope,
    LoadResult,
    LoadScope,
    OperationSpec,
    ParameterSpec,
    PluginArtifact,
    ResourceUsage,
    RiskLevel,
    Severity,
    VerificationReport,
    Violation,
    ViolationType,
)
from crows_nest.plugins.artifacts import (
    TestResult as ArtifactTestResult,
)
from crows_nest.plugins.storage import (
    ArtifactStorage,
    get_artifact_storage,
    reset_artifact_storage,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_storage() -> Generator[None, None, None]:
    """Reset artifact storage before and after each test."""
    reset_artifact_storage()
    yield
    reset_artifact_storage()


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage = tmp_path / "plugins"
    storage.mkdir()
    return storage


@pytest.fixture
def sample_artifact() -> PluginArtifact:
    """Create a sample plugin artifact."""
    return PluginArtifact(
        id=uuid4(),
        name="json_parser",
        description="Parse and validate JSON with custom schema",
        source_code="""
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="json.parse",
    description="Parse JSON string",
    required_capabilities=frozenset(),
)
async def json_parse(data: str) -> dict:
    import json
    return json.loads(data)
""",
        operations=(
            OperationSpec(
                name="json.parse",
                description="Parse JSON string",
                parameters=(
                    ParameterSpec(
                        name="data",
                        type_annotation="str",
                        description="JSON string to parse",
                        required=True,
                        default=None,
                    ),
                ),
                required_capabilities=frozenset(),
                return_type="dict",
            ),
        ),
        requested_capabilities=frozenset(),
        generated_by=uuid4(),
        generated_at=datetime.now(tz=UTC),
        generation_context={"reason": "Need JSON parsing capability"},
    )


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_required_parameter(self) -> None:
        """Required parameter has no default."""
        param = ParameterSpec(
            name="data",
            type_annotation="str",
            description="Input data",
            required=True,
            default=None,
        )
        assert param.name == "data"
        assert param.required is True
        assert param.default is None

    def test_optional_parameter(self) -> None:
        """Optional parameter has default value."""
        param = ParameterSpec(
            name="timeout",
            type_annotation="int",
            description="Timeout in seconds",
            required=False,
            default=30,
        )
        assert param.required is False
        assert param.default == 30

    def test_to_dict_and_from_dict(self) -> None:
        """ParameterSpec serializes to/from dict."""
        param = ParameterSpec(
            name="count",
            type_annotation="int",
            description="Number of items",
            required=False,
            default=10,
        )
        d = param.to_dict()
        restored = ParameterSpec.from_dict(d)
        assert restored == param


class TestOperationSpec:
    """Tests for OperationSpec dataclass."""

    def test_basic_operation(self) -> None:
        """OperationSpec captures operation metadata."""
        op = OperationSpec(
            name="test.op",
            description="A test operation",
            parameters=(
                ParameterSpec(
                    name="input",
                    type_annotation="str",
                    description="Input value",
                    required=True,
                    default=None,
                ),
            ),
            required_capabilities=frozenset({"file.read"}),
            return_type="dict",
        )
        assert op.name == "test.op"
        assert len(op.parameters) == 1
        assert "file.read" in op.required_capabilities

    def test_to_dict_and_from_dict(self) -> None:
        """OperationSpec serializes to/from dict."""
        op = OperationSpec(
            name="test.op",
            description="A test operation",
            parameters=(
                ParameterSpec(
                    name="input",
                    type_annotation="str",
                    description="Input",
                    required=True,
                    default=None,
                ),
            ),
            required_capabilities=frozenset({"file.read", "network"}),
            return_type="dict",
        )
        d = op.to_dict()
        restored = OperationSpec.from_dict(d)
        assert restored.name == op.name
        assert restored.required_capabilities == op.required_capabilities


class TestPluginArtifact:
    """Tests for PluginArtifact dataclass."""

    def test_content_hash(self, sample_artifact: PluginArtifact) -> None:
        """Artifact computes deterministic content hash."""
        hash1 = sample_artifact.content_hash()
        hash2 = sample_artifact.content_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_different_code_different_hash(self, sample_artifact: PluginArtifact) -> None:
        """Different source code produces different hash."""
        # Create artifact with different code
        other = PluginArtifact(
            id=sample_artifact.id,
            name=sample_artifact.name,
            description=sample_artifact.description,
            source_code="# Different code",
            operations=sample_artifact.operations,
            requested_capabilities=sample_artifact.requested_capabilities,
            generated_by=sample_artifact.generated_by,
            generated_at=sample_artifact.generated_at,
            generation_context=sample_artifact.generation_context,
        )
        assert sample_artifact.content_hash() != other.content_hash()

    def test_json_serialization(self, sample_artifact: PluginArtifact) -> None:
        """Artifact serializes to/from JSON."""
        json_str = sample_artifact.to_json()
        restored = PluginArtifact.from_json(json_str)

        assert restored.id == sample_artifact.id
        assert restored.name == sample_artifact.name
        assert restored.source_code == sample_artifact.source_code
        assert restored.content_hash() == sample_artifact.content_hash()


class TestResultModel:
    """Tests for ArtifactTestResult dataclass."""

    def test_passed_result(self) -> None:
        """ArtifactTestResult captures passing test."""
        result = ArtifactTestResult(
            test_name="test_parse_valid",
            passed=True,
            input_data={"data": '{"key": "value"}'},
            expected_output={"key": "value"},
            actual_output={"key": "value"},
            error=None,
            duration_ms=5,
        )
        assert result.passed is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """ArtifactTestResult captures failing test."""
        result = ArtifactTestResult(
            test_name="test_parse_invalid",
            passed=False,
            input_data={"data": "not json"},
            expected_output={"error": "parse error"},
            actual_output=None,
            error="JSONDecodeError: Expecting value",
            duration_ms=2,
        )
        assert result.passed is False
        assert result.error is not None


class TestViolation:
    """Tests for Violation dataclass."""

    def test_violation(self) -> None:
        """Violation captures security/resource issue."""
        violation = Violation(
            type=ViolationType.NETWORK,
            severity=Severity.ERROR,
            description="Attempted network connection",
            evidence={"host": "example.com", "port": 443},
        )
        assert violation.type == ViolationType.NETWORK
        assert violation.severity == Severity.ERROR


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_passing_report(self, sample_artifact: PluginArtifact) -> None:
        """VerificationReport captures successful verification."""
        report = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-12345",
            executed_at=datetime.now(tz=UTC),
            duration_ms=150,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(
                ArtifactTestResult(
                    test_name="test_parse",
                    passed=True,
                    input_data={"data": "{}"},
                    expected_output={},
                    actual_output={},
                    error=None,
                    duration_ms=5,
                ),
            ),
            resource_usage=ResourceUsage(
                cpu_time_ms=50,
                memory_peak_mb=10.5,
                file_descriptors=0,
                files_created=0,
                files_modified=0,
            ),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="All tests passed, no violations",
        )
        assert report.passed is True
        assert report.risk_level == RiskLevel.LOW

    def test_serialization(self, sample_artifact: PluginArtifact) -> None:
        """VerificationReport serializes to/from dict."""
        report = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-12345",
            executed_at=datetime.now(tz=UTC),
            duration_ms=150,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(),
            resource_usage=ResourceUsage(
                cpu_time_ms=50,
                memory_peak_mb=10.5,
                file_descriptors=0,
                files_created=0,
                files_modified=0,
            ),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="Passed",
        )
        d = report.to_dict()
        restored = VerificationReport.from_dict(d)
        assert restored.id == report.id
        assert restored.artifact_id == report.artifact_id


class TestApprovalDecision:
    """Tests for ApprovalDecision dataclass."""

    def test_approved_decision(self, sample_artifact: PluginArtifact) -> None:
        """ApprovalDecision captures approval."""
        decision = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=True,
            reason="Reviewed and approved",
            conditions=("Only for development use",),
            decided_by="user@example.com",
            decided_at=datetime.now(tz=UTC),
            decision_context={"reviewed_code": True},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        assert decision.approved is True
        assert decision.is_valid() is True

    def test_denied_decision(self, sample_artifact: PluginArtifact) -> None:
        """ApprovalDecision captures denial."""
        decision = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=False,
            reason="Code looks suspicious",
            conditions=(),
            decided_by="security@example.com",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.ONCE,
            expires_at=None,
        )
        assert decision.approved is False
        assert decision.is_valid() is False  # Denied = not valid

    def test_expired_decision(self, sample_artifact: PluginArtifact) -> None:
        """Expired approval is not valid."""
        decision = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=True,
            reason="Temporary approval",
            conditions=(),
            decided_by="user@example.com",
            decided_at=datetime.now(tz=UTC) - timedelta(days=2),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=datetime.now(tz=UTC) - timedelta(days=1),  # Expired yesterday
        )
        assert decision.approved is True
        assert decision.is_valid() is False  # Expired


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_successful_load(self, sample_artifact: PluginArtifact) -> None:
        """LoadResult captures successful plugin load."""
        result = LoadResult(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            approval_id=uuid4(),
            loaded=True,
            error=None,
            operations_registered=("json.parse",),
            capabilities_granted=frozenset(),
            loaded_at=datetime.now(tz=UTC),
            scope=LoadScope.SESSION,
        )
        assert result.loaded is True
        assert result.error is None

    def test_failed_load(self, sample_artifact: PluginArtifact) -> None:
        """LoadResult captures failed plugin load."""
        result = LoadResult(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            approval_id=uuid4(),
            loaded=False,
            error="ImportError: module not found",
            operations_registered=(),
            capabilities_granted=frozenset(),
            loaded_at=datetime.now(tz=UTC),
            scope=LoadScope.SESSION,
        )
        assert result.loaded is False
        assert "ImportError" in result.error  # type: ignore[operator]


class TestArtifactStorage:
    """Tests for ArtifactStorage class."""

    def test_save_and_load_artifact(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage saves and loads artifacts."""
        storage = ArtifactStorage(storage_dir)
        path = storage.save_artifact(sample_artifact)

        assert path.exists()
        assert path.suffix == ".json"

        loaded = storage.load_artifact(sample_artifact.id)
        assert loaded is not None
        assert loaded.id == sample_artifact.id
        assert loaded.content_hash() == sample_artifact.content_hash()

    def test_load_nonexistent_artifact(self, storage_dir: Path) -> None:
        """Loading nonexistent artifact returns None."""
        storage = ArtifactStorage(storage_dir)
        loaded = storage.load_artifact(uuid4())
        assert loaded is None

    def test_list_artifacts(self, storage_dir: Path, sample_artifact: PluginArtifact) -> None:
        """Storage lists all artifacts."""
        storage = ArtifactStorage(storage_dir)

        # Save multiple artifacts
        storage.save_artifact(sample_artifact)

        artifact2 = PluginArtifact(
            id=uuid4(),
            name="other_plugin",
            description="Another plugin",
            source_code="# Other code",
            operations=(),
            requested_capabilities=frozenset(),
            generated_by=uuid4(),
            generated_at=datetime.now(tz=UTC),
            generation_context={},
        )
        storage.save_artifact(artifact2)

        artifacts = storage.list_artifacts()
        assert len(artifacts) == 2
        ids = {a.id for a in artifacts}
        assert sample_artifact.id in ids
        assert artifact2.id in ids

    def test_delete_artifact(self, storage_dir: Path, sample_artifact: PluginArtifact) -> None:
        """Storage deletes artifacts."""
        storage = ArtifactStorage(storage_dir)
        storage.save_artifact(sample_artifact)

        assert storage.load_artifact(sample_artifact.id) is not None

        deleted = storage.delete_artifact(sample_artifact.id)
        assert deleted is True
        assert storage.load_artifact(sample_artifact.id) is None

        # Deleting again returns False
        deleted = storage.delete_artifact(sample_artifact.id)
        assert deleted is False

    def test_save_and_load_verification(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage saves and loads verification reports."""
        storage = ArtifactStorage(storage_dir)
        report = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-test",
            executed_at=datetime.now(tz=UTC),
            duration_ms=100,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(),
            resource_usage=ResourceUsage(
                cpu_time_ms=20,
                memory_peak_mb=5.0,
                file_descriptors=0,
                files_created=0,
                files_modified=0,
            ),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="Passed",
        )

        path = storage.save_verification(report)
        assert path.exists()

        loaded = storage.load_verification(report.id)
        assert loaded is not None
        assert loaded.id == report.id
        assert loaded.artifact_id == sample_artifact.id

    def test_load_verification_for_artifact(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage finds latest verification for artifact."""
        storage = ArtifactStorage(storage_dir)

        # Save older verification
        old_report = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-old",
            executed_at=datetime.now(tz=UTC) - timedelta(hours=1),
            duration_ms=100,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(),
            resource_usage=ResourceUsage(20, 5.0, 0, 0, 0),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="Old",
        )
        storage.save_verification(old_report)

        # Save newer verification
        new_report = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-new",
            executed_at=datetime.now(tz=UTC),
            duration_ms=100,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(),
            resource_usage=ResourceUsage(20, 5.0, 0, 0, 0),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="New",
        )
        storage.save_verification(new_report)

        latest = storage.load_verification_for_artifact(sample_artifact.id)
        assert latest is not None
        assert latest.id == new_report.id

    def test_save_and_load_approval(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage saves and loads approval decisions."""
        storage = ArtifactStorage(storage_dir)
        decision = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=True,
            reason="Approved for testing",
            conditions=(),
            decided_by="tester",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )

        path = storage.save_approval(decision)
        assert path.exists()

        loaded = storage.load_approval(decision.id)
        assert loaded is not None
        assert loaded.approved is True

    def test_load_approval_for_artifact(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage finds latest valid approval for artifact."""
        storage = ArtifactStorage(storage_dir)

        # Save expired approval
        expired = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=True,
            reason="Expired",
            conditions=(),
            decided_by="user",
            decided_at=datetime.now(tz=UTC) - timedelta(days=2),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=datetime.now(tz=UTC) - timedelta(days=1),
        )
        storage.save_approval(expired)

        # Save valid approval
        valid = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=uuid4(),
            approved=True,
            reason="Valid",
            conditions=(),
            decided_by="user",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        storage.save_approval(valid)

        latest = storage.load_approval_for_artifact(sample_artifact.id)
        assert latest is not None
        assert latest.id == valid.id

    def test_save_and_load_load_result(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage saves and loads load results."""
        storage = ArtifactStorage(storage_dir)
        result = LoadResult(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            approval_id=uuid4(),
            loaded=True,
            error=None,
            operations_registered=("json.parse",),
            capabilities_granted=frozenset(),
            loaded_at=datetime.now(tz=UTC),
            scope=LoadScope.SESSION,
        )

        path = storage.save_load_result(result)
        assert path.exists()

        loaded = storage.load_load_result(result.id)
        assert loaded is not None
        assert loaded.loaded is True

    def test_loaded_plugins_tracking(
        self, storage_dir: Path, sample_artifact: PluginArtifact
    ) -> None:
        """Storage tracks currently loaded plugins."""
        storage = ArtifactStorage(storage_dir)

        assert storage.get_loaded_plugins() == []

        storage.mark_plugin_loaded(sample_artifact.id)
        loaded = storage.get_loaded_plugins()
        assert sample_artifact.id in loaded

        # Mark another as loaded
        other_id = uuid4()
        storage.mark_plugin_loaded(other_id)
        loaded = storage.get_loaded_plugins()
        assert len(loaded) == 2

        # Unload one
        storage.mark_plugin_unloaded(sample_artifact.id)
        loaded = storage.get_loaded_plugins()
        assert sample_artifact.id not in loaded
        assert other_id in loaded

        # Clear all
        storage.clear_loaded_plugins()
        assert storage.get_loaded_plugins() == []

    def test_get_full_provenance(self, storage_dir: Path, sample_artifact: PluginArtifact) -> None:
        """Storage retrieves full provenance chain."""
        storage = ArtifactStorage(storage_dir)

        # Save artifact
        storage.save_artifact(sample_artifact)

        # Save verification
        verification = VerificationReport(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            sandbox_id="sandbox-test",
            executed_at=datetime.now(tz=UTC),
            duration_ms=100,
            loaded_successfully=True,
            operations_registered=("json.parse",),
            test_results=(),
            resource_usage=ResourceUsage(20, 5.0, 0, 0, 0),
            violations=(),
            passed=True,
            risk_level=RiskLevel.LOW,
            summary="Passed",
        )
        storage.save_verification(verification)

        # Save approval
        approval = ApprovalDecision(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            artifact_hash=sample_artifact.content_hash(),
            verification_id=verification.id,
            approved=True,
            reason="Approved",
            conditions=(),
            decided_by="user",
            decided_at=datetime.now(tz=UTC),
            decision_context={},
            scope=ApprovalScope.SESSION,
            expires_at=None,
        )
        storage.save_approval(approval)

        # Save load result
        load_result = LoadResult(
            id=uuid4(),
            artifact_id=sample_artifact.id,
            approval_id=approval.id,
            loaded=True,
            error=None,
            operations_registered=("json.parse",),
            capabilities_granted=frozenset(),
            loaded_at=datetime.now(tz=UTC),
            scope=LoadScope.SESSION,
        )
        storage.save_load_result(load_result)

        # Get full provenance
        provenance = storage.get_full_provenance(sample_artifact.id)
        assert provenance["artifact"] is not None
        assert provenance["verification"] is not None
        assert provenance["approval"] is not None
        assert provenance["load_result"] is not None


class TestGlobalStorage:
    """Tests for global storage functions."""

    def test_get_artifact_storage_default(self) -> None:
        """get_artifact_storage creates default storage."""
        storage = get_artifact_storage()
        assert storage is not None
        assert storage.base_dir == Path("plugins")

    def test_get_artifact_storage_custom_dir(self) -> None:
        """get_artifact_storage accepts custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reset_artifact_storage()
            storage = get_artifact_storage(tmpdir)
            assert storage.base_dir == Path(tmpdir)

    def test_get_artifact_storage_singleton(self) -> None:
        """get_artifact_storage returns same instance."""
        storage1 = get_artifact_storage()
        storage2 = get_artifact_storage()
        assert storage1 is storage2
