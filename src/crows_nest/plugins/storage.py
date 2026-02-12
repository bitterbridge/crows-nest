"""Storage layer for plugin artifacts and approvals.

Provides filesystem-based persistence for:
- Plugin artifacts (generated code)
- Approval decisions
- Load records
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from crows_nest.plugins.artifacts import (
    ApprovalDecision,
    LoadResult,
    PluginArtifact,
    VerificationReport,
)


class ArtifactStorage:
    """Filesystem storage for plugin artifacts and related data.

    Directory structure:
        plugins/
        ├── artifacts/
        │   └── {uuid}.json          # PluginArtifact
        ├── verifications/
        │   └── {uuid}.json          # VerificationReport
        ├── approvals/
        │   └── {uuid}.json          # ApprovalDecision
        ├── loads/
        │   └── {uuid}.json          # LoadResult
        └── loaded.json              # Currently loaded plugins
    """

    def __init__(self, base_dir: Path | str) -> None:
        """Initialize storage with base directory.

        Args:
            base_dir: Base directory for plugin storage.
        """
        self.base_dir = Path(base_dir)
        self._artifacts_dir = self.base_dir / "artifacts"
        self._verifications_dir = self.base_dir / "verifications"
        self._approvals_dir = self.base_dir / "approvals"
        self._loads_dir = self.base_dir / "loads"
        self._loaded_file = self.base_dir / "loaded.json"

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._verifications_dir.mkdir(parents=True, exist_ok=True)
        self._approvals_dir.mkdir(parents=True, exist_ok=True)
        self._loads_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------

    def save_artifact(self, artifact: PluginArtifact) -> Path:
        """Save a plugin artifact.

        Args:
            artifact: The artifact to save.

        Returns:
            Path to the saved artifact file.
        """
        self.ensure_directories()
        path = self._artifacts_dir / f"{artifact.id}.json"
        path.write_text(artifact.to_json())
        return path

    def load_artifact(self, artifact_id: UUID) -> PluginArtifact | None:
        """Load a plugin artifact by ID.

        Args:
            artifact_id: UUID of the artifact.

        Returns:
            The artifact, or None if not found.
        """
        path = self._artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            return None
        return PluginArtifact.from_json(path.read_text())

    def list_artifacts(self) -> list[PluginArtifact]:
        """List all stored artifacts.

        Returns:
            List of all artifacts.
        """
        if not self._artifacts_dir.exists():
            return []

        artifacts = []
        for path in self._artifacts_dir.glob("*.json"):
            try:
                artifacts.append(PluginArtifact.from_json(path.read_text()))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip invalid files
                continue
        return artifacts

    def delete_artifact(self, artifact_id: UUID) -> bool:
        """Delete a plugin artifact.

        Args:
            artifact_id: UUID of the artifact to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    # -------------------------------------------------------------------------
    # Verification Reports
    # -------------------------------------------------------------------------

    def save_verification(self, report: VerificationReport) -> Path:
        """Save a verification report.

        Args:
            report: The verification report to save.

        Returns:
            Path to the saved file.
        """
        self.ensure_directories()
        path = self._verifications_dir / f"{report.id}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))
        return path

    def load_verification(self, report_id: UUID) -> VerificationReport | None:
        """Load a verification report by ID.

        Args:
            report_id: UUID of the report.

        Returns:
            The report, or None if not found.
        """
        path = self._verifications_dir / f"{report_id}.json"
        if not path.exists():
            return None
        return VerificationReport.from_dict(json.loads(path.read_text()))

    def load_verification_for_artifact(self, artifact_id: UUID) -> VerificationReport | None:
        """Load the latest verification report for an artifact.

        Args:
            artifact_id: UUID of the artifact.

        Returns:
            The latest verification report, or None if not found.
        """
        if not self._verifications_dir.exists():
            return None

        latest: VerificationReport | None = None
        for path in self._verifications_dir.glob("*.json"):
            try:
                report = VerificationReport.from_dict(json.loads(path.read_text()))
                if report.artifact_id == artifact_id and (
                    latest is None or report.executed_at > latest.executed_at
                ):
                    latest = report
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return latest

    def list_verifications(self) -> list[VerificationReport]:
        """List all stored verification reports.

        Returns:
            List of all reports.
        """
        if not self._verifications_dir.exists():
            return []

        reports = []
        for path in self._verifications_dir.glob("*.json"):
            try:
                reports.append(VerificationReport.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return reports

    # -------------------------------------------------------------------------
    # Approval Decisions
    # -------------------------------------------------------------------------

    def save_approval(self, decision: ApprovalDecision) -> Path:
        """Save an approval decision.

        Args:
            decision: The approval decision to save.

        Returns:
            Path to the saved file.
        """
        self.ensure_directories()
        path = self._approvals_dir / f"{decision.id}.json"
        path.write_text(json.dumps(decision.to_dict(), indent=2))
        return path

    def load_approval(self, approval_id: UUID) -> ApprovalDecision | None:
        """Load an approval decision by ID.

        Args:
            approval_id: UUID of the decision.

        Returns:
            The decision, or None if not found.
        """
        path = self._approvals_dir / f"{approval_id}.json"
        if not path.exists():
            return None
        return ApprovalDecision.from_dict(json.loads(path.read_text()))

    def load_approval_for_artifact(self, artifact_id: UUID) -> ApprovalDecision | None:
        """Load the latest valid approval for an artifact.

        Args:
            artifact_id: UUID of the artifact.

        Returns:
            The latest valid approval, or None if not found.
        """
        if not self._approvals_dir.exists():
            return None

        latest: ApprovalDecision | None = None
        for path in self._approvals_dir.glob("*.json"):
            try:
                decision = ApprovalDecision.from_dict(json.loads(path.read_text()))
                if (
                    decision.artifact_id == artifact_id
                    and decision.is_valid()
                    and (latest is None or decision.decided_at > latest.decided_at)
                ):
                    latest = decision
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return latest

    def list_approvals(self, *, valid_only: bool = False) -> list[ApprovalDecision]:
        """List all stored approval decisions.

        Args:
            valid_only: If True, only return currently valid approvals.

        Returns:
            List of approval decisions.
        """
        if not self._approvals_dir.exists():
            return []

        decisions = []
        for path in self._approvals_dir.glob("*.json"):
            try:
                decision = ApprovalDecision.from_dict(json.loads(path.read_text()))
                if not valid_only or decision.is_valid():
                    decisions.append(decision)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return decisions

    # -------------------------------------------------------------------------
    # Load Records
    # -------------------------------------------------------------------------

    def save_load_result(self, result: LoadResult) -> Path:
        """Save a load result.

        Args:
            result: The load result to save.

        Returns:
            Path to the saved file.
        """
        self.ensure_directories()
        path = self._loads_dir / f"{result.id}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2))
        return path

    def load_load_result(self, load_id: UUID) -> LoadResult | None:
        """Load a load result by ID.

        Args:
            load_id: UUID of the load result.

        Returns:
            The load result, or None if not found.
        """
        path = self._loads_dir / f"{load_id}.json"
        if not path.exists():
            return None
        return LoadResult.from_dict(json.loads(path.read_text()))

    def list_load_results(self, *, successful_only: bool = False) -> list[LoadResult]:
        """List all load results.

        Args:
            successful_only: If True, only return successful loads.

        Returns:
            List of load results.
        """
        if not self._loads_dir.exists():
            return []

        results = []
        for path in self._loads_dir.glob("*.json"):
            try:
                result = LoadResult.from_dict(json.loads(path.read_text()))
                if not successful_only or result.loaded:
                    results.append(result)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return results

    # -------------------------------------------------------------------------
    # Currently Loaded Plugins Tracking
    # -------------------------------------------------------------------------

    def get_loaded_plugins(self) -> list[UUID]:
        """Get list of currently loaded plugin artifact IDs.

        Returns:
            List of artifact IDs that are currently loaded.
        """
        if not self._loaded_file.exists():
            return []

        try:
            data = json.loads(self._loaded_file.read_text())
            return [UUID(aid) for aid in data.get("loaded", [])]
        except (json.JSONDecodeError, KeyError, ValueError):
            return []

    def mark_plugin_loaded(self, artifact_id: UUID) -> None:
        """Mark a plugin as currently loaded.

        Args:
            artifact_id: UUID of the artifact that was loaded.
        """
        self.ensure_directories()
        loaded = self.get_loaded_plugins()
        if artifact_id not in loaded:
            loaded.append(artifact_id)
        self._loaded_file.write_text(json.dumps({"loaded": [str(aid) for aid in loaded]}, indent=2))

    def mark_plugin_unloaded(self, artifact_id: UUID) -> None:
        """Mark a plugin as no longer loaded.

        Args:
            artifact_id: UUID of the artifact that was unloaded.
        """
        loaded = self.get_loaded_plugins()
        if artifact_id in loaded:
            loaded.remove(artifact_id)
            self._loaded_file.write_text(
                json.dumps({"loaded": [str(aid) for aid in loaded]}, indent=2)
            )

    def clear_loaded_plugins(self) -> None:
        """Clear the list of loaded plugins (e.g., on restart)."""
        if self._loaded_file.exists():
            self._loaded_file.write_text(json.dumps({"loaded": []}, indent=2))

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_full_provenance(
        self, artifact_id: UUID
    ) -> dict[str, PluginArtifact | VerificationReport | ApprovalDecision | LoadResult | None]:
        """Get the full provenance chain for an artifact.

        Args:
            artifact_id: UUID of the artifact.

        Returns:
            Dict containing artifact, verification, approval, and load result.
        """
        artifact = self.load_artifact(artifact_id)
        verification = self.load_verification_for_artifact(artifact_id)
        approval = self.load_approval_for_artifact(artifact_id)

        # Find the load result that matches the approval
        load_result = None
        if approval:
            for result in self.list_load_results():
                if result.approval_id == approval.id:
                    load_result = result
                    break

        return {
            "artifact": artifact,
            "verification": verification,
            "approval": approval,
            "load_result": load_result,
        }


# Global storage instance
_storage: ArtifactStorage | None = None


def get_artifact_storage(base_dir: Path | str | None = None) -> ArtifactStorage:
    """Get or create the global artifact storage.

    Args:
        base_dir: Base directory for storage. If None, uses default.

    Returns:
        The artifact storage instance.
    """
    global _storage  # noqa: PLW0603
    if _storage is None:
        if base_dir is None:
            base_dir = Path("plugins")
        _storage = ArtifactStorage(base_dir)
    return _storage


def reset_artifact_storage() -> None:
    """Reset the global artifact storage (for testing)."""
    global _storage  # noqa: PLW0603
    _storage = None
