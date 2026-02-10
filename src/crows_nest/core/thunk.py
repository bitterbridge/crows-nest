"""Core thunk types and data models.

A thunk is a deferred computation - a lazy, pure, composable unit that can be
serialized, persisted, and executed later. Everything in Crow's Nest is a thunk:
agents, tools, tasks, and their compositions.
"""

from __future__ import annotations

import json
import traceback as tb
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping


class ThunkStatus(Enum):
    """Status of a thunk in its lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ThunkRef:
    """Reference to another thunk's output.

    Used to express dependencies between thunks. When a thunk has a ThunkRef
    in its inputs, the runtime will force the referenced thunk first and
    substitute the result.

    Attributes:
        thunk_id: The UUID of the referenced thunk.
        output_path: Optional JSONPath-like path to extract a subset of the output.
            If None, the entire output is used.
    """

    thunk_id: UUID
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "__thunk_ref__": True,
            "thunk_id": str(self.thunk_id),
            "output_path": self.output_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            thunk_id=UUID(data["thunk_id"]),
            output_path=data.get("output_path"),
        )

    @classmethod
    def is_ref_dict(cls, data: Any) -> bool:
        """Check if a dict represents a serialized ThunkRef."""
        return isinstance(data, dict) and data.get("__thunk_ref__") is True


@dataclass(frozen=True, slots=True)
class ThunkMetadata:
    """Metadata for tracing, debugging, and capability management.

    Attributes:
        created_at: When the thunk was created.
        created_by: ID of the parent thunk that created this one, or "user" for
            user-initiated thunks.
        trace_id: Correlation ID for distributed tracing.
        capabilities: Set of capabilities this thunk is allowed to use.
            Child thunks inherit a subset (inherit-and-restrict model).
        tags: Optional key-value pairs for categorization and filtering.
    """

    created_at: datetime
    created_by: str | None
    trace_id: str
    capabilities: frozenset[str]
    tags: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "trace_id": self.trace_id,
            "capabilities": sorted(self.capabilities),
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            trace_id=data["trace_id"],
            capabilities=frozenset(data.get("capabilities", [])),
            tags=frozenset(data.get("tags", {}).items()),
        )

    def restrict_capabilities(self, allowed: frozenset[str]) -> ThunkMetadata:
        """Create new metadata with capabilities restricted to the given set."""
        return ThunkMetadata(
            created_at=self.created_at,
            created_by=self.created_by,
            trace_id=self.trace_id,
            capabilities=self.capabilities & allowed,
            tags=self.tags,
        )


@dataclass(frozen=True, slots=True)
class ThunkError:
    """Error information from a failed thunk.

    Attributes:
        error_type: The exception class name.
        message: The error message.
        traceback: Optional formatted traceback string.
        context: Optional additional context for debugging.
    """

    error_type: str
    message: str
    traceback: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            error_type=data["error_type"],
            message=data["message"],
            traceback=data.get("traceback"),
            context=dict(data.get("context", {})),
        )

    @classmethod
    def from_exception(cls, exc: BaseException, context: dict[str, Any] | None = None) -> Self:
        """Create from an exception."""
        return cls(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=tb.format_exc(),
            context=context or {},
        )


def _serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Recursively serialize inputs, handling ThunkRefs."""
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, ThunkRef):
            result[key] = value.to_dict()
        elif isinstance(value, dict):
            result[key] = _serialize_inputs(value)
        elif isinstance(value, list):
            result[key] = [item.to_dict() if isinstance(item, ThunkRef) else item for item in value]
        else:
            result[key] = value
    return result


def _deserialize_inputs(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively deserialize inputs, restoring ThunkRefs."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if ThunkRef.is_ref_dict(value):
            result[key] = ThunkRef.from_dict(value)
        elif isinstance(value, dict):
            result[key] = _deserialize_inputs(value)
        elif isinstance(value, list):
            result[key] = [
                ThunkRef.from_dict(item) if ThunkRef.is_ref_dict(item) else item for item in value
            ]
        else:
            result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class Thunk:
    """A deferred computation.

    Thunks are the fundamental unit of computation in Crow's Nest. They represent
    a lazy, pure function call that can be serialized, persisted, and executed
    later by the runtime.

    Attributes:
        id: Unique identifier for this thunk.
        operation: The registered operation name (e.g., "shell.run", "agent.create").
        inputs: JSON-serializable dictionary of inputs. May contain ThunkRefs to
            express dependencies on other thunks.
        metadata: Tracing, capability, and debugging information.

    Note:
        Thunks are immutable and content-addressed by their ID (not by content).
        Two thunks with identical operations and inputs are still distinct thunks.
    """

    id: UUID
    operation: str
    inputs: dict[str, Any]
    metadata: ThunkMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "operation": self.operation,
            "inputs": _serialize_inputs(self.inputs),
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            id=UUID(data["id"]),
            operation=data["operation"],
            inputs=_deserialize_inputs(dict(data["inputs"])),
            metadata=ThunkMetadata.from_dict(data["metadata"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def create(
        cls,
        operation: str,
        inputs: dict[str, Any],
        *,
        capabilities: frozenset[str] | None = None,
        created_by: str | None = None,
        trace_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """Factory method to create a new thunk with auto-generated ID and metadata."""
        return cls(
            id=uuid4(),
            operation=operation,
            inputs=inputs,
            metadata=ThunkMetadata(
                created_at=datetime.now(UTC),
                created_by=created_by,
                trace_id=trace_id or str(uuid4()),
                capabilities=capabilities or frozenset(),
                tags=frozenset((tags or {}).items()),
            ),
        )

    def get_dependencies(self) -> list[ThunkRef]:
        """Extract all ThunkRefs from inputs (direct dependencies)."""
        refs: list[ThunkRef] = []
        self._collect_refs(self.inputs, refs)
        return refs

    def _collect_refs(self, value: Any, refs: list[ThunkRef]) -> None:
        """Recursively collect ThunkRefs from a value."""
        if isinstance(value, ThunkRef):
            refs.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                self._collect_refs(v, refs)
        elif isinstance(value, list):
            for item in value:
                self._collect_refs(item, refs)


@dataclass(frozen=True, slots=True)
class ThunkResult:
    """Result of forcing (executing) a thunk.

    Attributes:
        thunk_id: The ID of the thunk that was forced.
        status: Whether the thunk completed or failed.
        value: The result value if successful, None if failed.
        error: Error information if failed, None if successful.
        forced_at: When the thunk was forced.
        duration_ms: How long execution took in milliseconds.
    """

    thunk_id: UUID
    status: ThunkStatus
    value: Any
    error: ThunkError | None
    forced_at: datetime
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "thunk_id": str(self.thunk_id),
            "status": self.status.value,
            "value": self.value,
            "error": self.error.to_dict() if self.error else None,
            "forced_at": self.forced_at.isoformat(),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a dict."""
        return cls(
            thunk_id=UUID(data["thunk_id"]),
            status=ThunkStatus(data["status"]),
            value=data.get("value"),
            error=ThunkError.from_dict(data["error"]) if data.get("error") else None,
            forced_at=datetime.fromisoformat(data["forced_at"]),
            duration_ms=data["duration_ms"],
        )

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful execution."""
        return self.status == ThunkStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """Check if the result represents a failed execution."""
        return self.status == ThunkStatus.FAILED
