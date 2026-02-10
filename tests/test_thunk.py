"""Tests for core thunk types."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from crows_nest.core import (
    Thunk,
    ThunkError,
    ThunkMetadata,
    ThunkRef,
    ThunkResult,
    ThunkStatus,
)


class TestThunkRef:
    """Tests for ThunkRef."""

    def test_create_ref(self):
        """Create a basic ThunkRef."""
        thunk_id = uuid4()
        ref = ThunkRef(thunk_id=thunk_id)
        assert ref.thunk_id == thunk_id
        assert ref.output_path is None

    def test_create_ref_with_path(self):
        """Create a ThunkRef with output path."""
        thunk_id = uuid4()
        ref = ThunkRef(thunk_id=thunk_id, output_path="result.data")
        assert ref.output_path == "result.data"

    def test_serialization_roundtrip(self):
        """ThunkRef serializes and deserializes correctly."""
        original = ThunkRef(thunk_id=uuid4(), output_path="foo.bar")
        data = original.to_dict()
        restored = ThunkRef.from_dict(data)
        assert restored == original

    def test_is_ref_dict(self):
        """is_ref_dict correctly identifies serialized refs."""
        ref = ThunkRef(thunk_id=uuid4())
        data = ref.to_dict()
        assert ThunkRef.is_ref_dict(data) is True
        assert ThunkRef.is_ref_dict({"foo": "bar"}) is False
        assert ThunkRef.is_ref_dict("not a dict") is False


class TestThunkMetadata:
    """Tests for ThunkMetadata."""

    def test_create_metadata(self):
        """Create basic metadata."""
        now = datetime.now(UTC)
        meta = ThunkMetadata(
            created_at=now,
            created_by="user",
            trace_id="trace-123",
            capabilities=frozenset({"shell.run", "file.read"}),
        )
        assert meta.created_at == now
        assert meta.created_by == "user"
        assert meta.trace_id == "trace-123"
        assert meta.capabilities == frozenset({"shell.run", "file.read"})
        assert meta.tags == frozenset()

    def test_serialization_roundtrip(self):
        """ThunkMetadata serializes and deserializes correctly."""
        original = ThunkMetadata(
            created_at=datetime.now(UTC),
            created_by="parent-thunk",
            trace_id="trace-456",
            capabilities=frozenset({"cap1", "cap2"}),
            tags=frozenset([("env", "test"), ("priority", "high")]),
        )
        data = original.to_dict()
        restored = ThunkMetadata.from_dict(data)
        assert restored == original

    def test_restrict_capabilities(self):
        """restrict_capabilities creates new metadata with fewer capabilities."""
        meta = ThunkMetadata(
            created_at=datetime.now(UTC),
            created_by=None,
            trace_id="trace",
            capabilities=frozenset({"a", "b", "c"}),
        )
        restricted = meta.restrict_capabilities(frozenset({"b", "c", "d"}))
        assert restricted.capabilities == frozenset({"b", "c"})
        # Original unchanged
        assert meta.capabilities == frozenset({"a", "b", "c"})


class TestThunkError:
    """Tests for ThunkError."""

    def test_create_error(self):
        """Create a basic error."""
        error = ThunkError(
            error_type="ValueError",
            message="Something went wrong",
        )
        assert error.error_type == "ValueError"
        assert error.message == "Something went wrong"

    def test_from_exception(self):
        """Create error from an exception."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            error = ThunkError.from_exception(e, {"key": "value"})
            assert error.error_type == "ValueError"
            assert error.message == "test error"
            assert error.traceback is not None
            assert "ValueError" in error.traceback
            assert error.context == {"key": "value"}

    def test_serialization_roundtrip(self):
        """ThunkError serializes and deserializes correctly."""
        original = ThunkError(
            error_type="RuntimeError",
            message="failed",
            traceback="Traceback...",
            context={"op": "test"},
        )
        data = original.to_dict()
        restored = ThunkError.from_dict(data)
        assert restored == original


class TestThunk:
    """Tests for the main Thunk class."""

    def test_create_thunk_factory(self):
        """Thunk.create factory generates proper thunk."""
        thunk = Thunk.create(
            operation="test.op",
            inputs={"a": 1, "b": "hello"},
            capabilities=frozenset({"test"}),
            created_by="user",
            tags={"env": "test"},
        )
        assert thunk.operation == "test.op"
        assert thunk.inputs == {"a": 1, "b": "hello"}
        assert thunk.metadata.capabilities == frozenset({"test"})
        assert thunk.metadata.created_by == "user"
        assert isinstance(thunk.id, UUID)
        assert thunk.metadata.trace_id is not None

    def test_serialization_roundtrip(self):
        """Thunk serializes and deserializes correctly."""
        original = Thunk.create(
            operation="math.add",
            inputs={"a": 1, "b": 2},
            capabilities=frozenset({"math"}),
        )
        data = original.to_dict()
        restored = Thunk.from_dict(data)
        assert restored == original

    def test_json_roundtrip(self):
        """Thunk serializes to and from JSON."""
        original = Thunk.create(
            operation="test.op",
            inputs={"nested": {"key": "value"}},
        )
        json_str = original.to_json()
        restored = Thunk.from_json(json_str)
        assert restored == original

    def test_get_dependencies_empty(self):
        """Thunk with no refs has no dependencies."""
        thunk = Thunk.create("test.op", {"a": 1, "b": 2})
        assert thunk.get_dependencies() == []

    def test_get_dependencies_with_refs(self):
        """Thunk correctly extracts dependencies from inputs."""
        ref1 = ThunkRef(thunk_id=uuid4())
        ref2 = ThunkRef(thunk_id=uuid4())
        thunk = Thunk.create(
            "test.op",
            {
                "direct": ref1,
                "nested": {"inner": ref2},
                "in_list": [ref1],
                "plain": "value",
            },
        )
        deps = thunk.get_dependencies()
        assert len(deps) == 3  # ref1 appears twice
        assert ref1 in deps
        assert ref2 in deps

    def test_inputs_with_thunk_ref_serialize(self):
        """Inputs containing ThunkRefs serialize correctly."""
        ref = ThunkRef(thunk_id=uuid4(), output_path="data")
        thunk = Thunk.create("test.op", {"ref": ref, "value": 42})

        data = thunk.to_dict()
        assert data["inputs"]["ref"]["__thunk_ref__"] is True
        assert data["inputs"]["value"] == 42

        restored = Thunk.from_dict(data)
        assert isinstance(restored.inputs["ref"], ThunkRef)
        assert restored.inputs["ref"].thunk_id == ref.thunk_id


class TestThunkResult:
    """Tests for ThunkResult."""

    def test_success_result(self):
        """Create a successful result."""
        thunk_id = uuid4()
        result = ThunkResult(
            thunk_id=thunk_id,
            status=ThunkStatus.COMPLETED,
            value={"answer": 42},
            error=None,
            forced_at=datetime.now(UTC),
            duration_ms=100,
        )
        assert result.is_success is True
        assert result.is_failure is False
        assert result.value == {"answer": 42}

    def test_failure_result(self):
        """Create a failed result."""
        error = ThunkError(error_type="Error", message="failed")
        result = ThunkResult(
            thunk_id=uuid4(),
            status=ThunkStatus.FAILED,
            value=None,
            error=error,
            forced_at=datetime.now(UTC),
            duration_ms=50,
        )
        assert result.is_success is False
        assert result.is_failure is True
        assert result.error == error

    def test_serialization_roundtrip(self):
        """ThunkResult serializes and deserializes correctly."""
        original = ThunkResult(
            thunk_id=uuid4(),
            status=ThunkStatus.COMPLETED,
            value={"data": [1, 2, 3]},
            error=None,
            forced_at=datetime.now(UTC),
            duration_ms=200,
        )
        data = original.to_dict()
        restored = ThunkResult.from_dict(data)
        assert restored == original


class TestThunkStatus:
    """Tests for ThunkStatus enum."""

    def test_status_values(self):
        """ThunkStatus has expected values."""
        assert ThunkStatus.PENDING.value == "pending"
        assert ThunkStatus.RUNNING.value == "running"
        assert ThunkStatus.COMPLETED.value == "completed"
        assert ThunkStatus.FAILED.value == "failed"
