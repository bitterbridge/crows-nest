"""Tests for persistence backends."""

from datetime import UTC, datetime

import pytest

from crows_nest.core import Thunk, ThunkError, ThunkResult, ThunkStatus
from crows_nest.persistence import SQLiteBackend


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    async def backend(self):
        """Create an in-memory SQLite backend."""
        backend = await SQLiteBackend.create(":memory:")
        yield backend
        await backend.close()

    async def test_save_and_get_thunk(self, backend: SQLiteBackend):
        """Save and retrieve a thunk."""
        thunk = Thunk.create(
            "test.operation",
            {"key": "value", "number": 42},
            capabilities=frozenset({"cap1", "cap2"}),
            created_by="test",
            tags={"env": "test"},
        )

        await backend.save_thunk(thunk)
        loaded = await backend.get_thunk(thunk.id)

        assert loaded is not None
        assert loaded.id == thunk.id
        assert loaded.operation == thunk.operation
        assert loaded.inputs == thunk.inputs
        assert loaded.metadata.capabilities == thunk.metadata.capabilities

    async def test_get_nonexistent_thunk(self, backend: SQLiteBackend):
        """Getting nonexistent thunk returns None."""
        thunk = Thunk.create("test.op", {})
        loaded = await backend.get_thunk(thunk.id)
        assert loaded is None

    async def test_save_and_get_result(self, backend: SQLiteBackend):
        """Save and retrieve a result."""
        thunk = Thunk.create("test.op", {})
        await backend.save_thunk(thunk)

        result = ThunkResult(
            thunk_id=thunk.id,
            status=ThunkStatus.COMPLETED,
            value={"answer": 42},
            error=None,
            forced_at=datetime.now(UTC),
            duration_ms=100,
        )

        await backend.save_result(result)
        loaded = await backend.get_result(thunk.id)

        assert loaded is not None
        assert loaded.thunk_id == thunk.id
        assert loaded.status == ThunkStatus.COMPLETED
        assert loaded.value == {"answer": 42}
        assert loaded.duration_ms == 100

    async def test_save_failed_result(self, backend: SQLiteBackend):
        """Save and retrieve a failed result with error."""
        thunk = Thunk.create("test.op", {})
        await backend.save_thunk(thunk)

        error = ThunkError(
            error_type="ValueError",
            message="test error",
            traceback="Traceback...",
            context={"key": "value"},
        )
        result = ThunkResult(
            thunk_id=thunk.id,
            status=ThunkStatus.FAILED,
            value=None,
            error=error,
            forced_at=datetime.now(UTC),
            duration_ms=50,
        )

        await backend.save_result(result)
        loaded = await backend.get_result(thunk.id)

        assert loaded is not None
        assert loaded.is_failure
        assert loaded.error is not None
        assert loaded.error.error_type == "ValueError"
        assert loaded.error.message == "test error"

    async def test_get_nonexistent_result(self, backend: SQLiteBackend):
        """Getting nonexistent result returns None."""
        thunk = Thunk.create("test.op", {})
        loaded = await backend.get_result(thunk.id)
        assert loaded is None

    async def test_list_thunks(self, backend: SQLiteBackend):
        """List thunks with filtering."""
        thunk1 = Thunk.create("op.a", {})
        thunk2 = Thunk.create("op.b", {})
        thunk3 = Thunk.create("op.a", {})

        await backend.save_thunk(thunk1)
        await backend.save_thunk(thunk2)
        await backend.save_thunk(thunk3)

        # List all
        all_thunks = await backend.list_thunks()
        assert len(all_thunks) == 3

        # Filter by operation
        filtered = await backend.list_thunks(operation="op.a")
        assert len(filtered) == 2
        assert all(t.operation == "op.a" for t in filtered)

    async def test_list_thunks_by_status(self, backend: SQLiteBackend):
        """List thunks filtered by result status."""
        thunk1 = Thunk.create("op", {})
        thunk2 = Thunk.create("op", {})
        thunk3 = Thunk.create("op", {})

        await backend.save_thunk(thunk1)
        await backend.save_thunk(thunk2)
        await backend.save_thunk(thunk3)

        # Add results
        await backend.save_result(
            ThunkResult(
                thunk_id=thunk1.id,
                status=ThunkStatus.COMPLETED,
                value=1,
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=10,
            )
        )
        await backend.save_result(
            ThunkResult(
                thunk_id=thunk2.id,
                status=ThunkStatus.FAILED,
                value=None,
                error=ThunkError("Error", "msg"),
                forced_at=datetime.now(UTC),
                duration_ms=10,
            )
        )

        completed = await backend.list_thunks(status=ThunkStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == thunk1.id

        failed = await backend.list_thunks(status=ThunkStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].id == thunk2.id

    async def test_list_thunks_pagination(self, backend: SQLiteBackend):
        """List thunks with limit and offset."""
        # Create 5 thunks
        for i in range(5):
            await backend.save_thunk(Thunk.create(f"op.{i}", {}))

        page1 = await backend.list_thunks(limit=2, offset=0)
        page2 = await backend.list_thunks(limit=2, offset=2)
        page3 = await backend.list_thunks(limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

    async def test_delete_thunk(self, backend: SQLiteBackend):
        """Delete a thunk and its result."""
        thunk = Thunk.create("test.op", {})
        await backend.save_thunk(thunk)
        await backend.save_result(
            ThunkResult(
                thunk_id=thunk.id,
                status=ThunkStatus.COMPLETED,
                value=42,
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=10,
            )
        )

        # Verify exists
        assert await backend.get_thunk(thunk.id) is not None
        assert await backend.get_result(thunk.id) is not None

        # Delete
        deleted = await backend.delete_thunk(thunk.id)
        assert deleted is True

        # Verify gone
        assert await backend.get_thunk(thunk.id) is None
        assert await backend.get_result(thunk.id) is None

    async def test_delete_nonexistent(self, backend: SQLiteBackend):
        """Deleting nonexistent thunk returns False."""
        thunk = Thunk.create("test.op", {})
        deleted = await backend.delete_thunk(thunk.id)
        assert deleted is False

    async def test_update_thunk(self, backend: SQLiteBackend):
        """Saving thunk with same ID updates it."""
        thunk_id = Thunk.create("test.op", {"v": 1}).id

        # Create and save initial thunk
        thunk1 = Thunk(
            id=thunk_id,
            operation="test.op",
            inputs={"v": 1},
            metadata=Thunk.create("test.op", {}).metadata,
        )
        await backend.save_thunk(thunk1)

        # Save updated version
        thunk2 = Thunk(
            id=thunk_id,
            operation="test.op",
            inputs={"v": 2},  # Changed
            metadata=thunk1.metadata,
        )
        await backend.save_thunk(thunk2)

        # Should have updated value
        loaded = await backend.get_thunk(thunk_id)
        assert loaded is not None
        assert loaded.inputs["v"] == 2

    async def test_context_manager(self):
        """Backend can be used as async context manager."""
        async with await SQLiteBackend.create(":memory:") as backend:
            thunk = Thunk.create("test.op", {})
            await backend.save_thunk(thunk)
            loaded = await backend.get_thunk(thunk.id)
            assert loaded is not None
        # Backend should be closed after context

    async def test_thunk_with_nested_inputs(self, backend: SQLiteBackend):
        """Thunks with deeply nested inputs persist correctly."""
        thunk = Thunk.create(
            "complex.op",
            {
                "nested": {
                    "level1": {
                        "level2": {"value": 42},
                    },
                    "list": [1, 2, {"inner": "data"}],
                },
                "simple": "string",
            },
        )

        await backend.save_thunk(thunk)
        loaded = await backend.get_thunk(thunk.id)

        assert loaded is not None
        assert loaded.inputs["nested"]["level1"]["level2"]["value"] == 42
        assert loaded.inputs["nested"]["list"][2]["inner"] == "data"
