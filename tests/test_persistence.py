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

    async def test_save_thunks_batch(self, backend: SQLiteBackend):
        """Batch save multiple thunks efficiently."""
        thunks = [Thunk.create(f"batch.op.{i}", {"idx": i}) for i in range(10)]

        await backend.save_thunks_batch(thunks)

        # Verify all saved
        all_thunks = await backend.list_thunks(limit=20)
        assert len(all_thunks) == 10

        # Verify each one
        for thunk in thunks:
            loaded = await backend.get_thunk(thunk.id)
            assert loaded is not None
            assert loaded.operation == thunk.operation

    async def test_save_thunks_batch_empty(self, backend: SQLiteBackend):
        """Batch save with empty list does nothing."""
        await backend.save_thunks_batch([])
        all_thunks = await backend.list_thunks()
        assert len(all_thunks) == 0

    async def test_save_results_batch(self, backend: SQLiteBackend):
        """Batch save multiple results efficiently."""
        # Create thunks first
        thunks = [Thunk.create(f"batch.op.{i}", {}) for i in range(5)]
        await backend.save_thunks_batch(thunks)

        # Create results
        results = [
            ThunkResult(
                thunk_id=thunk.id,
                status=ThunkStatus.COMPLETED,
                value={"idx": i},
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=10 * i,
            )
            for i, thunk in enumerate(thunks)
        ]

        await backend.save_results_batch(results)

        # Verify all saved
        for thunk in thunks:
            result = await backend.get_result(thunk.id)
            assert result is not None
            assert result.is_success

    async def test_save_results_batch_empty(self, backend: SQLiteBackend):
        """Batch save with empty list does nothing."""
        await backend.save_results_batch([])

    async def test_count_thunks(self, backend: SQLiteBackend):
        """Count thunks with filtering."""
        # Create thunks
        thunks = [Thunk.create(f"count.op.{i % 2}", {}) for i in range(10)]
        await backend.save_thunks_batch(thunks)

        # Count all
        total = await backend.count_thunks()
        assert total == 10

        # Count by operation
        count_0 = await backend.count_thunks(operation="count.op.0")
        count_1 = await backend.count_thunks(operation="count.op.1")
        assert count_0 == 5
        assert count_1 == 5

    async def test_count_thunks_by_status(self, backend: SQLiteBackend):
        """Count thunks filtered by result status."""
        thunks = [Thunk.create("count.op", {}) for _ in range(5)]
        await backend.save_thunks_batch(thunks)

        # Add results for some
        results = [
            ThunkResult(
                thunk_id=thunks[i].id,
                status=ThunkStatus.COMPLETED if i < 3 else ThunkStatus.FAILED,
                value=i if i < 3 else None,
                error=ThunkError("Error", "msg") if i >= 3 else None,
                forced_at=datetime.now(UTC),
                duration_ms=10,
            )
            for i in range(4)  # One thunk has no result
        ]
        await backend.save_results_batch(results)

        completed = await backend.count_thunks(status=ThunkStatus.COMPLETED)
        failed = await backend.count_thunks(status=ThunkStatus.FAILED)

        assert completed == 3
        assert failed == 1

    async def test_get_thunks_by_ids(self, backend: SQLiteBackend):
        """Get multiple thunks by IDs in a single query."""
        thunks = [Thunk.create(f"multi.op.{i}", {"idx": i}) for i in range(5)]
        await backend.save_thunks_batch(thunks)

        # Get subset
        ids_to_get = [thunks[0].id, thunks[2].id, thunks[4].id]
        result = await backend.get_thunks_by_ids(ids_to_get)

        assert len(result) == 3
        assert thunks[0].id in result
        assert thunks[2].id in result
        assert thunks[4].id in result
        assert result[thunks[0].id].inputs["idx"] == 0
        assert result[thunks[2].id].inputs["idx"] == 2
        assert result[thunks[4].id].inputs["idx"] == 4

    async def test_get_thunks_by_ids_empty(self, backend: SQLiteBackend):
        """Get with empty IDs list returns empty dict."""
        result = await backend.get_thunks_by_ids([])
        assert result == {}

    async def test_get_thunks_by_ids_partial(self, backend: SQLiteBackend):
        """Get with some nonexistent IDs returns only existing."""
        thunk = Thunk.create("partial.op", {})
        await backend.save_thunk(thunk)

        fake_id = Thunk.create("fake", {}).id
        result = await backend.get_thunks_by_ids([thunk.id, fake_id])

        assert len(result) == 1
        assert thunk.id in result

    async def test_get_results_by_ids(self, backend: SQLiteBackend):
        """Get multiple results by IDs in a single query."""
        thunks = [Thunk.create(f"result.op.{i}", {}) for i in range(5)]
        await backend.save_thunks_batch(thunks)

        # Save results for some thunks
        results = [
            ThunkResult(
                thunk_id=thunks[i].id,
                status=ThunkStatus.COMPLETED,
                value={"idx": i},
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=10 * i,
            )
            for i in range(3)  # Only first 3
        ]
        await backend.save_results_batch(results)

        # Get all results
        ids_to_get = [t.id for t in thunks]
        result_map = await backend.get_results_by_ids(ids_to_get)

        # Should only have 3 results
        assert len(result_map) == 3
        assert thunks[0].id in result_map
        assert thunks[2].id in result_map
        assert thunks[4].id not in result_map  # No result saved

    async def test_get_results_by_ids_empty(self, backend: SQLiteBackend):
        """Get results with empty IDs list returns empty dict."""
        result = await backend.get_results_by_ids([])
        assert result == {}
