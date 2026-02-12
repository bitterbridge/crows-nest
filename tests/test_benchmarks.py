"""Performance benchmark tests.

Run benchmarks with:
    uv run pytest tests/test_benchmarks.py -v --benchmark-only
    uv run pytest tests/test_benchmarks.py -v --benchmark-compare
    uv run pytest tests/test_benchmarks.py -v --benchmark-autosave

Skip benchmarks in regular test runs:
    uv run pytest -m "not benchmark"
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from crows_nest.core.registry import ThunkRegistry
from crows_nest.core.runtime import ThunkRuntime, _topological_sort
from crows_nest.core.thunk import Thunk, ThunkRef, ThunkResult, ThunkStatus

# Mark all tests in this module as benchmarks
pytestmark = pytest.mark.benchmark


class TestThunkCreationBenchmarks:
    """Benchmarks for thunk creation performance."""

    def test_thunk_create_simple(self, benchmark) -> None:
        """Benchmark creating a simple thunk with minimal inputs."""

        def create_thunk():
            return Thunk.create(
                "test.operation",
                {"arg": "value"},
                capabilities=frozenset({"test.cap"}),
            )

        result = benchmark(create_thunk)
        assert result.id is not None
        assert result.operation == "test.operation"

    def test_thunk_create_complex_inputs(self, benchmark) -> None:
        """Benchmark creating a thunk with complex nested inputs."""

        def create_thunk():
            return Thunk.create(
                "test.operation",
                {
                    "config": {
                        "nested": {"deeply": {"value": 42}},
                        "list": [1, 2, 3, 4, 5],
                    },
                    "flags": ["a", "b", "c"],
                    "metadata": {"key1": "val1", "key2": "val2"},
                },
                capabilities=frozenset({"cap1", "cap2", "cap3"}),
                tags={"env": "test", "version": "1.0"},
            )

        result = benchmark(create_thunk)
        assert result.id is not None

    def test_thunk_create_with_dependencies(self, benchmark) -> None:
        """Benchmark creating a thunk with dependency references."""
        dep_ids = [uuid4() for _ in range(5)]

        def create_thunk():
            return Thunk.create(
                "test.aggregate",
                {
                    "inputs": [ThunkRef(dep_id) for dep_id in dep_ids],
                    "config": {"merge_strategy": "concat"},
                },
                capabilities=frozenset({"test.aggregate"}),
            )

        result = benchmark(create_thunk)
        assert len(result.get_dependencies()) == 5


class TestThunkSerializationBenchmarks:
    """Benchmarks for thunk serialization/deserialization."""

    @pytest.fixture
    def sample_thunk(self) -> Thunk:
        """Create a sample thunk for serialization tests."""
        return Thunk.create(
            "test.operation",
            {
                "config": {"nested": {"value": 42}, "list": list(range(10))},
                "flags": ["a", "b", "c"],
            },
            capabilities=frozenset({"cap1", "cap2"}),
            tags={"env": "test"},
        )

    def test_thunk_to_dict(self, benchmark, sample_thunk: Thunk) -> None:
        """Benchmark thunk to dict serialization."""
        result = benchmark(sample_thunk.to_dict)
        assert "id" in result
        assert "operation" in result

    def test_thunk_to_json(self, benchmark, sample_thunk: Thunk) -> None:
        """Benchmark thunk to JSON string serialization."""
        result = benchmark(sample_thunk.to_json)
        assert isinstance(result, str)

    def test_thunk_from_dict(self, benchmark, sample_thunk: Thunk) -> None:
        """Benchmark thunk from dict deserialization."""
        data = sample_thunk.to_dict()
        result = benchmark(lambda: Thunk.from_dict(data))
        assert result.id == sample_thunk.id

    def test_thunk_from_json(self, benchmark, sample_thunk: Thunk) -> None:
        """Benchmark thunk from JSON string deserialization."""
        json_str = sample_thunk.to_json()
        result = benchmark(lambda: Thunk.from_json(json_str))
        assert result.id == sample_thunk.id

    def test_thunk_roundtrip(self, benchmark, sample_thunk: Thunk) -> None:
        """Benchmark full serialize/deserialize roundtrip."""

        def roundtrip():
            json_str = sample_thunk.to_json()
            return Thunk.from_json(json_str)

        result = benchmark(roundtrip)
        assert result.id == sample_thunk.id


class TestDependencyResolutionBenchmarks:
    """Benchmarks for dependency graph resolution."""

    def _create_linear_chain(self, n: int) -> dict[UUID, Thunk]:
        """Create a linear chain of dependent thunks."""
        thunks: dict[UUID, Thunk] = {}
        prev_id: UUID | None = None

        for i in range(n):
            inputs: dict = {"step": i}
            if prev_id is not None:
                inputs["dep"] = ThunkRef(prev_id)

            thunk = Thunk.create(
                f"step.{i}",
                inputs,
                capabilities=frozenset({f"step.{i}"}),
            )
            thunks[thunk.id] = thunk
            prev_id = thunk.id

        return thunks

    def _create_diamond(self, width: int = 10) -> dict[UUID, Thunk]:
        """Create a diamond dependency pattern (1 -> many -> 1)."""
        thunks: dict[UUID, Thunk] = {}

        # Root thunk
        root = Thunk.create("diamond.root", {"init": True}, capabilities=frozenset({"test"}))
        thunks[root.id] = root

        # Middle layer
        middle_ids: list[UUID] = []
        for i in range(width):
            t = Thunk.create(
                f"diamond.middle.{i}",
                {"dep": ThunkRef(root.id), "idx": i},
                capabilities=frozenset({"test"}),
            )
            thunks[t.id] = t
            middle_ids.append(t.id)

        # Final aggregator
        final = Thunk.create(
            "diamond.final",
            {"deps": [ThunkRef(mid) for mid in middle_ids]},
            capabilities=frozenset({"test"}),
        )
        thunks[final.id] = final

        return thunks

    def test_topological_sort_linear_10(self, benchmark) -> None:
        """Benchmark topological sort of 10 linear thunks."""
        thunks = self._create_linear_chain(10)
        result = benchmark(lambda: _topological_sort(thunks))
        assert len(result) == 10

    def test_topological_sort_linear_100(self, benchmark) -> None:
        """Benchmark topological sort of 100 linear thunks."""
        thunks = self._create_linear_chain(100)
        result = benchmark(lambda: _topological_sort(thunks))
        assert len(result) == 100

    def test_topological_sort_diamond_10(self, benchmark) -> None:
        """Benchmark topological sort of diamond with 10 middle nodes."""
        thunks = self._create_diamond(10)
        result = benchmark(lambda: _topological_sort(thunks))
        assert len(result) == 12  # 1 root + 10 middle + 1 final

    def test_topological_sort_diamond_50(self, benchmark) -> None:
        """Benchmark topological sort of diamond with 50 middle nodes."""
        thunks = self._create_diamond(50)
        result = benchmark(lambda: _topological_sort(thunks))
        assert len(result) == 52

    def test_get_dependencies(self, benchmark) -> None:
        """Benchmark extracting dependencies from a thunk."""
        refs = [ThunkRef(uuid4()) for _ in range(20)]
        thunk = Thunk.create(
            "test.multi_dep",
            {
                "direct": refs[:5],
                "nested": {"inner": refs[5:10]},
                "deep": {"level1": {"level2": refs[10:15]}},
                "mixed": refs[15:],
            },
            capabilities=frozenset({"test"}),
        )

        result = benchmark(thunk.get_dependencies)
        assert len(result) == 20


class TestRuntimeBenchmarks:
    """Benchmarks for runtime execution overhead."""

    @pytest.fixture
    def registry(self) -> ThunkRegistry:
        """Create a registry with test operations."""
        reg = ThunkRegistry()

        @reg.operation("test.noop", required_capabilities=frozenset({"test.noop"}))
        async def noop() -> str:
            return "done"

        @reg.operation("test.add", required_capabilities=frozenset({"test.add"}))
        async def add(a: int, b: int) -> int:
            return a + b

        @reg.operation("test.passthrough", required_capabilities=frozenset({"test.passthrough"}))
        async def passthrough(value: str) -> str:
            return value

        return reg

    @pytest.fixture
    def runtime(self, registry: ThunkRegistry) -> ThunkRuntime:
        """Create a runtime for testing."""
        return ThunkRuntime(registry=registry)

    @pytest.fixture
    def event_loop(self):
        """Create a dedicated event loop for runtime benchmarks."""
        import asyncio

        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_runtime_force_noop(self, benchmark, runtime: ThunkRuntime, event_loop) -> None:
        """Benchmark forcing a no-op thunk (measures pure overhead)."""
        thunk = Thunk.create(
            "test.noop",
            {},
            capabilities=frozenset({"test.noop"}),
        )

        def force():
            runtime.clear_cache()
            return event_loop.run_until_complete(runtime.force(thunk))

        result = benchmark(force)
        assert result.is_success

    def test_runtime_force_simple_compute(
        self, benchmark, runtime: ThunkRuntime, event_loop
    ) -> None:
        """Benchmark forcing a simple computation thunk."""
        thunk = Thunk.create(
            "test.add",
            {"a": 1, "b": 2},
            capabilities=frozenset({"test.add"}),
        )

        def force():
            runtime.clear_cache()
            return event_loop.run_until_complete(runtime.force(thunk))

        result = benchmark(force)
        assert result.is_success
        assert result.value == 3

    def test_runtime_force_with_dependency(
        self, benchmark, runtime: ThunkRuntime, event_loop
    ) -> None:
        """Benchmark forcing a thunk that depends on another."""
        dep = Thunk.create(
            "test.passthrough",
            {"value": "input"},
            capabilities=frozenset({"test.passthrough"}),
        )
        main = Thunk.create(
            "test.passthrough",
            {"value": ThunkRef(dep.id)},
            capabilities=frozenset({"test.passthrough"}),
        )

        def force():
            runtime.clear_cache()
            return event_loop.run_until_complete(
                runtime.force_all({dep.id: dep, main.id: main}, target=main.id)
            )

        result = benchmark(force)
        assert result.is_success


class TestDatabaseBenchmarks:
    """Benchmarks for database operations."""

    @pytest.fixture
    def backend(self):
        """Create an in-memory SQLite backend."""
        import asyncio

        from crows_nest.persistence.sqlite import SQLiteBackend

        loop = asyncio.new_event_loop()
        backend = loop.run_until_complete(SQLiteBackend.create(":memory:"))
        yield backend, loop
        loop.run_until_complete(backend.close())
        loop.close()

    def test_db_save_thunk(self, benchmark, backend) -> None:
        """Benchmark saving a thunk to the database."""
        db, loop = backend
        thunk = Thunk.create(
            "test.operation",
            {"config": {"nested": {"value": 42}}, "list": list(range(10))},
            capabilities=frozenset({"test.cap"}),
        )

        def save():
            loop.run_until_complete(db.save_thunk(thunk))

        benchmark(save)

    def test_db_get_thunk(self, benchmark, backend) -> None:
        """Benchmark retrieving a thunk from the database."""
        db, loop = backend
        thunk = Thunk.create(
            "test.operation",
            {"config": {"value": 42}},
            capabilities=frozenset({"test.cap"}),
        )
        loop.run_until_complete(db.save_thunk(thunk))

        def get():
            return loop.run_until_complete(db.get_thunk(thunk.id))

        result = benchmark(get)
        assert result is not None
        assert result.id == thunk.id

    def test_db_save_result(self, benchmark, backend) -> None:
        """Benchmark saving a result to the database."""
        from datetime import UTC, datetime

        db, loop = backend
        thunk = Thunk.create("test.op", {}, capabilities=frozenset({"test"}))
        loop.run_until_complete(db.save_thunk(thunk))

        result = ThunkResult(
            thunk_id=thunk.id,
            status=ThunkStatus.COMPLETED,
            value={"output": "test result"},
            error=None,
            forced_at=datetime.now(UTC),
            duration_ms=42,
        )

        def save():
            loop.run_until_complete(db.save_result(result))

        benchmark(save)

    def test_db_list_thunks_100(self, benchmark, backend) -> None:
        """Benchmark listing 100 thunks from the database."""
        db, loop = backend

        # Create 100 thunks
        for i in range(100):
            thunk = Thunk.create(
                f"test.op.{i}",
                {"idx": i},
                capabilities=frozenset({"test"}),
            )
            loop.run_until_complete(db.save_thunk(thunk))

        def list_all():
            return loop.run_until_complete(db.list_thunks(limit=100))

        result = benchmark(list_all)
        assert len(result) == 100

    def test_db_roundtrip(self, benchmark, backend) -> None:
        """Benchmark full save/load roundtrip."""
        from datetime import UTC, datetime

        db, loop = backend

        def roundtrip():
            thunk = Thunk.create(
                "test.roundtrip",
                {"data": list(range(10))},
                capabilities=frozenset({"test"}),
            )
            loop.run_until_complete(db.save_thunk(thunk))

            result = ThunkResult(
                thunk_id=thunk.id,
                status=ThunkStatus.COMPLETED,
                value={"output": 42},
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=10,
            )
            loop.run_until_complete(db.save_result(result))

            loaded_thunk = loop.run_until_complete(db.get_thunk(thunk.id))
            loaded_result = loop.run_until_complete(db.get_result(thunk.id))
            return loaded_thunk, loaded_result

        thunk, result = benchmark(roundtrip)
        assert thunk is not None
        assert result is not None

    def test_db_batch_save_10_thunks(self, benchmark, backend) -> None:
        """Benchmark batch saving 10 thunks."""
        db, loop = backend
        thunks = [
            Thunk.create(f"batch.op.{i}", {"idx": i}, capabilities=frozenset({"test"}))
            for i in range(10)
        ]

        def batch_save():
            loop.run_until_complete(db.save_thunks_batch(thunks))

        benchmark(batch_save)

    def test_db_count_thunks(self, benchmark, backend) -> None:
        """Benchmark counting thunks."""
        db, loop = backend

        # Create some thunks first
        thunks = [Thunk.create("count.op", {}, capabilities=frozenset({"test"})) for _ in range(50)]
        loop.run_until_complete(db.save_thunks_batch(thunks))

        def count():
            return loop.run_until_complete(db.count_thunks())

        result = benchmark(count)
        assert result == 50

    def test_db_get_thunks_by_ids_10(self, benchmark, backend) -> None:
        """Benchmark batch fetching 10 thunks by ID."""
        db, loop = backend

        # Create thunks first
        thunks = [
            Thunk.create(f"multi.op.{i}", {"idx": i}, capabilities=frozenset({"test"}))
            for i in range(20)
        ]
        loop.run_until_complete(db.save_thunks_batch(thunks))
        ids_to_get = [t.id for t in thunks[:10]]

        def multi_get():
            return loop.run_until_complete(db.get_thunks_by_ids(ids_to_get))

        result = benchmark(multi_get)
        assert len(result) == 10


class TestThunkResultBenchmarks:
    """Benchmarks for ThunkResult operations."""

    @pytest.fixture
    def sample_result(self) -> ThunkResult:
        """Create a sample result for benchmarking."""
        from datetime import UTC, datetime

        return ThunkResult(
            thunk_id=uuid4(),
            status=ThunkStatus.COMPLETED,
            value={"data": list(range(100)), "nested": {"key": "value"}},
            error=None,
            forced_at=datetime.now(UTC),
            duration_ms=42,
        )

    def test_result_to_dict(self, benchmark, sample_result: ThunkResult) -> None:
        """Benchmark result serialization."""
        result = benchmark(sample_result.to_dict)
        assert "thunk_id" in result

    def test_result_from_dict(self, benchmark, sample_result: ThunkResult) -> None:
        """Benchmark result deserialization."""
        data = sample_result.to_dict()
        result = benchmark(lambda: ThunkResult.from_dict(data))
        assert result.thunk_id == sample_result.thunk_id
