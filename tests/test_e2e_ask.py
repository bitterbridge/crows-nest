"""End-to-end integration tests for the ask functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from crows_nest.core import ThunkRuntime, get_global_registry
from crows_nest.orchestrator.executor import TaskExecutor, TaskPlanner, ask
from crows_nest.persistence.sqlite import SQLiteBackend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
async def backend() -> AsyncGenerator[SQLiteBackend, None]:
    """Create a temporary SQLite backend for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        backend = await SQLiteBackend.create(str(db_path))
        try:
            yield backend
        finally:
            await backend.close()


@pytest.fixture
def runtime(backend: SQLiteBackend) -> ThunkRuntime:
    """Create a runtime with the test backend."""
    return ThunkRuntime(
        registry=get_global_registry(),
        persistence=backend,
    )


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    def test_plan_simple_task(self) -> None:
        """Planner creates plan for simple task."""
        planner = TaskPlanner()
        plan = planner.plan("show current directory", frozenset())

        assert plan.task_description == "show current directory"
        assert len(plan.steps) == 1
        assert plan.steps[0].operation == "shell.run"
        assert plan.steps[0].inputs["command"] == "pwd"

    def test_plan_count_files(self) -> None:
        """Planner creates multi-step plan for count task."""
        planner = TaskPlanner()
        plan = planner.plan("count python files", frozenset())

        assert len(plan.steps) == 2
        assert plan.steps[0].operation == "shell.pipe"
        assert plan.steps[1].operation == "verify.output"
        assert plan.steps[1].depends_on == [0]

    def test_plan_unknown_task_raises(self) -> None:
        """Planner raises for unknown tasks without LLM."""
        planner = TaskPlanner()

        with pytest.raises(ValueError, match="Cannot create execution plan"):
            planner.plan("do something unknown", frozenset())


class TestTaskExecutor:
    """Tests for TaskExecutor."""

    async def test_execute_simple_task(self, backend: SQLiteBackend, runtime: ThunkRuntime) -> None:
        """Executor runs simple task successfully."""
        planner = TaskPlanner()
        executor = TaskExecutor(runtime=runtime, persistence=backend)

        plan = planner.plan("show current directory", frozenset())
        capabilities = frozenset({"shell.run", "shell.pipe", "verify.output"})

        result = await executor.execute(plan, capabilities)

        assert result.success
        assert result.steps_completed == 1
        assert result.total_steps == 1
        assert result.output  # Should have some output
        assert len(result.thunk_ids) == 1

    async def test_execute_with_verification(
        self, backend: SQLiteBackend, runtime: ThunkRuntime
    ) -> None:
        """Executor runs task with verification step."""
        planner = TaskPlanner()
        executor = TaskExecutor(runtime=runtime, persistence=backend)

        plan = planner.plan("count files", frozenset())
        capabilities = frozenset({"shell.run", "shell.pipe", "verify.output"})

        result = await executor.execute(plan, capabilities)

        assert result.success
        assert result.steps_completed == 2
        assert result.total_steps == 2
        # Output should be the count (an integer)
        assert result.output.isdigit() or result.output.strip().isdigit()

    async def test_execute_with_progress(
        self, backend: SQLiteBackend, runtime: ThunkRuntime
    ) -> None:
        """Executor yields progress updates."""
        planner = TaskPlanner()
        executor = TaskExecutor(runtime=runtime, persistence=backend)

        plan = planner.plan("show date", frozenset())
        capabilities = frozenset({"shell.run", "shell.pipe", "verify.output"})

        progress_received = []
        async for item in executor.execute_with_progress(plan, capabilities):
            progress_received.append(item)

        # Should have at least one progress update and final result
        assert len(progress_received) >= 2
        # Last item should be the result
        from crows_nest.orchestrator.executor import ExecutionResult

        assert isinstance(progress_received[-1], ExecutionResult)


class TestAskFunction:
    """Tests for the ask() function."""

    async def test_ask_simple(self, backend: SQLiteBackend, runtime: ThunkRuntime) -> None:
        """ask() executes simple task."""
        result = await ask(
            query="show current directory",
            runtime=runtime,
            persistence=backend,
        )

        assert result.success
        assert result.query == "show current directory"
        assert result.output
        assert "/" in result.output  # Should contain a path

    async def test_ask_list_files(self, backend: SQLiteBackend, runtime: ThunkRuntime) -> None:
        """ask() executes list files task."""
        result = await ask(
            query="list files",
            runtime=runtime,
            persistence=backend,
        )

        assert result.success
        assert result.output
        # Should contain typical file listing
        assert "total" in result.output.lower() or "\n" in result.output

    async def test_ask_count_python_files(
        self, backend: SQLiteBackend, runtime: ThunkRuntime
    ) -> None:
        """ask() executes count python files task."""
        result = await ask(
            query="count python files",
            runtime=runtime,
            persistence=backend,
        )

        assert result.success
        assert result.output.strip().isdigit()
        count = int(result.output.strip())
        assert count >= 0  # Should find some python files

    async def test_ask_unknown_task(self, backend: SQLiteBackend, runtime: ThunkRuntime) -> None:
        """ask() returns error for unknown task."""
        result = await ask(
            query="do something impossible",
            runtime=runtime,
            persistence=backend,
        )

        assert not result.success
        assert result.error is not None
        assert "Cannot create execution plan" in result.error

    async def test_ask_with_progress(self, backend: SQLiteBackend, runtime: ThunkRuntime) -> None:
        """ask() calls progress callback."""
        progress_calls = []

        def on_progress(p):
            progress_calls.append(p)

        result = await ask(
            query="whoami",
            runtime=runtime,
            persistence=backend,
            on_progress=on_progress,
        )

        assert result.success
        assert len(progress_calls) >= 1
        # Should have running and success for each step
        assert any(p.status == "running" for p in progress_calls)
        assert any(p.status == "success" for p in progress_calls)
