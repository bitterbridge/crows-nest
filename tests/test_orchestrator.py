"""Tests for orchestrator and planning operations."""

from __future__ import annotations

import pytest

from crows_nest.orchestrator.planner import (
    ExecutionPlan,
    ExecutionStep,
    PlannerConfig,
    create_plan,
    orchestrate_plan,
)


class TestExecutionStep:
    """Tests for ExecutionStep dataclass."""

    def test_to_dict(self) -> None:
        """Steps serialize to dict."""
        step = ExecutionStep(
            operation="shell.run",
            inputs={"command": "ls", "args": ["-la"]},
            description="List files",
            depends_on=[0],
        )
        d = step.to_dict()
        assert d["operation"] == "shell.run"
        assert d["inputs"]["command"] == "ls"
        assert d["description"] == "List files"
        assert d["depends_on"] == [0]

    def test_from_dict(self) -> None:
        """Steps deserialize from dict."""
        data = {
            "operation": "shell.run",
            "inputs": {"command": "echo", "args": ["test"]},
            "description": "Echo test",
        }
        step = ExecutionStep.from_dict(data)
        assert step.operation == "shell.run"
        assert step.inputs["command"] == "echo"


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_to_dict(self) -> None:
        """Plans serialize to dict."""
        plan = ExecutionPlan(
            task_description="Test task",
            steps=[
                ExecutionStep(
                    operation="shell.run",
                    inputs={"command": "ls"},
                    description="List files",
                ),
            ],
            summary="Execute test",
        )
        d = plan.to_dict()
        assert d["task_description"] == "Test task"
        assert len(d["steps"]) == 1
        assert d["summary"] == "Execute test"

    def test_to_thunks_single_step(self) -> None:
        """Single-step plans convert to thunks."""
        plan = ExecutionPlan(
            task_description="Test",
            steps=[
                ExecutionStep(
                    operation="shell.run",
                    inputs={"command": "ls", "args": []},
                    description="List files",
                ),
            ],
            summary="Test plan",
        )
        caps = frozenset({"shell.run"})
        thunks = plan.to_thunks(caps)

        assert len(thunks) == 1
        assert thunks[0].operation == "shell.run"
        assert thunks[0].inputs["command"] == "ls"
        assert thunks[0].metadata.capabilities == caps

    def test_to_thunks_with_dependencies(self) -> None:
        """Plans with dependencies include ThunkRefs."""
        plan = ExecutionPlan(
            task_description="Count files",
            steps=[
                ExecutionStep(
                    operation="shell.run",
                    inputs={"command": "ls", "args": ["-1"]},
                    description="List files",
                ),
                ExecutionStep(
                    operation="verify.output",
                    inputs={"schema": {"type": "list"}, "value": None},
                    description="Verify output",
                    depends_on=[0],
                ),
            ],
            summary="Test plan",
        )
        caps = frozenset({"shell.run"})
        thunks = plan.to_thunks(caps)

        assert len(thunks) == 2
        # Second thunk should have a reference to first
        second_inputs = thunks[1].inputs
        assert "value" in second_inputs
        # ThunkRef is now stored as an object, not a dict
        from crows_nest.core.thunk import ThunkRef

        assert isinstance(second_inputs["value"], ThunkRef)
        assert second_inputs["value"].thunk_id == thunks[0].id


class TestPatternMatching:
    """Tests for task pattern matching."""

    def test_list_files_pattern(self) -> None:
        """'list files' matches the list files pattern."""
        plan = create_plan("list files", frozenset())
        assert len(plan.steps) == 1
        assert plan.steps[0].inputs["command"] == "ls"

    def test_count_files_pattern(self) -> None:
        """'count files' matches the count files pattern."""
        plan = create_plan("count files", frozenset())
        assert len(plan.steps) == 2
        # First step is a pipe
        assert plan.steps[0].operation == "shell.pipe"
        # Second step is verification
        assert plan.steps[1].operation == "verify.output"

    def test_list_python_files_pattern(self) -> None:
        """'list python files' matches the pattern."""
        plan = create_plan("list python files", frozenset())
        assert len(plan.steps) == 1
        assert plan.steps[0].inputs["command"] == "find"
        assert "*.py" in plan.steps[0].inputs["args"]

    def test_count_python_files_pattern(self) -> None:
        """'count python files' matches the pattern."""
        plan = create_plan("count python files", frozenset())
        assert plan.steps[0].operation == "shell.pipe"

    def test_show_date_pattern(self) -> None:
        """'show date' matches the date pattern."""
        plan = create_plan("show date", frozenset())
        assert plan.steps[0].inputs["command"] == "date"

    def test_current_directory_pattern(self) -> None:
        """'show current directory' matches the pwd pattern."""
        plan = create_plan("show current directory", frozenset())
        assert plan.steps[0].inputs["command"] == "pwd"

    def test_whoami_pattern(self) -> None:
        """'whoami' matches the user pattern."""
        plan = create_plan("whoami", frozenset())
        assert plan.steps[0].inputs["command"] == "whoami"

    def test_system_info_pattern(self) -> None:
        """'system info' matches the uname pattern."""
        plan = create_plan("system info", frozenset())
        assert plan.steps[0].inputs["command"] == "uname"


class TestSimpleCommandParsing:
    """Tests for simple command parsing."""

    def test_run_allowed_command(self) -> None:
        """'run ls' parses to ls command."""
        plan = create_plan("run ls", frozenset())
        assert plan.steps[0].inputs["command"] == "ls"

    def test_run_with_args(self) -> None:
        """'run ls -la' includes arguments."""
        plan = create_plan("run ls -la", frozenset())
        assert plan.steps[0].inputs["command"] == "ls"
        assert "-la" in plan.steps[0].inputs["args"]

    def test_execute_prefix(self) -> None:
        """'execute pwd' parses correctly."""
        plan = create_plan("execute pwd", frozenset())
        assert plan.steps[0].inputs["command"] == "pwd"

    def test_run_disallowed_rejected(self) -> None:
        """'run rm' fails for disallowed commands."""
        with pytest.raises(ValueError, match="Cannot create execution plan"):
            create_plan("run rm -rf", frozenset())


class TestUnmatchedTasks:
    """Tests for tasks that don't match patterns."""

    def test_complex_task_fails(self) -> None:
        """Complex natural language tasks fail without LLM."""
        with pytest.raises(ValueError, match="Cannot create execution plan"):
            create_plan("analyze the codebase and find security issues", frozenset())

    def test_random_text_fails(self) -> None:
        """Random text fails."""
        with pytest.raises(ValueError, match="Cannot create execution plan"):
            create_plan("hello world how are you", frozenset())


class TestOrchestrateOperation:
    """Tests for the orchestrate.plan thunk operation."""

    async def test_plan_simple_task(self) -> None:
        """orchestrate.plan creates a plan for simple tasks."""
        result = await orchestrate_plan(
            task_description="list files",
            available_capabilities=["shell.run"],
        )
        assert result["task_description"] == "list files"
        assert len(result["steps"]) >= 1
        assert "summary" in result
        assert "thunk_ids" in result

    async def test_plan_count_files(self) -> None:
        """orchestrate.plan creates multi-step plan for count files."""
        result = await orchestrate_plan(
            task_description="count files",
            available_capabilities=["shell.run", "shell.pipe"],
        )
        assert len(result["steps"]) == 2
        assert result["steps"][1]["operation"] == "verify.output"

    async def test_plan_unknown_task(self) -> None:
        """orchestrate.plan fails for unknown tasks."""
        with pytest.raises(ValueError, match="Cannot create execution plan"):
            await orchestrate_plan(
                task_description="do something complicated",
                available_capabilities=[],
            )


class TestPlannerConfig:
    """Tests for planner configuration."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = PlannerConfig()
        assert config.max_steps == 10
        assert "shell.run" in config.allowed_operations

    def test_custom_config(self) -> None:
        """Custom config values are respected."""
        config = PlannerConfig(max_steps=5)
        assert config.max_steps == 5
