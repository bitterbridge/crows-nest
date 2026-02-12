"""Task execution orchestrator.

Provides end-to-end execution of natural language tasks by:
1. Planning: Converting queries to execution plans (pattern matching or LLM)
2. Execution: Running the plan's thunks in dependency order
3. Result formatting: Returning human-readable results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from crows_nest.core.thunk import ThunkStatus
from crows_nest.orchestrator.planner import (
    ExecutionPlan,
    ExecutionStep,
    PlannerConfig,
    _match_task_pattern,
    _parse_simple_command,
)
from crows_nest.shell.operations import ALLOWED_COMMANDS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from crows_nest.agents.llm import LLMProvider
    from crows_nest.core.runtime import ThunkRuntime
    from crows_nest.persistence.protocol import PersistenceBackend


class LLMPlanStep(BaseModel):
    """A step in an LLM-generated plan."""

    command: str = Field(description="Shell command to run")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    description: str = Field(description="What this step does")
    depends_on_previous: bool = Field(
        default=False,
        description="Whether this step needs output from the previous step",
    )


class LLMPlanResponse(BaseModel):
    """Response from LLM for task planning."""

    steps: list[LLMPlanStep] = Field(description="Steps to execute")
    summary: str = Field(description="Brief summary of the plan")
    can_execute: bool = Field(
        default=True,
        description="Whether this task can be executed with available commands",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Why the task cannot be executed, if applicable",
    )


@dataclass
class ExecutionProgress:
    """Progress update during execution."""

    step_index: int
    total_steps: int
    step_description: str
    status: str  # "pending", "running", "success", "failure"
    output: str | None = None
    error: str | None = None


@dataclass
class ExecutionResult:
    """Result of executing a task."""

    success: bool
    query: str
    output: str
    steps_completed: int
    total_steps: int
    duration_ms: int
    error: str | None = None
    thunk_ids: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Format result for display."""
        if self.success:
            return f"{self.output}"
        return f"Error: {self.error or 'Unknown error'}"


class TaskPlanner:
    """Plans task execution using pattern matching or LLM.

    First attempts fast pattern matching for common tasks.
    Falls back to LLM for complex or unfamiliar tasks.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            llm_provider: Optional LLM for complex task planning.
            config: Planner configuration.
        """
        self._llm = llm_provider
        self._config = config or PlannerConfig()

    def plan(
        self,
        query: str,
        capabilities: frozenset[str],
    ) -> ExecutionPlan:
        """Create an execution plan for a query.

        Args:
            query: Natural language task description.
            capabilities: Available capabilities.

        Returns:
            ExecutionPlan ready for execution.

        Raises:
            ValueError: If no plan can be created.
        """
        # Try pattern matching first (fast path)
        steps = _match_task_pattern(query)
        if steps:
            return ExecutionPlan(
                task_description=query,
                steps=steps,
                summary=f"Execute {len(steps)} step(s) for: {query}",
            )

        # Try simple command parsing
        steps = _parse_simple_command(query)
        if steps:
            return ExecutionPlan(
                task_description=query,
                steps=steps,
                summary=f"Execute command for: {query}",
            )

        # If LLM is available, use it for complex tasks
        if self._llm is not None:
            return self._plan_with_llm(query, capabilities)

        # No plan possible
        msg = (
            f"Cannot create execution plan for: {query}\n"
            "Try a simpler task like 'list files' or 'count python files'"
        )
        raise ValueError(msg)

    async def plan_async(
        self,
        query: str,
        capabilities: frozenset[str],
    ) -> ExecutionPlan:
        """Create an execution plan asynchronously.

        Args:
            query: Natural language task description.
            capabilities: Available capabilities.

        Returns:
            ExecutionPlan ready for execution.
        """
        # Try pattern matching first (fast path)
        steps = _match_task_pattern(query)
        if steps:
            return ExecutionPlan(
                task_description=query,
                steps=steps,
                summary=f"Execute {len(steps)} step(s) for: {query}",
            )

        # Try simple command parsing
        steps = _parse_simple_command(query)
        if steps:
            return ExecutionPlan(
                task_description=query,
                steps=steps,
                summary=f"Execute command for: {query}",
            )

        # If LLM is available, use it for complex tasks
        if self._llm is not None:
            return await self._plan_with_llm_async(query, capabilities)

        # No plan possible
        msg = (
            f"Cannot create execution plan for: {query}\n"
            "Try a simpler task like 'list files' or 'count python files'"
        )
        raise ValueError(msg)

    def _plan_with_llm(
        self,
        query: str,
        _capabilities: frozenset[str],
    ) -> ExecutionPlan:
        """Use LLM to create a plan for a complex task."""
        from crows_nest.agents.llm import ChatMessage

        system_prompt = self._build_planning_prompt()
        messages = [ChatMessage(role="user", content=query)]

        result = self._llm.complete(  # type: ignore[union-attr]
            messages=messages,
            output_schema=LLMPlanResponse,
            system_prompt=system_prompt,
        )

        response: LLMPlanResponse = result.content
        return self._convert_llm_plan(query, response)

    async def _plan_with_llm_async(
        self,
        query: str,
        _capabilities: frozenset[str],
    ) -> ExecutionPlan:
        """Use LLM to create a plan asynchronously."""
        from crows_nest.agents.llm import ChatMessage

        system_prompt = self._build_planning_prompt()
        messages = [ChatMessage(role="user", content=query)]

        result = await self._llm.complete_async(  # type: ignore[union-attr]
            messages=messages,
            output_schema=LLMPlanResponse,
            system_prompt=system_prompt,
        )

        response: LLMPlanResponse = result.content
        return self._convert_llm_plan(query, response)

    def _build_planning_prompt(self) -> str:
        """Build the system prompt for task planning."""
        commands_list = ", ".join(sorted(ALLOWED_COMMANDS))
        return f"""You are a task planner that converts natural language requests into plans.

Available commands: {commands_list}

Rules:
1. Only use commands from the available list
2. Break complex tasks into simple steps
3. Each step should be a single command
4. If a task cannot be done with available commands, set can_execute=false
5. Keep plans simple and focused

Examples of good plans:
- "count lines in file.txt" -> cat file.txt | wc -l
- "find large files" -> find . -type f -size +1M
- "show disk usage" -> df -h
"""

    def _convert_llm_plan(
        self,
        query: str,
        response: LLMPlanResponse,
    ) -> ExecutionPlan:
        """Convert LLM response to ExecutionPlan."""
        if not response.can_execute:
            msg = response.rejection_reason or f"Cannot execute: {query}"
            raise ValueError(msg)

        steps: list[ExecutionStep] = []
        for i, llm_step in enumerate(response.steps):
            # Validate command is allowed
            if llm_step.command not in ALLOWED_COMMANDS:
                msg = f"Command not allowed: {llm_step.command}"
                raise ValueError(msg)

            depends_on = [i - 1] if llm_step.depends_on_previous and i > 0 else []

            steps.append(
                ExecutionStep(
                    operation="shell.run",
                    inputs={"command": llm_step.command, "args": llm_step.args},
                    description=llm_step.description,
                    depends_on=depends_on,
                )
            )

        return ExecutionPlan(
            task_description=query,
            steps=steps,
            summary=response.summary,
        )


class TaskExecutor:
    """Executes task plans and returns results.

    Handles the full lifecycle:
    1. Convert plan to thunks
    2. Save thunks to persistence
    3. Execute thunks in order
    4. Collect and format results
    """

    def __init__(
        self,
        runtime: ThunkRuntime,
        persistence: PersistenceBackend,
    ) -> None:
        """Initialize the executor.

        Args:
            runtime: ThunkRuntime for execution.
            persistence: Backend for thunk storage.
        """
        self._runtime = runtime
        self._persistence = persistence

    def _extract_output(self, value: Any) -> str:
        """Extract human-readable output from a result value."""
        if isinstance(value, dict):
            # For verification results, extract the actual_value
            if "actual_value" in value and "passed" in value:
                actual = value.get("actual_value")
                return str(actual) if actual is not None else ""
            # For shell results, get stdout
            return value.get("stdout", value.get("output", str(value)))
        return str(value) if value else ""

    async def execute(
        self,
        plan: ExecutionPlan,
        capabilities: frozenset[str],
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> ExecutionResult:
        """Execute a plan and return results.

        Args:
            plan: The execution plan.
            capabilities: Capabilities for thunk execution.
            on_progress: Optional callback for progress updates.

        Returns:
            ExecutionResult with output or error.
        """
        start_time = datetime.now(UTC)
        thunks = plan.to_thunks(capabilities)
        thunk_ids = [str(t.id) for t in thunks]

        # Save all thunks
        for thunk in thunks:
            await self._persistence.save_thunk(thunk)

        # Execute thunks in order
        last_output: str | None = None
        last_error: str | None = None
        steps_completed = 0

        for i, thunk in enumerate(thunks):
            step = plan.steps[i] if i < len(plan.steps) else None
            description = step.description if step else f"Step {i + 1}"

            # Report progress
            if on_progress:
                on_progress(
                    ExecutionProgress(
                        step_index=i,
                        total_steps=len(thunks),
                        step_description=description,
                        status="running",
                    )
                )

            try:
                result = await self._runtime.force(thunk)

                if result.status == ThunkStatus.COMPLETED:
                    steps_completed += 1
                    last_output = self._extract_output(result.value)

                    if on_progress:
                        on_progress(
                            ExecutionProgress(
                                step_index=i,
                                total_steps=len(thunks),
                                step_description=description,
                                status="success",
                                output=last_output,
                            )
                        )
                else:
                    # Execution failed
                    last_error = result.error.message if result.error else "Unknown execution error"
                    if on_progress:
                        on_progress(
                            ExecutionProgress(
                                step_index=i,
                                total_steps=len(thunks),
                                step_description=description,
                                status="failure",
                                error=last_error,
                            )
                        )
                    break

            except Exception as e:
                last_error = str(e)
                if on_progress:
                    on_progress(
                        ExecutionProgress(
                            step_index=i,
                            total_steps=len(thunks),
                            step_description=description,
                            status="failure",
                            error=last_error,
                        )
                    )
                break

        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        success = steps_completed == len(thunks)

        return ExecutionResult(
            success=success,
            query=plan.task_description,
            output=last_output.strip() if last_output else "",
            steps_completed=steps_completed,
            total_steps=len(thunks),
            duration_ms=duration_ms,
            error=last_error,
            thunk_ids=thunk_ids,
        )

    async def execute_with_progress(
        self,
        plan: ExecutionPlan,
        capabilities: frozenset[str],
    ) -> AsyncGenerator[ExecutionProgress | ExecutionResult, None]:
        """Execute a plan yielding progress updates.

        Args:
            plan: The execution plan.
            capabilities: Capabilities for thunk execution.

        Yields:
            ExecutionProgress during execution, then ExecutionResult at the end.
        """
        progress_updates: list[ExecutionProgress] = []

        def collect_progress(p: ExecutionProgress) -> None:
            progress_updates.append(p)

        # Start execution
        result = await self.execute(plan, capabilities, on_progress=collect_progress)

        # Yield all progress updates
        for p in progress_updates:
            yield p

        # Yield final result
        yield result


async def ask(
    query: str,
    runtime: ThunkRuntime,
    persistence: PersistenceBackend,
    llm_provider: LLMProvider | None = None,
    capabilities: frozenset[str] | None = None,
    on_progress: Callable[[ExecutionProgress], None] | None = None,
) -> ExecutionResult:
    """Execute a natural language query end-to-end.

    This is the main entry point for the ask functionality.

    Args:
        query: Natural language task description.
        runtime: ThunkRuntime for execution.
        persistence: Backend for thunk storage.
        llm_provider: Optional LLM for complex task planning.
        capabilities: Capabilities for execution (defaults to shell.run, shell.pipe).
        on_progress: Optional callback for progress updates.

    Returns:
        ExecutionResult with output or error.
    """
    # Default capabilities for shell execution
    if capabilities is None:
        capabilities = frozenset({"shell.run", "shell.pipe", "verify.output"})

    # Create planner and executor
    planner = TaskPlanner(llm_provider=llm_provider)
    executor = TaskExecutor(runtime=runtime, persistence=persistence)

    # Plan the task
    try:
        plan = await planner.plan_async(query, capabilities)
    except ValueError as e:
        return ExecutionResult(
            success=False,
            query=query,
            output="",
            steps_completed=0,
            total_steps=0,
            duration_ms=0,
            error=str(e),
        )

    # Execute the plan
    return await executor.execute(plan, capabilities, on_progress=on_progress)
