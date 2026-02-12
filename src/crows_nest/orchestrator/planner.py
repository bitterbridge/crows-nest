"""Task planning and orchestration.

The pid0 orchestrator converts natural language task descriptions
into executable plans consisting of shell commands and verification steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from crows_nest.core.registry import thunk_operation
from crows_nest.core.thunk import Thunk, ThunkRef
from crows_nest.shell.operations import ALLOWED_COMMANDS


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""

    operation: str
    inputs: dict[str, Any]
    description: str
    depends_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "inputs": self.inputs,
            "description": self.description,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionStep:
        """Create from dictionary."""
        return cls(
            operation=data["operation"],
            inputs=data["inputs"],
            description=data.get("description", ""),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class ExecutionPlan:
    """A complete execution plan with multiple steps."""

    task_description: str
    steps: list[ExecutionStep]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_description": self.task_description,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.summary,
        }

    def to_thunks(
        self,
        capabilities: frozenset[str],
        _trace_id: str | None = None,
    ) -> list[Thunk]:
        """Convert plan to a list of Thunk objects.

        Args:
            capabilities: Capabilities to grant to thunks.
            _trace_id: Optional trace ID for linking (future use).

        Returns:
            List of Thunk objects ready for execution.
        """
        thunks: list[Thunk] = []
        step_to_thunk: dict[int, Thunk] = {}

        for i, step in enumerate(self.steps):
            # Build inputs with dependency references
            inputs = dict(step.inputs)

            # If this step depends on others, add ThunkRefs to inputs
            # For verification steps, the 'value' input should reference previous output
            for dep_idx in step.depends_on:
                if dep_idx in step_to_thunk:
                    dep_thunk = step_to_thunk[dep_idx]
                    # For verify.output, use the 'value' field for the dependency
                    # Extract stdout from shell results
                    if step.operation == "verify.output" and inputs.get("value") is None:
                        inputs["value"] = ThunkRef(thunk_id=dep_thunk.id, output_path="stdout")
                    else:
                        # For other operations, add as 'stdin' dependency
                        inputs["_dependency_refs"] = inputs.get("_dependency_refs", [])
                        inputs["_dependency_refs"].append(ThunkRef(thunk_id=dep_thunk.id))

            # Create thunk
            thunk = Thunk.create(
                operation=step.operation,
                inputs=inputs,
                capabilities=capabilities,
                tags={"step": str(i), "description": step.description},
            )

            thunks.append(thunk)
            step_to_thunk[i] = thunk

        return thunks


class PlannerConfig(BaseModel):
    """Configuration for the task planner."""

    max_steps: int = Field(default=10, description="Maximum steps in a plan")
    allowed_operations: list[str] = Field(
        default_factory=lambda: ["shell.run", "shell.pipe", "verify.output"],
        description="Operations the planner can use",
    )


class TaskPlanInput(BaseModel):
    """Input schema for orchestrate.plan operation."""

    task_description: str = Field(description="Natural language task description")
    available_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities available for execution",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional planner configuration",
    )


class TaskPlanOutput(BaseModel):
    """Output schema for orchestrate.plan operation."""

    task_description: str = Field(description="Original task description")
    steps: list[dict[str, Any]] = Field(description="Execution steps")
    summary: str = Field(description="Human-readable plan summary")
    thunk_ids: list[str] = Field(description="IDs of created thunks")


# Common task patterns that can be executed without LLM
TASK_PATTERNS: dict[str, list[ExecutionStep]] = {
    "list files": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "ls", "args": ["-la"]},
            description="List all files in current directory",
        ),
    ],
    "list python files": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "find", "args": [".", "-name", "*.py", "-type", "f"]},
            description="Find all Python files",
        ),
    ],
    "count files": [
        ExecutionStep(
            operation="shell.pipe",
            inputs={
                "commands": [
                    {"command": "ls", "args": ["-1"]},
                    {"command": "wc", "args": ["-l"]},
                ],
            },
            description="Count files in current directory",
        ),
        ExecutionStep(
            operation="verify.output",
            inputs={
                "schema": {"type": "integer", "min_value": 0},
                "value": None,  # Will be filled from previous step
            },
            description="Verify count is a non-negative integer",
            depends_on=[0],
        ),
    ],
    "count python files": [
        ExecutionStep(
            operation="shell.pipe",
            inputs={
                "commands": [
                    {"command": "find", "args": [".", "-name", "*.py", "-type", "f"]},
                    {"command": "wc", "args": ["-l"]},
                ],
            },
            description="Count Python files",
        ),
        ExecutionStep(
            operation="verify.output",
            inputs={
                "schema": {"type": "integer", "min_value": 0},
                "value": None,
            },
            description="Verify count is a non-negative integer",
            depends_on=[0],
        ),
    ],
    "show current directory": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "pwd", "args": []},
            description="Print working directory",
        ),
    ],
    "show date": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "date", "args": []},
            description="Show current date and time",
        ),
    ],
    "whoami": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "whoami", "args": []},
            description="Show current user",
        ),
    ],
    "system info": [
        ExecutionStep(
            operation="shell.run",
            inputs={"command": "uname", "args": ["-a"]},
            description="Show system information",
        ),
    ],
}


def _match_task_pattern(task: str) -> list[ExecutionStep] | None:
    """Try to match a task description to a known pattern.

    Args:
        task: The task description.

    Returns:
        Execution steps if matched, None otherwise.
    """
    task_lower = task.lower().strip()

    # Direct matches
    if task_lower in TASK_PATTERNS:
        return TASK_PATTERNS[task_lower]

    # Fuzzy matches
    for pattern, steps in TASK_PATTERNS.items():
        if pattern in task_lower or task_lower in pattern:
            return steps

    # Keyword matching
    if "python" in task_lower and "file" in task_lower:
        if "count" in task_lower or "how many" in task_lower:
            return TASK_PATTERNS["count python files"]
        return TASK_PATTERNS["list python files"]

    if "file" in task_lower:
        if "count" in task_lower or "how many" in task_lower:
            return TASK_PATTERNS["count files"]
        if "list" in task_lower or "show" in task_lower:
            return TASK_PATTERNS["list files"]

    if "date" in task_lower or "time" in task_lower:
        return TASK_PATTERNS["show date"]

    if "directory" in task_lower or "pwd" in task_lower:
        return TASK_PATTERNS["show current directory"]

    if "user" in task_lower or "whoami" in task_lower:
        return TASK_PATTERNS["whoami"]

    if "system" in task_lower or "uname" in task_lower:
        return TASK_PATTERNS["system info"]

    return None


def _parse_simple_command(task: str) -> list[ExecutionStep] | None:
    """Try to parse a simple command from the task.

    Args:
        task: The task description.

    Returns:
        Single execution step if parseable, None otherwise.
    """
    # Handle "run X" or "execute X" patterns
    task_lower = task.lower().strip()

    prefixes = ["run ", "execute ", "do "]
    for prefix in prefixes:
        if task_lower.startswith(prefix):
            rest = task[len(prefix) :].strip()
            parts = rest.split()
            if parts:
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []

                # Only allow if command is in allowlist
                if command in ALLOWED_COMMANDS:
                    return [
                        ExecutionStep(
                            operation="shell.run",
                            inputs={"command": command, "args": args},
                            description=f"Run {command} command",
                        ),
                    ]

    return None


def create_plan(
    task_description: str,
    _available_capabilities: frozenset[str],
    config: PlannerConfig | None = None,
) -> ExecutionPlan:
    """Create an execution plan for a task.

    This uses pattern matching and simple parsing. For complex tasks,
    an LLM-based planner would be used (future enhancement).

    Args:
        task_description: Natural language task description.
        _available_capabilities: Capabilities available for execution (future use for filtering).
        config: Optional planner configuration.

    Returns:
        ExecutionPlan with steps to execute.

    Raises:
        ValueError: If no plan can be created for the task.
    """
    config = config or PlannerConfig()

    # Try pattern matching first
    steps = _match_task_pattern(task_description)
    if steps:
        return ExecutionPlan(
            task_description=task_description,
            steps=steps,
            summary=f"Execute {len(steps)} step(s) for: {task_description}",
        )

    # Try simple command parsing
    steps = _parse_simple_command(task_description)
    if steps:
        return ExecutionPlan(
            task_description=task_description,
            steps=steps,
            summary=f"Execute command for: {task_description}",
        )

    # If we can't create a plan, raise an error
    # In a full implementation, this would fall back to LLM-based planning
    msg = f"Cannot create execution plan for task: {task_description}"
    raise ValueError(msg)


@thunk_operation(
    name="orchestrate.plan",
    description="Create an execution plan for a natural language task",
    required_capabilities=frozenset({"orchestrate.plan"}),
)
async def orchestrate_plan(
    task_description: str,
    available_capabilities: list[str] | None = None,
    config: dict[str, Any] | None = None,
    *,
    _capabilities: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Create an execution plan for a task.

    Args:
        task_description: Natural language description of the task.
        available_capabilities: List of capabilities available for execution.
        config: Optional planner configuration.
        _capabilities: Capabilities passed by runtime (internal).

    Returns:
        Dictionary with plan steps and summary.

    Raises:
        ValueError: If no plan can be created.
    """
    caps = frozenset(available_capabilities or [])
    planner_config = PlannerConfig(**config) if config else None

    # Create plan
    plan = create_plan(task_description, caps, planner_config)

    # Convert to thunks (but don't persist them - that's the caller's job)
    thunks = plan.to_thunks(caps)

    return {
        "task_description": plan.task_description,
        "steps": [step.to_dict() for step in plan.steps],
        "summary": plan.summary,
        "thunk_ids": [str(t.id) for t in thunks],
    }
