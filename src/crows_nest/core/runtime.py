"""Thunk runtime - the execution engine.

The runtime forces thunks by resolving their dependencies and executing
their operations. It handles dependency ordering, capability checking,
and lifecycle events.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from crows_nest.core.registry import OperationNotFoundError, ThunkRegistry
from crows_nest.core.thunk import Thunk, ThunkError, ThunkRef, ThunkResult, ThunkStatus

if TYPE_CHECKING:
    from crows_nest.persistence.protocol import PersistenceBackend


class RuntimeEvent(Enum):
    """Events emitted during thunk execution."""

    THUNK_CREATED = "thunk.created"
    THUNK_QUEUED = "thunk.queued"
    THUNK_FORCING = "thunk.forcing"
    THUNK_COMPLETED = "thunk.completed"
    THUNK_FAILED = "thunk.failed"
    DEPENDENCY_RESOLVED = "dependency.resolved"


@dataclass
class RuntimeEventData:
    """Data associated with a runtime event."""

    event: RuntimeEvent
    thunk_id: UUID
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for event listeners
EventListener = Callable[[RuntimeEventData], Awaitable[None]]


class RuntimeError(Exception):
    """Base exception for runtime errors."""


class CapabilityError(RuntimeError):
    """Raised when a thunk lacks required capabilities."""

    def __init__(self, operation: str, required: frozenset[str], available: frozenset[str]) -> None:
        self.operation = operation
        self.required = required
        self.available = available
        missing = required - available
        super().__init__(
            f"Operation '{operation}' requires capabilities {sorted(missing)} "
            f"but thunk only has {sorted(available)}"
        )


class CyclicDependencyError(RuntimeError):
    """Raised when thunks have cyclic dependencies."""

    def __init__(self, cycle: list[UUID]) -> None:
        self.cycle = cycle
        super().__init__(f"Cyclic dependency detected: {' -> '.join(str(id) for id in cycle)}")


class DependencyNotFoundError(RuntimeError):
    """Raised when a thunk references a dependency that doesn't exist."""

    def __init__(self, thunk_id: UUID, dependency_id: UUID) -> None:
        self.thunk_id = thunk_id
        self.dependency_id = dependency_id
        super().__init__(f"Thunk {thunk_id} references non-existent dependency {dependency_id}")


def _extract_value_at_path(value: Any, path: str | None) -> Any:
    """Extract a value at a JSONPath-like path.

    Currently supports simple dot-notation paths like "foo.bar.baz".
    """
    if path is None:
        return value

    parts = path.split(".")
    current = value

    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Path component '{part}' not found in {current}")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
                current = current[index]
            except (ValueError, IndexError) as e:
                raise KeyError(f"Invalid list index '{part}'") from e
        else:
            raise TypeError(f"Cannot traverse path '{part}' on type {type(current).__name__}")

    return current


def _substitute_refs(value: Any, results: dict[UUID, ThunkResult]) -> Any:
    """Recursively substitute ThunkRefs with their resolved values."""
    if isinstance(value, ThunkRef):
        result = results.get(value.thunk_id)
        if result is None:
            raise DependencyNotFoundError(UUID(int=0), value.thunk_id)
        if result.is_failure:
            raise RuntimeError(f"Dependency {value.thunk_id} failed: {result.error}")
        return _extract_value_at_path(result.value, value.output_path)

    if isinstance(value, dict):
        return {k: _substitute_refs(v, results) for k, v in value.items()}

    if isinstance(value, list):
        return [_substitute_refs(item, results) for item in value]

    return value


def _topological_sort(thunks: dict[UUID, Thunk]) -> list[UUID]:
    """Sort thunks in dependency order (dependencies first).

    Uses Kahn's algorithm for topological sorting.

    Raises:
        CyclicDependencyError: If there's a cycle in the dependency graph.
    """
    # Build adjacency list and in-degree count
    in_degree: dict[UUID, int] = dict.fromkeys(thunks, 0)
    dependents: dict[UUID, list[UUID]] = {tid: [] for tid in thunks}

    for tid, thunk in thunks.items():
        for ref in thunk.get_dependencies():
            if ref.thunk_id in thunks:
                in_degree[tid] += 1
                dependents[ref.thunk_id].append(tid)

    # Start with thunks that have no dependencies
    queue = [tid for tid, degree in in_degree.items() if degree == 0]
    result: list[UUID] = []

    while queue:
        # Process in deterministic order
        queue.sort()
        current = queue.pop(0)
        result.append(current)

        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If not all thunks are in result, there's a cycle
    if len(result) != len(thunks):
        # Find a cycle for error reporting
        remaining = set(thunks.keys()) - set(result)
        cycle = list(remaining)[:5]  # Just show first few
        raise CyclicDependencyError(cycle)

    return result


@dataclass
class ThunkRuntime:
    """Runtime for executing thunks.

    The runtime is responsible for:
    - Resolving thunk dependencies in topological order
    - Checking capability requirements
    - Executing operation handlers
    - Emitting lifecycle events
    - Persisting results

    Example:
        registry = ThunkRegistry()
        runtime = ThunkRuntime(registry=registry)

        # Register operations
        @registry.operation("math.add")
        async def add(a: int, b: int) -> int:
            return a + b

        # Create and force a thunk
        thunk = Thunk.create("math.add", {"a": 1, "b": 2})
        result = await runtime.force(thunk)
        print(result.value)  # 3
    """

    registry: ThunkRegistry
    persistence: PersistenceBackend | None = None
    _listeners: dict[RuntimeEvent, list[EventListener]] = field(default_factory=dict)
    _results_cache: dict[UUID, ThunkResult] = field(default_factory=dict)

    async def force(self, thunk: Thunk) -> ThunkResult:
        """Force a single thunk, resolving dependencies first.

        Args:
            thunk: The thunk to force.

        Returns:
            The result of forcing the thunk.
        """
        return await self.force_all({thunk.id: thunk}, target=thunk.id)

    async def force_all(
        self,
        thunks: dict[UUID, Thunk],
        target: UUID | None = None,
    ) -> ThunkResult:
        """Force multiple thunks, resolving dependencies.

        Args:
            thunks: Map of thunk ID to thunk.
            target: If specified, return the result for this thunk.
                Otherwise return result of the last thunk in topological order.

        Returns:
            The result of the target thunk.
        """
        if not thunks:
            raise ValueError("No thunks to force")

        # Validate all dependencies exist
        for thunk in thunks.values():
            for ref in thunk.get_dependencies():
                if ref.thunk_id not in thunks and ref.thunk_id not in self._results_cache:
                    # Try to load from persistence
                    if self.persistence:
                        existing = await self.persistence.get_result(ref.thunk_id)
                        if existing:
                            self._results_cache[ref.thunk_id] = existing
                            continue
                    raise DependencyNotFoundError(thunk.id, ref.thunk_id)

        # Sort thunks in dependency order
        order = _topological_sort(thunks)

        # Force each thunk in order
        for thunk_id in order:
            thunk = thunks[thunk_id]

            # Skip if already have result
            if thunk_id in self._results_cache:
                continue

            await self._emit(RuntimeEvent.THUNK_FORCING, thunk_id)

            result = await self._force_single(thunk)
            self._results_cache[thunk_id] = result

            if self.persistence:
                await self.persistence.save_result(result)

            if result.is_success:
                await self._emit(RuntimeEvent.THUNK_COMPLETED, thunk_id, {"value": result.value})
            else:
                await self._emit(RuntimeEvent.THUNK_FAILED, thunk_id, {"error": result.error})

        # Return target result or last
        target_id = target or order[-1]
        return self._results_cache[target_id]

    async def _force_single(self, thunk: Thunk) -> ThunkResult:
        """Force a single thunk (dependencies must already be resolved)."""
        start_time = time.monotonic()

        try:
            # Get operation info
            try:
                op_info = self.registry.get(thunk.operation)
            except OperationNotFoundError as e:
                return ThunkResult(
                    thunk_id=thunk.id,
                    status=ThunkStatus.FAILED,
                    value=None,
                    error=ThunkError.from_exception(e, {"operation": thunk.operation}),
                    forced_at=datetime.now(UTC),
                    duration_ms=int((time.monotonic() - start_time) * 1000),
                )

            # Check capabilities
            if op_info.required_capabilities:
                missing = op_info.required_capabilities - thunk.metadata.capabilities
                if missing:
                    raise CapabilityError(
                        thunk.operation,
                        op_info.required_capabilities,
                        thunk.metadata.capabilities,
                    )

            # Resolve dependencies in inputs
            resolved_inputs = _substitute_refs(thunk.inputs, self._results_cache)

            # Execute handler
            handler = op_info.handler
            result_value = await handler(**resolved_inputs)

            # Handle thunk-returning-thunk pattern
            if isinstance(result_value, Thunk):
                # Store the new thunk and force it
                new_thunk = result_value
                if self.persistence:
                    await self.persistence.save_thunk(new_thunk)
                await self._emit(RuntimeEvent.THUNK_CREATED, new_thunk.id)

                # Recursively force the returned thunk
                return await self.force(new_thunk)

            return ThunkResult(
                thunk_id=thunk.id,
                status=ThunkStatus.COMPLETED,
                value=result_value,
                error=None,
                forced_at=datetime.now(UTC),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        except Exception as e:
            return ThunkResult(
                thunk_id=thunk.id,
                status=ThunkStatus.FAILED,
                value=None,
                error=ThunkError.from_exception(e, {"operation": thunk.operation}),
                forced_at=datetime.now(UTC),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

    def get_result(self, thunk_id: UUID) -> ThunkResult | None:
        """Get a cached result for a thunk."""
        return self._results_cache.get(thunk_id)

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()

    # Event handling

    def on(self, event: RuntimeEvent, listener: EventListener) -> None:
        """Register an event listener."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def off(self, event: RuntimeEvent, listener: EventListener) -> None:
        """Unregister an event listener."""
        if event in self._listeners:
            with contextlib.suppress(ValueError):
                self._listeners[event].remove(listener)

    async def _emit(
        self,
        event: RuntimeEvent,
        thunk_id: UUID,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to all listeners."""
        if event not in self._listeners:
            return

        event_data = RuntimeEventData(
            event=event,
            thunk_id=thunk_id,
            timestamp=datetime.now(UTC),
            data=data or {},
        )

        for listener in self._listeners[event]:
            try:  # noqa: SIM105 - can't use contextlib.suppress with await
                await listener(event_data)
            except Exception:  # noqa: S110
                # Don't let listener errors break execution
                # TODO: Add structured logging here
                pass
