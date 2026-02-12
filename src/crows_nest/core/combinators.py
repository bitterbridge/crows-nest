"""Thunk combinators for composing operations.

Provides higher-order operations that combine, sequence, and transform thunks.
These are the building blocks for complex workflows.
"""

from __future__ import annotations

import asyncio
from typing import Any

from crows_nest.core.registry import get_global_registry, thunk_operation
from crows_nest.core.thunk import Thunk


@thunk_operation(
    name="thunk.sequence",
    description="Execute thunks in sequence, passing each output to the next as context.",
    required_capabilities=frozenset(),  # Capabilities checked per-thunk
)
async def thunk_sequence(
    thunks: list[dict[str, Any]],
    initial_context: dict[str, Any] | None = None,
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    """Execute thunks in sequence.

    Args:
        thunks: List of thunk specifications, each with 'operation' and 'inputs'.
        initial_context: Optional context passed to the first thunk.
        stop_on_failure: If True, stop sequence on first failure.

    Returns:
        Dict with results from each thunk and final output.

    Example:
        thunk.sequence(thunks=[
            {"operation": "shell.run", "inputs": {"command": "ls"}},
            {"operation": "verify.output", "inputs": {"schema": {...}}}
        ])
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    results: list[dict[str, Any]] = []
    outputs: list[Any] = []
    final_output: Any = None

    for i, thunk_spec in enumerate(thunks):
        operation = thunk_spec.get("operation")
        inputs = thunk_spec.get("inputs", {})

        if not operation:
            results.append(
                {
                    "index": i,
                    "status": "failed",
                    "error": "Missing 'operation' in thunk spec",
                }
            )
            if stop_on_failure:
                break
            continue

        # Only merge initial_context for first thunk, not accumulated outputs
        merged_inputs = {**initial_context, **inputs} if i == 0 and initial_context else inputs

        thunk = Thunk.create(operation=operation, inputs=merged_inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            results.append(
                {
                    "index": i,
                    "operation": operation,
                    "status": "failed",
                    "error": error_msg,
                }
            )
            if stop_on_failure:
                break
        else:
            output = result.value
            results.append(
                {
                    "index": i,
                    "operation": operation,
                    "status": "success",
                    "output": output,
                }
            )
            final_output = output
            outputs.append(output)

    return {
        "total": len(thunks),
        "completed": len(results),
        "success": all(r.get("status") == "success" for r in results),
        "results": results,
        "final_output": final_output,
    }


@thunk_operation(
    name="thunk.parallel",
    description="Execute thunks concurrently and collect all results.",
    required_capabilities=frozenset(),  # Capabilities checked per-thunk
)
async def thunk_parallel(
    thunks: list[dict[str, Any]],
    max_concurrency: int | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Execute thunks in parallel.

    Args:
        thunks: List of thunk specifications, each with 'operation' and 'inputs'.
        max_concurrency: Maximum concurrent executions (None = unlimited).
        fail_fast: If True, cancel remaining thunks on first failure.

    Returns:
        Dict with results from all thunks.

    Example:
        thunk.parallel(thunks=[
            {"operation": "shell.run", "inputs": {"command": "task1"}},
            {"operation": "shell.run", "inputs": {"command": "task2"}},
            {"operation": "shell.run", "inputs": {"command": "task3"}}
        ], max_concurrency=2)
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def run_one(index: int, spec: dict[str, Any]) -> dict[str, Any]:
        operation = spec.get("operation")
        inputs = spec.get("inputs", {})

        if not operation:
            return {
                "index": index,
                "status": "failed",
                "error": "Missing 'operation' in thunk spec",
            }

        async def execute() -> dict[str, Any]:
            thunk = Thunk.create(operation=operation, inputs=inputs)
            result = await runtime.force(thunk)

            if result.is_failure:
                error_msg = result.error.message if result.error else "Unknown error"
                return {
                    "index": index,
                    "operation": operation,
                    "status": "failed",
                    "error": error_msg,
                }
            return {
                "index": index,
                "operation": operation,
                "status": "success",
                "output": result.value,
            }

        if semaphore:
            async with semaphore:
                return await execute()
        return await execute()

    if fail_fast:
        # Use TaskGroup for fail-fast behavior
        results: list[dict[str, Any]] = [{}] * len(thunks)
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(run_one(i, spec)) for i, spec in enumerate(thunks)]
            for i, task in enumerate(tasks):
                results[i] = task.result()
        except* Exception:
            # Collect whatever results we have
            for i, task in enumerate(tasks):
                if task.done() and not task.cancelled():
                    results[i] = task.result()
                else:
                    results[i] = {"index": i, "status": "cancelled"}
    else:
        # Gather all, don't cancel on failure
        results = await asyncio.gather(*[run_one(i, spec) for i, spec in enumerate(thunks)])

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    return {
        "total": len(thunks),
        "succeeded": succeeded,
        "failed": failed,
        "all_success": failed == 0,
        "results": list(results),
    }


@thunk_operation(
    name="thunk.race",
    description="Execute thunks concurrently, return the first successful result.",
    required_capabilities=frozenset(),
)
async def thunk_race(
    thunks: list[dict[str, Any]],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Race thunks, returning first success.

    Args:
        thunks: List of thunk specifications.
        timeout_seconds: Optional timeout for the entire race.

    Returns:
        Dict with the winning thunk's result or all failures.

    Example:
        thunk.race(thunks=[
            {"operation": "api.call", "inputs": {"endpoint": "primary"}},
            {"operation": "api.call", "inputs": {"endpoint": "backup"}}
        ])
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    async def run_one(index: int, spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        operation = spec.get("operation")
        inputs = spec.get("inputs", {})

        if not operation:
            raise ValueError(f"Missing 'operation' in thunk spec at index {index}")

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            raise RuntimeError(error_msg)

        return index, {
            "operation": operation,
            "output": result.value,
        }

    tasks = [asyncio.create_task(run_one(i, spec)) for i, spec in enumerate(thunks)]
    errors: list[dict[str, Any]] = []

    try:
        if timeout_seconds:
            deadline = asyncio.get_event_loop().time() + timeout_seconds

        while tasks:
            if timeout_seconds:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    for t in tasks:
                        t.cancel()
                    return {
                        "winner": None,
                        "timed_out": True,
                        "errors": errors,
                    }
                done, pending = await asyncio.wait(
                    tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            tasks = list(pending)

            for task in done:
                try:
                    index, result = task.result()
                    # Winner! Cancel remaining tasks
                    for t in tasks:
                        t.cancel()
                    return {
                        "winner": index,
                        "result": result,
                        "timed_out": False,
                    }
                except Exception as e:
                    errors.append({"error": str(e)})

        # All failed
        return {
            "winner": None,
            "timed_out": False,
            "errors": errors,
        }
    finally:
        for t in tasks:
            t.cancel()


@thunk_operation(
    name="thunk.retry",
    description="Retry a thunk on failure with configurable backoff.",
    required_capabilities=frozenset(),
)
async def thunk_retry(
    operation: str,
    inputs: dict[str, Any],
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    exponential_backoff: bool = True,
    max_delay_seconds: float = 60.0,
    retry_on: list[str] | None = None,
) -> dict[str, Any]:
    """Retry a thunk with backoff.

    Args:
        operation: The operation to execute.
        inputs: Inputs to the operation.
        max_attempts: Maximum number of attempts.
        base_delay_seconds: Initial delay between retries.
        exponential_backoff: If True, double delay after each failure.
        max_delay_seconds: Maximum delay between retries.
        retry_on: List of error substrings to retry on (None = retry all).

    Returns:
        Dict with result or final error after all attempts.

    Example:
        thunk.retry(
            operation="api.call",
            inputs={"url": "https://flaky.api/data"},
            max_attempts=5,
            exponential_backoff=True
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    attempts: list[dict[str, Any]] = []
    delay = base_delay_seconds

    for attempt in range(1, max_attempts + 1):
        # Create fresh thunk each attempt (new ID)
        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_success:
            return {
                "success": True,
                "attempts": attempt,
                "output": result.value,
                "history": attempts,
            }

        # Failed - record and maybe retry
        error_str = result.error.message if result.error else "Unknown error"
        attempts.append(
            {
                "attempt": attempt,
                "error": error_str,
            }
        )

        # Check if we should retry this error
        if retry_on and not any(substr in error_str for substr in retry_on):
            return {
                "success": False,
                "attempts": attempt,
                "error": error_str,
                "reason": "Error not in retry_on list",
                "history": attempts,
            }

        # Don't delay after final attempt
        if attempt < max_attempts:
            await asyncio.sleep(delay)
            if exponential_backoff:
                delay = min(delay * 2, max_delay_seconds)

    return {
        "success": False,
        "attempts": max_attempts,
        "error": attempts[-1]["error"] if attempts else "Unknown error",
        "history": attempts,
    }


@thunk_operation(
    name="thunk.timeout",
    description="Execute a thunk with a timeout.",
    required_capabilities=frozenset(),
)
async def thunk_timeout(
    operation: str,
    inputs: dict[str, Any],
    timeout_seconds: float,
    default: Any = None,
) -> dict[str, Any]:
    """Execute a thunk with a timeout.

    Args:
        operation: The operation to execute.
        inputs: Inputs to the operation.
        timeout_seconds: Maximum time to wait.
        default: Value to return on timeout (if not provided, raises error).

    Returns:
        Dict with result or timeout indicator.

    Example:
        thunk.timeout(
            operation="shell.run",
            inputs={"command": "slow_task"},
            timeout_seconds=30
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    thunk = Thunk.create(operation=operation, inputs=inputs)

    try:
        result = await asyncio.wait_for(runtime.force(thunk), timeout=timeout_seconds)
        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return {
                "timed_out": False,
                "output": None,
                "error": error_msg,
            }
        return {
            "timed_out": False,
            "output": result.value,
        }
    except TimeoutError:
        return {
            "timed_out": True,
            "output": default,
            "timeout_seconds": timeout_seconds,
        }


@thunk_operation(
    name="thunk.fallback",
    description="Try primary thunk, fall back to secondary on failure.",
    required_capabilities=frozenset(),
)
async def thunk_fallback(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    fallback_on_timeout: bool = True,
    primary_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute primary thunk with fallback.

    Args:
        primary: Primary thunk specification.
        fallback: Fallback thunk specification.
        fallback_on_timeout: Whether to use fallback on timeout.
        primary_timeout_seconds: Timeout for primary (None = no timeout).

    Returns:
        Dict with result and which thunk was used.

    Example:
        thunk.fallback(
            primary={"operation": "api.call", "inputs": {"endpoint": "main"}},
            fallback={"operation": "cache.get", "inputs": {"key": "cached_data"}}
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    primary_error: str | None = None

    # Try primary
    operation = primary.get("operation")
    inputs = primary.get("inputs", {})
    if not operation:
        primary_error = "Missing 'operation' in primary thunk spec"
    else:
        thunk = Thunk.create(operation=operation, inputs=inputs)
        try:
            if primary_timeout_seconds:
                result = await asyncio.wait_for(
                    runtime.force(thunk), timeout=primary_timeout_seconds
                )
            else:
                result = await runtime.force(thunk)

            if result.is_success:
                return {
                    "used": "primary",
                    "output": result.value,
                }
            primary_error = result.error.message if result.error else "Unknown error"
        except TimeoutError:
            if not fallback_on_timeout:
                return {
                    "used": "none",
                    "output": None,
                    "primary_error": "Timeout",
                }
            primary_error = "Timeout"

    # Try fallback
    operation = fallback.get("operation")
    inputs = fallback.get("inputs", {})
    if not operation:
        return {
            "used": "none",
            "output": None,
            "primary_error": primary_error,
            "fallback_error": "Missing 'operation' in fallback thunk spec",
        }

    thunk = Thunk.create(operation=operation, inputs=inputs)
    result = await runtime.force(thunk)

    if result.is_success:
        return {
            "used": "fallback",
            "output": result.value,
            "primary_error": primary_error,
        }
    return {
        "used": "none",
        "output": None,
        "primary_error": primary_error,
        "fallback_error": result.error.message if result.error else "Unknown error",
    }


@thunk_operation(
    name="thunk.conditional",
    description="Execute a thunk only if a condition is met.",
    required_capabilities=frozenset(),
)
async def thunk_conditional(
    condition: dict[str, Any],
    then_thunk: dict[str, Any],
    else_thunk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conditional thunk execution.

    Args:
        condition: Thunk that returns a boolean-like value.
        then_thunk: Thunk to execute if condition is truthy.
        else_thunk: Optional thunk to execute if condition is falsy.

    Returns:
        Dict with which branch was taken and its result.

    Example:
        thunk.conditional(
            condition={"operation": "file.exists", "inputs": {"path": "/tmp/flag"}},
            then_thunk={"operation": "file.read", "inputs": {"path": "/tmp/flag"}},
            else_thunk={"operation": "file.create", "inputs": {"path": "/tmp/flag"}}
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    async def run_thunk(spec: dict[str, Any]) -> Any:
        operation = spec.get("operation")
        inputs = spec.get("inputs", {})
        if not operation:
            raise ValueError("Missing 'operation' in thunk spec")
        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)
        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            raise RuntimeError(error_msg)
        return result.value

    # Evaluate condition
    condition_result = await run_thunk(condition)

    # Determine truthiness
    is_truthy = bool(condition_result)
    if isinstance(condition_result, dict):
        # Check common patterns for dict results
        is_truthy = bool(
            condition_result.get("success", condition_result.get("exists", bool(condition_result)))
        )

    if is_truthy:
        output = await run_thunk(then_thunk)
        return {
            "branch": "then",
            "condition_result": condition_result,
            "output": output,
        }
    elif else_thunk:
        output = await run_thunk(else_thunk)
        return {
            "branch": "else",
            "condition_result": condition_result,
            "output": output,
        }
    else:
        return {
            "branch": "skipped",
            "condition_result": condition_result,
            "output": None,
        }


@thunk_operation(
    name="thunk.map",
    description="Apply a thunk to each item in a collection.",
    required_capabilities=frozenset(),
)
async def thunk_map(
    operation: str,
    items: list[Any],
    input_key: str = "item",
    extra_inputs: dict[str, Any] | None = None,
    max_concurrency: int | None = None,
    stop_on_failure: bool = False,
) -> dict[str, Any]:
    """Map a thunk over a collection.

    Args:
        operation: The operation to apply to each item.
        items: List of items to process.
        input_key: Key name for the item in inputs.
        extra_inputs: Additional inputs to include with each call.
        max_concurrency: Maximum concurrent executions.
        stop_on_failure: If True, stop on first failure.

    Returns:
        Dict with results for each item.

    Example:
        thunk.map(
            operation="file.process",
            items=["a.txt", "b.txt", "c.txt"],
            input_key="filename",
            max_concurrency=2
        )
    """
    # Build thunk specs for parallel execution
    thunks = []
    for item in items:
        inputs = {input_key: item}
        if extra_inputs:
            inputs.update(extra_inputs)
        thunks.append({"operation": operation, "inputs": inputs})

    # Use parallel combinator
    result = await thunk_parallel(
        thunks=thunks,
        max_concurrency=max_concurrency,
        fail_fast=stop_on_failure,
    )

    # Add items to results for clarity
    for i, item in enumerate(items):
        if i < len(result["results"]):
            result["results"][i]["item"] = item

    return result


@thunk_operation(
    name="thunk.reduce",
    description="Reduce a collection using a thunk as the reducer.",
    required_capabilities=frozenset(),
)
async def thunk_reduce(
    operation: str,
    items: list[Any],
    initial: Any,
    accumulator_key: str = "accumulator",
    item_key: str = "item",
    extra_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce a collection with a thunk.

    Args:
        operation: The reducer operation (takes accumulator and item).
        items: List of items to reduce.
        initial: Initial accumulator value.
        accumulator_key: Key name for accumulator in inputs.
        item_key: Key name for current item in inputs.
        extra_inputs: Additional inputs to include with each call.

    Returns:
        Dict with final accumulated value.

    Example:
        thunk.reduce(
            operation="math.add",
            items=[1, 2, 3, 4, 5],
            initial=0,
            accumulator_key="sum",
            item_key="value"
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    accumulator = initial
    steps: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        inputs = {
            accumulator_key: accumulator,
            item_key: item,
        }
        if extra_inputs:
            inputs.update(extra_inputs)

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            steps.append(
                {
                    "index": i,
                    "item": item,
                    "error": error_msg,
                    "status": "failed",
                }
            )
            return {
                "success": False,
                "result": accumulator,
                "failed_at": i,
                "steps": steps,
            }

        output = result.value

        # Extract new accumulator value
        if isinstance(output, dict):
            accumulator = output.get("result", output.get("value", output))
        else:
            accumulator = output

        steps.append(
            {
                "index": i,
                "item": item,
                "accumulator": accumulator,
                "status": "success",
            }
        )

    return {
        "success": True,
        "result": accumulator,
        "steps": steps,
    }
