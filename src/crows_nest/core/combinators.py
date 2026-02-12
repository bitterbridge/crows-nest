"""Thunk combinators for composing operations.

Provides higher-order operations that combine, sequence, and transform thunks.
These are the building blocks for complex workflows.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from crows_nest.core.registry import get_global_registry, thunk_operation
from crows_nest.core.thunk import Thunk

if TYPE_CHECKING:
    from crows_nest.core.context import ThunkContext


@thunk_operation(
    name="thunk.sequence",
    description="Execute thunks in sequence, passing each output to the next as context.",
    required_capabilities=frozenset(),  # Capabilities checked per-thunk
)
async def thunk_sequence(
    thunks: list[dict[str, Any]],
    initial_context: dict[str, Any] | None = None,
    stop_on_failure: bool = True,
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Execute thunks in sequence.

    Args:
        thunks: List of thunk specifications, each with 'operation' and 'inputs'.
        initial_context: Optional context passed to the first thunk.
        stop_on_failure: If True, stop sequence on first failure.
        _context: Execution context for cancellation (injected by runtime).

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
        # Check for cancellation before each step
        if _context:
            _context.raise_if_cancelled()

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
        result = await runtime.force(thunk, context=_context)

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
    _context: ThunkContext | None = None,
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
        _context: Execution context for cancellation (injected by runtime).

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
        # Check for cancellation before each attempt
        if _context:
            _context.raise_if_cancelled()

        # Create fresh thunk each attempt (new ID)
        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk, context=_context)

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
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Reduce a collection with a thunk.

    Args:
        operation: The reducer operation (takes accumulator and item).
        items: List of items to reduce.
        initial: Initial accumulator value.
        accumulator_key: Key name for accumulator in inputs.
        item_key: Key name for current item in inputs.
        extra_inputs: Additional inputs to include with each call.
        _context: Execution context for cancellation (injected by runtime).

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
        # Check for cancellation before each step
        if _context:
            _context.raise_if_cancelled()

        inputs = {
            accumulator_key: accumulator,
            item_key: item,
        }
        if extra_inputs:
            inputs.update(extra_inputs)

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk, context=_context)

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


@thunk_operation(
    name="thunk.catch",
    description="Execute a thunk and catch errors, optionally recovering with a handler.",
    required_capabilities=frozenset(),
)
async def thunk_catch(
    operation: str,
    inputs: dict[str, Any],
    default: Any = None,
    error_handler: dict[str, Any] | None = None,
    catch_types: list[str] | None = None,
) -> dict[str, Any]:
    """Catch errors from a thunk.

    Args:
        operation: The operation to execute.
        inputs: Inputs to the operation.
        default: Default value to return on error (if no error_handler).
        error_handler: Optional thunk to call on error (receives error info as input).
        catch_types: List of error type substrings to catch (None = catch all).

    Returns:
        Dict with result or caught error info.

    Example:
        thunk.catch(
            operation="api.call",
            inputs={"url": "https://example.com"},
            default={"cached": True},
            catch_types=["ConnectionError", "Timeout"]
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    thunk = Thunk.create(operation=operation, inputs=inputs)
    result = await runtime.force(thunk)

    if result.is_success:
        return {
            "caught": False,
            "output": result.value,
        }

    # Error occurred
    error_type = result.error.error_type if result.error else "Unknown"
    error_msg = result.error.message if result.error else "Unknown error"

    # Check if we should catch this error type
    if catch_types and not any(t in error_type for t in catch_types):
        # Don't catch - propagate as failure
        return {
            "caught": False,
            "output": None,
            "error": error_msg,
            "error_type": error_type,
            "propagated": True,
        }

    # Caught the error
    if error_handler:
        # Run error handler with error info
        handler_op = error_handler.get("operation")
        handler_inputs = error_handler.get("inputs", {})
        if handler_op:
            # Add error info to handler inputs
            handler_inputs = {
                **handler_inputs,
                "_error": error_msg,
                "_error_type": error_type,
            }
            handler_thunk = Thunk.create(operation=handler_op, inputs=handler_inputs)
            handler_result = await runtime.force(handler_thunk)
            if handler_result.is_success:
                return {
                    "caught": True,
                    "recovered": True,
                    "output": handler_result.value,
                    "original_error": error_msg,
                }
            # Handler also failed
            return {
                "caught": True,
                "recovered": False,
                "output": default,
                "original_error": error_msg,
                "handler_error": (
                    handler_result.error.message if handler_result.error else "Unknown"
                ),
            }

    # No handler - return default
    return {
        "caught": True,
        "recovered": False,
        "output": default,
        "original_error": error_msg,
        "error_type": error_type,
    }


@thunk_operation(
    name="thunk.ensure",
    description="Execute a thunk with guaranteed cleanup, regardless of success or failure.",
    required_capabilities=frozenset(),
)
async def thunk_ensure(
    operation: str,
    inputs: dict[str, Any],
    finally_thunk: dict[str, Any],
    pass_result_to_finally: bool = False,
) -> dict[str, Any]:
    """Execute with guaranteed cleanup.

    Args:
        operation: The main operation to execute.
        inputs: Inputs to the main operation.
        finally_thunk: Thunk to execute regardless of main result.
        pass_result_to_finally: If True, pass main result to finally thunk.

    Returns:
        Dict with main result and finally execution status.

    Example:
        thunk.ensure(
            operation="db.query",
            inputs={"sql": "SELECT *"},
            finally_thunk={"operation": "db.close", "inputs": {}}
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    # Execute main operation
    thunk = Thunk.create(operation=operation, inputs=inputs)
    result = await runtime.force(thunk)

    main_success = result.is_success
    main_output = result.value if main_success else None
    main_error = result.error.message if result.error else None

    # Always execute finally thunk
    finally_op = finally_thunk.get("operation")
    finally_inputs = finally_thunk.get("inputs", {})

    if pass_result_to_finally:
        finally_inputs = {
            **finally_inputs,
            "_main_success": main_success,
            "_main_output": main_output,
            "_main_error": main_error,
        }

    finally_success = False
    finally_error = None

    if finally_op:
        finally_thunk_obj = Thunk.create(operation=finally_op, inputs=finally_inputs)
        finally_result = await runtime.force(finally_thunk_obj)
        finally_success = finally_result.is_success
        if not finally_success:
            finally_error = finally_result.error.message if finally_result.error else "Unknown"

    return {
        "success": main_success,
        "output": main_output,
        "error": main_error,
        "finally_executed": True,
        "finally_success": finally_success,
        "finally_error": finally_error,
    }


@thunk_operation(
    name="thunk.tap",
    description="Execute a thunk and run a side-effect without changing the result.",
    required_capabilities=frozenset(),
)
async def thunk_tap(
    operation: str,
    inputs: dict[str, Any],
    tap_thunk: dict[str, Any],
    tap_on_error: bool = False,
    ignore_tap_errors: bool = True,
) -> dict[str, Any]:
    """Execute with side-effect tap.

    Useful for logging, monitoring, or triggering events without
    affecting the main result.

    Args:
        operation: The main operation to execute.
        inputs: Inputs to the main operation.
        tap_thunk: Side-effect thunk to execute (receives main output).
        tap_on_error: Whether to run tap on error (default False).
        ignore_tap_errors: Whether to ignore tap failures (default True).

    Returns:
        Dict with main result (tap doesn't affect output).

    Example:
        thunk.tap(
            operation="user.create",
            inputs={"email": "test@example.com"},
            tap_thunk={"operation": "log.event", "inputs": {"event": "user_created"}}
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    # Execute main operation
    thunk = Thunk.create(operation=operation, inputs=inputs)
    result = await runtime.force(thunk)

    main_success = result.is_success
    main_output = result.value if main_success else None
    main_error = result.error.message if result.error else None

    # Decide whether to run tap
    should_tap = main_success or tap_on_error

    tap_executed = False
    tap_success = False
    tap_error = None

    if should_tap:
        tap_op = tap_thunk.get("operation")
        tap_inputs = tap_thunk.get("inputs", {})

        if tap_op:
            # Pass main result info to tap
            tap_inputs = {
                **tap_inputs,
                "_tapped_success": main_success,
                "_tapped_output": main_output,
                "_tapped_error": main_error,
            }
            tap_thunk_obj = Thunk.create(operation=tap_op, inputs=tap_inputs)
            tap_result = await runtime.force(tap_thunk_obj)
            tap_executed = True
            tap_success = tap_result.is_success
            if not tap_success:
                tap_error = tap_result.error.message if tap_result.error else "Unknown"

    # Build response - tap doesn't change main result
    response: dict[str, Any] = {
        "success": main_success,
        "output": main_output,
        "tap_executed": tap_executed,
        "tap_success": tap_success,
    }

    if main_error:
        response["error"] = main_error
    if tap_error and not ignore_tap_errors:
        response["tap_error"] = tap_error

    return response


@thunk_operation(
    name="thunk.validate",
    description="Execute a thunk and validate its output against a schema or predicate.",
    required_capabilities=frozenset(),
)
async def thunk_validate(
    operation: str,
    inputs: dict[str, Any],
    schema: dict[str, Any] | None = None,
    validator_thunk: dict[str, Any] | None = None,
    required_keys: list[str] | None = None,
    on_invalid: str = "error",
) -> dict[str, Any]:
    """Execute and validate output.

    Args:
        operation: The operation to execute.
        inputs: Inputs to the operation.
        schema: JSON Schema to validate against (simplified check).
        validator_thunk: Custom validator thunk (receives output, returns success).
        required_keys: List of keys that must exist in output dict.
        on_invalid: What to do on invalid output: "error" or "warn".

    Returns:
        Dict with result and validation status.

    Example:
        thunk.validate(
            operation="api.fetch",
            inputs={"url": "https://api.example.com/data"},
            required_keys=["id", "name", "status"],
            on_invalid="error"
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    # Execute main operation
    thunk = Thunk.create(operation=operation, inputs=inputs)
    result = await runtime.force(thunk)

    if result.is_failure:
        return {
            "success": False,
            "output": None,
            "error": result.error.message if result.error else "Unknown",
            "validated": False,
        }

    output = result.value
    validation_errors: list[str] = []

    # Check required keys
    if required_keys and isinstance(output, dict):
        missing = [k for k in required_keys if k not in output]
        if missing:
            validation_errors.append(f"Missing required keys: {missing}")

    # Check schema (simplified type checking)
    if schema and isinstance(output, dict):
        schema_type = schema.get("type")
        if schema_type == "object":
            if not isinstance(output, dict):
                validation_errors.append(f"Expected object, got {type(output).__name__}")
            else:
                # Check required properties
                required_props = schema.get("required", [])
                missing_props = [p for p in required_props if p not in output]
                if missing_props:
                    validation_errors.append(f"Missing required properties: {missing_props}")

                # Check property types (simplified)
                properties = schema.get("properties", {})
                for prop, prop_schema in properties.items():
                    if prop in output:
                        expected_type = prop_schema.get("type")
                        actual_value = output[prop]
                        if not _check_type(actual_value, expected_type):
                            validation_errors.append(
                                f"Property '{prop}' has wrong type: "
                                f"expected {expected_type}, got {type(actual_value).__name__}"
                            )
        elif schema_type and not _check_type(output, schema_type):
            validation_errors.append(f"Expected {schema_type}, got {type(output).__name__}")

    # Run custom validator
    if validator_thunk and not validation_errors:
        validator_op = validator_thunk.get("operation")
        validator_inputs = validator_thunk.get("inputs", {})
        if validator_op:
            validator_inputs = {**validator_inputs, "_output": output}
            validator_thunk_obj = Thunk.create(operation=validator_op, inputs=validator_inputs)
            validator_result = await runtime.force(validator_thunk_obj)
            if validator_result.is_failure:
                err = validator_result.error.message if validator_result.error else "Unknown"
                validation_errors.append(f"Validator failed: {err}")
            elif validator_result.value:
                # Check if validator returned success indicator
                if isinstance(validator_result.value, dict):
                    if not validator_result.value.get("valid", True):
                        validation_errors.append(
                            validator_result.value.get("reason", "Validation failed")
                        )
                elif validator_result.value is False:
                    validation_errors.append("Validator returned False")

    is_valid = len(validation_errors) == 0

    if is_valid:
        return {
            "success": True,
            "output": output,
            "validated": True,
            "valid": True,
        }

    if on_invalid == "error":
        return {
            "success": False,
            "output": output,
            "validated": True,
            "valid": False,
            "validation_errors": validation_errors,
        }

    # on_invalid == "warn" - return output but flag as invalid
    return {
        "success": True,
        "output": output,
        "validated": True,
        "valid": False,
        "validation_warnings": validation_errors,
    }


def _check_type(value: Any, expected_type: str | None) -> bool:
    """Check if value matches expected JSON Schema type."""
    if expected_type is None:
        return True
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # Unknown type, assume valid
    return isinstance(value, expected)


# =============================================================================
# Data Routing Combinators
# =============================================================================


@thunk_operation(
    name="thunk.zip",
    description="Execute multiple thunks and combine their results pairwise.",
    required_capabilities=frozenset(),
)
async def thunk_zip(
    thunks: list[dict[str, Any]],
    keys: list[str] | None = None,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Zip results from multiple thunks.

    Executes thunks in parallel and combines results into a single structure,
    either as a list (default) or as a dict with provided keys.

    Args:
        thunks: List of thunk specifications to execute.
        keys: Optional keys to use for dict output (must match thunks length).
        fail_fast: If True, fail if any thunk fails.

    Returns:
        Dict with zipped results.

    Example:
        thunk.zip(
            thunks=[
                {"operation": "user.get", "inputs": {"id": 1}},
                {"operation": "user.settings", "inputs": {"id": 1}},
                {"operation": "user.preferences", "inputs": {"id": 1}}
            ],
            keys=["user", "settings", "preferences"]
        )
        # Returns: {"zipped": {"user": {...}, "settings": {...}, "preferences": {...}}}
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    if keys and len(keys) != len(thunks):
        return {
            "success": False,
            "error": f"Keys length ({len(keys)}) must match thunks length ({len(thunks)})",
        }

    # Execute all thunks in parallel
    async def run_one(index: int, spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        operation = spec.get("operation")
        inputs = spec.get("inputs", {})

        if not operation:
            return index, {"status": "failed", "error": "Missing 'operation'"}

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return index, {"status": "failed", "error": error_msg}
        return index, {"status": "success", "output": result.value}

    results_list = await asyncio.gather(*[run_one(i, spec) for i, spec in enumerate(thunks)])
    results = [r[1] for r in sorted(results_list, key=lambda x: x[0])]

    # Check for failures
    failures = [r for r in results if r["status"] == "failed"]
    if failures and fail_fast:
        return {
            "success": False,
            "error": failures[0]["error"],
            "failed_count": len(failures),
            "results": results,
        }

    # Extract outputs
    outputs = [r.get("output") for r in results]

    # Build zipped result
    zipped = dict(zip(keys, outputs, strict=True)) if keys else outputs

    return {
        "success": len(failures) == 0,
        "zipped": zipped,
        "total": len(thunks),
        "succeeded": len(thunks) - len(failures),
        "failed": len(failures),
    }


@thunk_operation(
    name="thunk.fanout",
    description="Run the same input through multiple different thunks.",
    required_capabilities=frozenset(),
)
async def thunk_fanout(
    input_data: dict[str, Any],
    thunks: list[dict[str, Any]],
    input_key: str = "data",
    merge_results: bool = False,
) -> dict[str, Any]:
    """Fan out input to multiple thunks.

    Takes a single input and broadcasts it to multiple thunks for parallel
    processing. Each thunk receives the input under the specified key.

    Args:
        input_data: The data to broadcast to all thunks.
        thunks: List of thunk specifications (operation only, inputs merged).
        input_key: Key name for the input data in each thunk's inputs.
        merge_results: If True, merge all results into single dict.

    Returns:
        Dict with results from all thunks.

    Example:
        thunk.fanout(
            input_data={"text": "Hello world"},
            thunks=[
                {"operation": "text.uppercase"},
                {"operation": "text.reverse"},
                {"operation": "text.wordcount"}
            ],
            input_key="text_data"
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    async def run_one(index: int, spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        operation = spec.get("operation")
        base_inputs = spec.get("inputs", {})
        inputs = {**base_inputs, input_key: input_data}

        if not operation:
            return index, {"status": "failed", "error": "Missing 'operation'"}

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return index, {
                "status": "failed",
                "operation": operation,
                "error": error_msg,
            }
        return index, {
            "status": "success",
            "operation": operation,
            "output": result.value,
        }

    results_list = await asyncio.gather(*[run_one(i, spec) for i, spec in enumerate(thunks)])
    results = [r[1] for r in sorted(results_list, key=lambda x: x[0])]

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - succeeded

    response: dict[str, Any] = {
        "total": len(thunks),
        "succeeded": succeeded,
        "failed": failed,
        "all_success": failed == 0,
        "results": results,
    }

    if merge_results:
        merged: dict[str, Any] = {}
        for r in results:
            if r["status"] == "success" and isinstance(r.get("output"), dict):
                merged.update(r["output"])
        response["merged"] = merged

    return response


@thunk_operation(
    name="thunk.merge",
    description="Execute thunks and merge their dict results into one.",
    required_capabilities=frozenset(),
)
async def thunk_merge(
    thunks: list[dict[str, Any]],
    conflict_strategy: str = "last_wins",
    required_all: bool = False,
) -> dict[str, Any]:
    """Merge results from multiple thunks.

    Executes thunks and merges their dict outputs into a single dict.

    Args:
        thunks: List of thunk specifications.
        conflict_strategy: How to handle key conflicts:
            - "last_wins": Later values overwrite earlier (default)
            - "first_wins": Keep first value seen
            - "collect": Collect all values into a list
        required_all: If True, fail if any thunk fails.

    Returns:
        Dict with merged results.

    Example:
        thunk.merge(
            thunks=[
                {"operation": "config.load", "inputs": {"file": "defaults.yaml"}},
                {"operation": "config.load", "inputs": {"file": "user.yaml"}},
                {"operation": "config.load", "inputs": {"file": "local.yaml"}}
            ],
            conflict_strategy="last_wins"
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    async def run_one(index: int, spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        operation = spec.get("operation")
        inputs = spec.get("inputs", {})

        if not operation:
            return index, {"status": "failed", "error": "Missing 'operation'"}

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return index, {"status": "failed", "error": error_msg}
        return index, {"status": "success", "output": result.value}

    results_list = await asyncio.gather(*[run_one(i, spec) for i, spec in enumerate(thunks)])
    results = [r[1] for r in sorted(results_list, key=lambda x: x[0])]

    failures = [r for r in results if r["status"] == "failed"]
    if failures and required_all:
        return {
            "success": False,
            "error": failures[0]["error"],
            "failed_count": len(failures),
        }

    # Merge outputs
    merged: dict[str, Any] = {}
    conflicts: list[str] = []

    for result in results:
        if result["status"] != "success":
            continue
        output = result.get("output")
        if not isinstance(output, dict):
            continue

        for key, value in output.items():
            if key in merged:
                conflicts.append(key)
                if conflict_strategy == "first_wins":
                    continue  # Keep existing
                elif conflict_strategy == "collect":
                    if not isinstance(merged[key], list):
                        merged[key] = [merged[key]]
                    merged[key].append(value)
                else:  # last_wins (default)
                    merged[key] = value
            else:
                merged[key] = value

    return {
        "success": True,
        "merged": merged,
        "conflicts": list(set(conflicts)),
        "sources": len(thunks),
        "merged_count": len(merged),
    }


@thunk_operation(
    name="thunk.partition",
    description="Split a collection based on a predicate thunk.",
    required_capabilities=frozenset(),
)
async def thunk_partition(
    items: list[Any],
    predicate: dict[str, Any],
    item_key: str = "item",
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Partition a collection based on a predicate.

    Evaluates a predicate thunk for each item and splits the collection
    into items that pass (truthy) and items that fail (falsy).

    Args:
        items: List of items to partition.
        predicate: Thunk specification that returns truthy/falsy value.
        item_key: Key name for the item in predicate inputs.
        max_concurrency: Maximum concurrent predicate evaluations.

    Returns:
        Dict with 'passed' and 'failed' lists.

    Example:
        thunk.partition(
            items=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            predicate={"operation": "math.is_even", "inputs": {}},
            item_key="number"
        )
        # Returns: {"passed": [2, 4, 6, 8, 10], "failed": [1, 3, 5, 7, 9]}
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    predicate_op = predicate.get("operation")
    predicate_base_inputs = predicate.get("inputs", {})

    if not predicate_op:
        return {
            "success": False,
            "error": "Missing 'operation' in predicate",
        }

    async def evaluate(index: int, item: Any) -> tuple[int, Any, bool, str | None]:
        async def run() -> tuple[int, Any, bool, str | None]:
            inputs = {**predicate_base_inputs, item_key: item}
            thunk = Thunk.create(operation=predicate_op, inputs=inputs)
            result = await runtime.force(thunk)

            if result.is_failure:
                error_msg = result.error.message if result.error else "Unknown"
                return index, item, False, error_msg

            # Determine truthiness
            output = result.value
            is_truthy = bool(output)
            if isinstance(output, dict):
                is_truthy = bool(
                    output.get("result", output.get("passed", output.get("success", bool(output))))
                )
            return index, item, is_truthy, None

        if semaphore:
            async with semaphore:
                return await run()
        return await run()

    eval_results = await asyncio.gather(*[evaluate(i, item) for i, item in enumerate(items)])

    passed: list[Any] = []
    failed: list[Any] = []
    errors: list[dict[str, Any]] = []

    for index, item, is_truthy, error in sorted(eval_results, key=lambda x: x[0]):
        if error:
            errors.append({"index": index, "item": item, "error": error})
            failed.append(item)  # Errors go to failed by default
        elif is_truthy:
            passed.append(item)
        else:
            failed.append(item)

    return {
        "success": len(errors) == 0,
        "passed": passed,
        "failed": failed,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "errors": errors if errors else None,
    }


@thunk_operation(
    name="thunk.pipe",
    description="Chain thunks where each output becomes the next input.",
    required_capabilities=frozenset(),
)
async def thunk_pipe(
    thunks: list[dict[str, Any]],
    initial_input: dict[str, Any] | None = None,
    output_key: str = "result",
    input_key: str = "input",
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Pipe data through a chain of thunks.

    Unlike sequence, pipe automatically passes each thunk's output as input
    to the next thunk in the chain.

    Args:
        thunks: List of thunk specifications to chain.
        initial_input: Initial data to pass to first thunk.
        output_key: Key to extract from each output (or None for whole output).
        input_key: Key name to use when passing to next thunk.
        _context: Execution context for cancellation (injected by runtime).

    Returns:
        Dict with final output and pipeline history.

    Example:
        thunk.pipe(
            thunks=[
                {"operation": "text.tokenize"},
                {"operation": "text.filter_stopwords"},
                {"operation": "text.stem"},
            ],
            initial_input={"text": "Hello world"},
            input_key="tokens"
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    current_data: Any = initial_input or {}
    history: list[dict[str, Any]] = []

    for i, thunk_spec in enumerate(thunks):
        # Check for cancellation before each stage
        if _context:
            _context.raise_if_cancelled()

        operation = thunk_spec.get("operation")
        base_inputs = thunk_spec.get("inputs", {})

        if not operation:
            return {
                "success": False,
                "error": f"Missing 'operation' in thunk spec at index {i}",
                "failed_at": i,
                "history": history,
            }

        # Build inputs: merge base inputs with piped data
        if isinstance(current_data, dict):
            inputs = {**current_data, **base_inputs}
        else:
            inputs = {**base_inputs, input_key: current_data}

        thunk = Thunk.create(operation=operation, inputs=inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            history.append(
                {
                    "index": i,
                    "operation": operation,
                    "status": "failed",
                    "error": error_msg,
                }
            )
            return {
                "success": False,
                "error": error_msg,
                "failed_at": i,
                "history": history,
            }

        output = result.value
        history.append(
            {
                "index": i,
                "operation": operation,
                "status": "success",
                "output": output,
            }
        )

        # Pass output to next stage
        current_data = output

    # Extract final output if output_key specified and present
    final_output = current_data
    if isinstance(current_data, dict) and output_key in current_data:
        final_output = current_data[output_key]

    return {
        "success": True,
        "output": final_output,
        "stages": len(thunks),
        "history": history,
    }


@thunk_operation(
    name="thunk.filter",
    description="Filter a collection keeping items that pass a predicate.",
    required_capabilities=frozenset(),
)
async def thunk_filter(
    items: list[Any],
    predicate: dict[str, Any],
    item_key: str = "item",
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Filter items using a predicate thunk.

    Simpler than partition - just returns items that pass the predicate.

    Args:
        items: List of items to filter.
        predicate: Thunk specification that returns truthy/falsy value.
        item_key: Key name for the item in predicate inputs.
        max_concurrency: Maximum concurrent predicate evaluations.

    Returns:
        Dict with filtered items.

    Example:
        thunk.filter(
            items=[1, 2, 3, 4, 5],
            predicate={"operation": "math.is_positive"},
            item_key="number"
        )
    """
    # Reuse partition and just return the passed items
    result = await thunk_partition(
        items=items,
        predicate=predicate,
        item_key=item_key,
        max_concurrency=max_concurrency,
    )

    return {
        "success": result["success"],
        "items": result["passed"],
        "count": result["passed_count"],
        "filtered_out": result["failed_count"],
        "errors": result.get("errors"),
    }


@thunk_operation(
    name="thunk.repeat",
    description="Execute a thunk multiple times, collecting results.",
    required_capabilities=frozenset(),
)
async def thunk_repeat(
    operation: str,
    inputs: dict[str, Any],
    times: int,
    parallel: bool = False,
    stop_on_failure: bool = False,
    include_index: bool = False,
    index_key: str = "iteration",
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Repeat a thunk multiple times.

    Useful for polling, batch operations, or stress testing.

    Args:
        operation: The operation to repeat.
        inputs: Base inputs for each execution.
        times: Number of times to repeat.
        parallel: If True, run all iterations concurrently.
        stop_on_failure: If True, stop on first failure (sequential only).
        include_index: If True, include iteration index in inputs.
        index_key: Key name for the iteration index.
        _context: Execution context for cancellation (injected by runtime).

    Returns:
        Dict with all results.

    Example:
        thunk.repeat(
            operation="api.poll",
            inputs={"endpoint": "/status"},
            times=5,
            parallel=True
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())

    async def run_one(index: int) -> dict[str, Any]:
        iter_inputs = {**inputs}
        if include_index:
            iter_inputs[index_key] = index

        thunk = Thunk.create(operation=operation, inputs=iter_inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return {"index": index, "status": "failed", "error": error_msg}
        return {"index": index, "status": "success", "output": result.value}

    if parallel:
        results = await asyncio.gather(*[run_one(i) for i in range(times)])
        results = list(results)
    else:
        results = []
        for i in range(times):
            # Check for cancellation before each iteration (sequential only)
            if _context:
                _context.raise_if_cancelled()
            result = await run_one(i)
            results.append(result)
            if stop_on_failure and result["status"] == "failed":
                break

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - succeeded

    return {
        "success": failed == 0,
        "total": times,
        "completed": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@thunk_operation(
    name="thunk.while",
    description="Execute a thunk repeatedly while a condition is true.",
    required_capabilities=frozenset(),
)
async def thunk_while(
    condition: dict[str, Any],
    body: dict[str, Any],
    max_iterations: int = 100,
    pass_output_to_condition: bool = False,
    pass_output_to_body: bool = False,
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Loop while condition is true.

    Evaluates condition first, then executes body if true. Repeats until
    condition is false or max_iterations reached.

    Args:
        condition: Thunk that returns truthy/falsy value.
        body: Thunk to execute each iteration.
        max_iterations: Safety limit to prevent infinite loops.
        pass_output_to_condition: Pass previous body output to condition.
        pass_output_to_body: Pass previous body output to next body execution.
        _context: Execution context for cancellation (injected by runtime).

    Returns:
        Dict with all iteration results.

    Example:
        thunk.while(
            condition={"operation": "queue.has_items", "inputs": {"queue": "tasks"}},
            body={"operation": "queue.process_one", "inputs": {"queue": "tasks"}},
            max_iterations=1000
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    iterations: list[dict[str, Any]] = []
    last_output: Any = None

    async def check_condition() -> bool:
        cond_op = condition.get("operation")
        cond_inputs = condition.get("inputs", {})

        if not cond_op:
            return False

        if pass_output_to_condition and last_output is not None:
            if isinstance(last_output, dict):
                cond_inputs = {**cond_inputs, **last_output}
            else:
                cond_inputs = {**cond_inputs, "_previous": last_output}

        thunk = Thunk.create(operation=cond_op, inputs=cond_inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            return False

        output = result.value
        if isinstance(output, dict):
            return bool(output.get("result", output.get("continue", output.get("success", True))))
        return bool(output)

    async def run_body() -> tuple[bool, Any, str | None]:
        body_op = body.get("operation")
        body_inputs = body.get("inputs", {})

        if not body_op:
            return False, None, "Missing 'operation' in body"

        if pass_output_to_body and last_output is not None:
            if isinstance(last_output, dict):
                body_inputs = {**body_inputs, **last_output}
            else:
                body_inputs = {**body_inputs, "_previous": last_output}

        thunk = Thunk.create(operation=body_op, inputs=body_inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return False, None, error_msg
        return True, result.value, None

    for i in range(max_iterations):
        # Check for cancellation before each iteration
        if _context:
            _context.raise_if_cancelled()

        # Check condition
        should_continue = await check_condition()
        if not should_continue:
            break

        # Run body
        success, output, error = await run_body()
        iterations.append(
            {
                "iteration": i,
                "status": "success" if success else "failed",
                "output": output,
                "error": error,
            }
        )

        if not success:
            return {
                "success": False,
                "iterations": len(iterations),
                "max_reached": False,
                "error": error,
                "results": iterations,
            }

        last_output = output

    max_reached = len(iterations) >= max_iterations

    return {
        "success": True,
        "iterations": len(iterations),
        "max_reached": max_reached,
        "final_output": last_output,
        "results": iterations,
    }


@thunk_operation(
    name="thunk.until",
    description="Execute a thunk repeatedly until a condition becomes true.",
    required_capabilities=frozenset(),
)
async def thunk_until(
    condition: dict[str, Any],
    body: dict[str, Any],
    max_iterations: int = 100,
    check_before: bool = False,
    pass_output_to_condition: bool = True,
    _context: ThunkContext | None = None,
) -> dict[str, Any]:
    """Loop until condition becomes true.

    By default, executes body first then checks condition (do-until).
    Set check_before=True for while-not semantics.

    Args:
        condition: Thunk that returns truthy when loop should stop.
        body: Thunk to execute each iteration.
        max_iterations: Safety limit to prevent infinite loops.
        check_before: If True, check condition before body (while-not).
        pass_output_to_condition: Pass body output to condition check.
        _context: Execution context for cancellation (injected by runtime).

    Returns:
        Dict with all iteration results.

    Example:
        thunk.until(
            condition={"operation": "job.is_complete", "inputs": {"job_id": "123"}},
            body={"operation": "job.check_status", "inputs": {"job_id": "123"}},
            max_iterations=60
        )
    """
    from crows_nest.core.runtime import ThunkRuntime

    runtime = ThunkRuntime(get_global_registry())
    iterations: list[dict[str, Any]] = []
    last_output: Any = None

    async def check_condition() -> bool:
        cond_op = condition.get("operation")
        cond_inputs = condition.get("inputs", {})

        if not cond_op:
            return True  # Stop if no condition

        if pass_output_to_condition and last_output is not None:
            if isinstance(last_output, dict):
                cond_inputs = {**cond_inputs, **last_output}
            else:
                cond_inputs = {**cond_inputs, "_result": last_output}

        thunk = Thunk.create(operation=cond_op, inputs=cond_inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            return True  # Stop on condition error

        output = result.value
        if isinstance(output, dict):
            # Check common "done" indicators
            return bool(
                output.get(
                    "done",
                    output.get("complete", output.get("result", output.get("success", False))),
                )
            )
        return bool(output)

    async def run_body() -> tuple[bool, Any, str | None]:
        body_op = body.get("operation")
        body_inputs = body.get("inputs", {})

        if not body_op:
            return False, None, "Missing 'operation' in body"

        thunk = Thunk.create(operation=body_op, inputs=body_inputs)
        result = await runtime.force(thunk, context=_context)

        if result.is_failure:
            error_msg = result.error.message if result.error else "Unknown error"
            return False, None, error_msg
        return True, result.value, None

    for i in range(max_iterations):
        # Check for cancellation before each iteration
        if _context:
            _context.raise_if_cancelled()

        # Check before (while-not style)
        if check_before:
            should_stop = await check_condition()
            if should_stop:
                break

        # Run body
        success, output, error = await run_body()
        last_output = output

        iterations.append(
            {
                "iteration": i,
                "status": "success" if success else "failed",
                "output": output,
                "error": error,
            }
        )

        if not success:
            return {
                "success": False,
                "condition_met": False,
                "iterations": len(iterations),
                "max_reached": False,
                "error": error,
                "results": iterations,
            }

        # Check after (do-until style)
        if not check_before:
            should_stop = await check_condition()
            if should_stop:
                return {
                    "success": True,
                    "condition_met": True,
                    "iterations": len(iterations),
                    "max_reached": False,
                    "final_output": last_output,
                    "results": iterations,
                }

    max_reached = len(iterations) >= max_iterations

    return {
        "success": not max_reached,  # If we hit max, condition wasn't met
        "condition_met": False,
        "iterations": len(iterations),
        "max_reached": max_reached,
        "final_output": last_output,
        "results": iterations,
    }
