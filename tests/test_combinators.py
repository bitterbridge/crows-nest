"""Tests for thunk combinators."""

from __future__ import annotations

import pytest

from crows_nest.core.combinators import (
    thunk_catch,
    thunk_conditional,
    thunk_ensure,
    thunk_fallback,
    thunk_fanout,
    thunk_filter,
    thunk_map,
    thunk_merge,
    thunk_parallel,
    thunk_partition,
    thunk_pipe,
    thunk_race,
    thunk_reduce,
    thunk_repeat,
    thunk_retry,
    thunk_sequence,
    thunk_tap,
    thunk_timeout,
    thunk_until,
    thunk_validate,
    thunk_while,
    thunk_zip,
)
from crows_nest.core.registry import get_global_registry, thunk_operation


@pytest.fixture
def _register_test_ops():
    """Register test operations for combinator testing."""
    registry = get_global_registry()

    @thunk_operation(
        name="test.echo",
        description="Echo the input value.",
        required_capabilities=frozenset(),
    )
    async def echo(value: str) -> dict:
        return {"echoed": value}

    @thunk_operation(
        name="test.add",
        description="Add two numbers.",
        required_capabilities=frozenset(),
    )
    async def add(a: int, b: int) -> dict:
        return {"result": a + b}

    @thunk_operation(
        name="test.fail",
        description="Always fails.",
        required_capabilities=frozenset(),
    )
    async def fail(message: str = "error") -> dict:
        raise ValueError(message)

    @thunk_operation(
        name="test.slow",
        description="Slow operation for timeout testing.",
        required_capabilities=frozenset(),
    )
    async def slow(delay: float = 1.0) -> dict:
        import asyncio

        await asyncio.sleep(delay)
        return {"completed": True}

    @thunk_operation(
        name="test.accumulate",
        description="Accumulator for reduce testing.",
        required_capabilities=frozenset(),
    )
    async def accumulate(accumulator: int, item: int) -> dict:
        return {"result": accumulator + item}

    @thunk_operation(
        name="test.check_flag",
        description="Returns success based on flag.",
        required_capabilities=frozenset(),
    )
    async def check_flag(flag: bool) -> dict:
        return {"success": flag}

    @thunk_operation(
        name="test.noop",
        description="Does nothing, returns empty dict.",
        required_capabilities=frozenset(),
    )
    async def noop(**kwargs) -> dict:
        return {"noop": True, "received": kwargs}

    @thunk_operation(
        name="test.recover",
        description="Recovery handler for catch testing.",
        required_capabilities=frozenset(),
    )
    async def recover(_error: str = "", _error_type: str = "", **kwargs) -> dict:
        return {"recovered": True, "from_error": _error}

    @thunk_operation(
        name="test.validator",
        description="Validates output for validate testing.",
        required_capabilities=frozenset(),
    )
    async def validator(_output: dict | None = None, require_key: str = "") -> dict:
        if _output and require_key and require_key not in _output:
            return {"valid": False, "reason": f"Missing key: {require_key}"}
        return {"valid": True}

    @thunk_operation(
        name="test.get_user",
        description="Simulates getting user data.",
        required_capabilities=frozenset(),
    )
    async def get_user(id: int) -> dict:
        return {"id": id, "name": f"User{id}"}

    @thunk_operation(
        name="test.get_settings",
        description="Simulates getting settings.",
        required_capabilities=frozenset(),
    )
    async def get_settings(id: int) -> dict:
        return {"theme": "dark", "language": "en"}

    @thunk_operation(
        name="test.transform",
        description="Transform input data.",
        required_capabilities=frozenset(),
    )
    async def transform(data: dict, prefix: str = "") -> dict:
        return {"transformed": True, "prefix": prefix, "original": data}

    @thunk_operation(
        name="test.is_even",
        description="Check if number is even.",
        required_capabilities=frozenset(),
    )
    async def is_even(item: int) -> dict:
        return {"result": item % 2 == 0}

    @thunk_operation(
        name="test.config_a",
        description="Returns config section A.",
        required_capabilities=frozenset(),
    )
    async def config_a(**kwargs) -> dict:
        return {"setting1": "a", "shared": "from_a"}

    @thunk_operation(
        name="test.config_b",
        description="Returns config section B.",
        required_capabilities=frozenset(),
    )
    async def config_b(**kwargs) -> dict:
        return {"setting2": "b", "shared": "from_b"}

    @thunk_operation(
        name="test.double",
        description="Double a number.",
        required_capabilities=frozenset(),
    )
    async def double(value: int | None = None, result: int | None = None, **kwargs) -> dict:
        # Accept either 'value' or 'result' as input for pipe compatibility
        num = value if value is not None else (result if result is not None else 0)
        return {"result": num * 2}

    @thunk_operation(
        name="test.add_one",
        description="Add one to a number.",
        required_capabilities=frozenset(),
    )
    async def add_one(result: int = 0, **kwargs) -> dict:
        return {"result": result + 1}

    # Stateful counter for loop testing
    _counter = {"value": 0}

    @thunk_operation(
        name="test.counter_increment",
        description="Increment counter and return value.",
        required_capabilities=frozenset(),
    )
    async def counter_increment(**kwargs) -> dict:
        _counter["value"] += 1
        return {"count": _counter["value"]}

    @thunk_operation(
        name="test.counter_check",
        description="Check if counter reached target.",
        required_capabilities=frozenset(),
    )
    async def counter_check(target: int = 5, count: int = 0, **kwargs) -> dict:
        return {"result": count >= target, "done": count >= target}

    @thunk_operation(
        name="test.counter_reset",
        description="Reset counter to zero.",
        required_capabilities=frozenset(),
    )
    async def counter_reset(**kwargs) -> dict:
        _counter["value"] = 0
        return {"reset": True}

    @thunk_operation(
        name="test.should_continue",
        description="Return whether to continue looping.",
        required_capabilities=frozenset(),
    )
    async def should_continue(count: int = 0, limit: int = 3, **kwargs) -> dict:
        return {"result": count < limit, "continue": count < limit}

    yield

    # Cleanup
    for op in [
        "test.echo",
        "test.add",
        "test.fail",
        "test.slow",
        "test.accumulate",
        "test.check_flag",
        "test.noop",
        "test.recover",
        "test.validator",
        "test.get_user",
        "test.get_settings",
        "test.transform",
        "test.is_even",
        "test.config_a",
        "test.config_b",
        "test.double",
        "test.add_one",
        "test.counter_increment",
        "test.counter_check",
        "test.counter_reset",
        "test.should_continue",
    ]:
        registry.unregister(op)


class TestThunkSequence:
    """Tests for thunk.sequence combinator."""

    @pytest.mark.asyncio
    async def test_simple_sequence(self, _register_test_ops):
        """Should execute thunks in order."""
        result = await thunk_sequence(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "first"}},
                {"operation": "test.echo", "inputs": {"value": "second"}},
            ]
        )
        assert result["success"] is True
        assert result["completed"] == 2
        assert result["results"][0]["output"]["echoed"] == "first"
        assert result["results"][1]["output"]["echoed"] == "second"

    @pytest.mark.asyncio
    async def test_sequence_stops_on_failure(self, _register_test_ops):
        """Should stop on first failure by default."""
        result = await thunk_sequence(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "ok"}},
                {"operation": "test.fail", "inputs": {"message": "boom"}},
                {"operation": "test.echo", "inputs": {"value": "never"}},
            ]
        )
        assert result["success"] is False
        assert result["completed"] == 2  # Stopped at failure
        assert result["results"][1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_sequence_continues_on_failure(self, _register_test_ops):
        """Should continue on failure when stop_on_failure=False."""
        result = await thunk_sequence(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "ok"}},
                {"operation": "test.fail", "inputs": {"message": "boom"}},
                {"operation": "test.echo", "inputs": {"value": "after"}},
            ],
            stop_on_failure=False,
        )
        assert result["completed"] == 3


class TestThunkParallel:
    """Tests for thunk.parallel combinator."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, _register_test_ops):
        """Should execute thunks concurrently."""
        result = await thunk_parallel(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "a"}},
                {"operation": "test.echo", "inputs": {"value": "b"}},
                {"operation": "test.echo", "inputs": {"value": "c"}},
            ]
        )
        assert result["total"] == 3
        assert result["succeeded"] == 3
        assert result["failed"] == 0
        assert result["all_success"] is True

    @pytest.mark.asyncio
    async def test_parallel_with_failure(self, _register_test_ops):
        """Should collect failures without stopping others."""
        result = await thunk_parallel(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "ok"}},
                {"operation": "test.fail", "inputs": {"message": "boom"}},
                {"operation": "test.echo", "inputs": {"value": "also ok"}},
            ]
        )
        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert result["all_success"] is False


class TestThunkRace:
    """Tests for thunk.race combinator."""

    @pytest.mark.asyncio
    async def test_race_returns_first_success(self, _register_test_ops):
        """Should return first successful result."""
        result = await thunk_race(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "fast"}},
                {"operation": "test.slow", "inputs": {"delay": 10.0}},
            ]
        )
        assert result["winner"] == 0
        assert result["timed_out"] is False
        assert result["result"]["output"]["echoed"] == "fast"

    @pytest.mark.asyncio
    async def test_race_all_fail(self, _register_test_ops):
        """Should return all errors when all fail."""
        result = await thunk_race(
            thunks=[
                {"operation": "test.fail", "inputs": {"message": "err1"}},
                {"operation": "test.fail", "inputs": {"message": "err2"}},
            ]
        )
        assert result["winner"] is None
        assert len(result["errors"]) == 2


class TestThunkRetry:
    """Tests for thunk.retry combinator."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_try(self, _register_test_ops):
        """Should succeed on first attempt."""
        result = await thunk_retry(
            operation="test.echo",
            inputs={"value": "hello"},
            max_attempts=3,
        )
        assert result["success"] is True
        assert result["attempts"] == 1
        assert result["output"]["echoed"] == "hello"

    @pytest.mark.asyncio
    async def test_retry_fails_all_attempts(self, _register_test_ops):
        """Should fail after max attempts."""
        result = await thunk_retry(
            operation="test.fail",
            inputs={"message": "always fails"},
            max_attempts=3,
            base_delay_seconds=0.01,  # Fast for testing
        )
        assert result["success"] is False
        assert result["attempts"] == 3
        assert len(result["history"]) == 3


class TestThunkTimeout:
    """Tests for thunk.timeout combinator."""

    @pytest.mark.asyncio
    async def test_completes_before_timeout(self, _register_test_ops):
        """Should return result when completed in time."""
        result = await thunk_timeout(
            operation="test.echo",
            inputs={"value": "quick"},
            timeout_seconds=5.0,
        )
        assert result["timed_out"] is False
        assert result["output"]["echoed"] == "quick"

    @pytest.mark.asyncio
    async def test_times_out(self, _register_test_ops):
        """Should timeout on slow operations."""
        result = await thunk_timeout(
            operation="test.slow",
            inputs={"delay": 10.0},
            timeout_seconds=0.1,
            default={"fallback": True},
        )
        assert result["timed_out"] is True
        assert result["output"]["fallback"] is True


class TestThunkFallback:
    """Tests for thunk.fallback combinator."""

    @pytest.mark.asyncio
    async def test_uses_primary(self, _register_test_ops):
        """Should use primary when it succeeds."""
        result = await thunk_fallback(
            primary={"operation": "test.echo", "inputs": {"value": "primary"}},
            fallback={"operation": "test.echo", "inputs": {"value": "fallback"}},
        )
        assert result["used"] == "primary"
        assert result["output"]["echoed"] == "primary"

    @pytest.mark.asyncio
    async def test_uses_fallback_on_error(self, _register_test_ops):
        """Should use fallback when primary fails."""
        result = await thunk_fallback(
            primary={"operation": "test.fail", "inputs": {"message": "oops"}},
            fallback={"operation": "test.echo", "inputs": {"value": "backup"}},
        )
        assert result["used"] == "fallback"
        assert result["output"]["echoed"] == "backup"
        assert "primary_error" in result


class TestThunkConditional:
    """Tests for thunk.conditional combinator."""

    @pytest.mark.asyncio
    async def test_executes_then_branch(self, _register_test_ops):
        """Should execute then branch when condition is true."""
        result = await thunk_conditional(
            condition={"operation": "test.check_flag", "inputs": {"flag": True}},
            then_thunk={"operation": "test.echo", "inputs": {"value": "yes"}},
            else_thunk={"operation": "test.echo", "inputs": {"value": "no"}},
        )
        assert result["branch"] == "then"
        assert result["output"]["echoed"] == "yes"

    @pytest.mark.asyncio
    async def test_executes_else_branch(self, _register_test_ops):
        """Should execute else branch when condition is false."""
        result = await thunk_conditional(
            condition={"operation": "test.check_flag", "inputs": {"flag": False}},
            then_thunk={"operation": "test.echo", "inputs": {"value": "yes"}},
            else_thunk={"operation": "test.echo", "inputs": {"value": "no"}},
        )
        assert result["branch"] == "else"
        assert result["output"]["echoed"] == "no"

    @pytest.mark.asyncio
    async def test_skips_when_no_else(self, _register_test_ops):
        """Should skip when condition is false and no else branch."""
        result = await thunk_conditional(
            condition={"operation": "test.check_flag", "inputs": {"flag": False}},
            then_thunk={"operation": "test.echo", "inputs": {"value": "yes"}},
        )
        assert result["branch"] == "skipped"
        assert result["output"] is None


class TestThunkMap:
    """Tests for thunk.map combinator."""

    @pytest.mark.asyncio
    async def test_maps_over_items(self, _register_test_ops):
        """Should apply operation to each item."""
        result = await thunk_map(
            operation="test.echo",
            items=["a", "b", "c"],
            input_key="value",
        )
        assert result["total"] == 3
        assert result["succeeded"] == 3
        assert result["results"][0]["output"]["echoed"] == "a"
        assert result["results"][1]["output"]["echoed"] == "b"
        assert result["results"][2]["output"]["echoed"] == "c"

    @pytest.mark.asyncio
    async def test_map_includes_items_in_results(self, _register_test_ops):
        """Should include original items in results."""
        result = await thunk_map(
            operation="test.echo",
            items=["x", "y"],
            input_key="value",
        )
        assert result["results"][0]["item"] == "x"
        assert result["results"][1]["item"] == "y"


class TestThunkReduce:
    """Tests for thunk.reduce combinator."""

    @pytest.mark.asyncio
    async def test_reduces_collection(self, _register_test_ops):
        """Should reduce items with accumulator."""
        result = await thunk_reduce(
            operation="test.accumulate",
            items=[1, 2, 3, 4, 5],
            initial=0,
            accumulator_key="accumulator",
            item_key="item",
        )
        assert result["success"] is True
        assert result["result"] == 15  # 0 + 1 + 2 + 3 + 4 + 5

    @pytest.mark.asyncio
    async def test_reduce_empty_list(self, _register_test_ops):
        """Should return initial value for empty list."""
        result = await thunk_reduce(
            operation="test.accumulate",
            items=[],
            initial=42,
        )
        assert result["success"] is True
        assert result["result"] == 42


class TestThunkCatch:
    """Tests for thunk.catch combinator."""

    @pytest.mark.asyncio
    async def test_catch_success_no_error(self, _register_test_ops):
        """Should pass through success without catching."""
        result = await thunk_catch(
            operation="test.echo",
            inputs={"value": "hello"},
            default={"fallback": True},
        )
        assert result["caught"] is False
        assert result["output"]["echoed"] == "hello"

    @pytest.mark.asyncio
    async def test_catch_error_with_default(self, _register_test_ops):
        """Should catch error and return default."""
        result = await thunk_catch(
            operation="test.fail",
            inputs={"message": "oops"},
            default={"fallback": True},
        )
        assert result["caught"] is True
        assert result["output"]["fallback"] is True
        assert "original_error" in result

    @pytest.mark.asyncio
    async def test_catch_error_with_handler(self, _register_test_ops):
        """Should catch error and run recovery handler."""
        result = await thunk_catch(
            operation="test.fail",
            inputs={"message": "oops"},
            error_handler={"operation": "test.recover", "inputs": {}},
        )
        assert result["caught"] is True
        assert result["recovered"] is True
        assert result["output"]["recovered"] is True

    @pytest.mark.asyncio
    async def test_catch_selective_error_types(self, _register_test_ops):
        """Should only catch specified error types."""
        result = await thunk_catch(
            operation="test.fail",
            inputs={"message": "oops"},
            catch_types=["NetworkError"],  # Won't match ValueError
            default={"fallback": True},
        )
        # Should propagate since ValueError not in catch_types
        assert result["caught"] is False
        assert result["propagated"] is True


class TestThunkEnsure:
    """Tests for thunk.ensure combinator."""

    @pytest.mark.asyncio
    async def test_ensure_runs_finally_on_success(self, _register_test_ops):
        """Should run finally thunk even on success."""
        result = await thunk_ensure(
            operation="test.echo",
            inputs={"value": "main"},
            finally_thunk={"operation": "test.noop", "inputs": {}},
        )
        assert result["success"] is True
        assert result["output"]["echoed"] == "main"
        assert result["finally_executed"] is True
        assert result["finally_success"] is True

    @pytest.mark.asyncio
    async def test_ensure_runs_finally_on_failure(self, _register_test_ops):
        """Should run finally thunk even on failure."""
        result = await thunk_ensure(
            operation="test.fail",
            inputs={"message": "boom"},
            finally_thunk={"operation": "test.noop", "inputs": {}},
        )
        assert result["success"] is False
        assert result["error"] is not None
        assert result["finally_executed"] is True
        assert result["finally_success"] is True

    @pytest.mark.asyncio
    async def test_ensure_passes_result_to_finally(self, _register_test_ops):
        """Should pass main result to finally thunk when requested."""
        result = await thunk_ensure(
            operation="test.echo",
            inputs={"value": "main"},
            finally_thunk={"operation": "test.noop", "inputs": {}},
            pass_result_to_finally=True,
        )
        assert result["success"] is True
        assert result["finally_executed"] is True


class TestThunkTap:
    """Tests for thunk.tap combinator."""

    @pytest.mark.asyncio
    async def test_tap_runs_side_effect(self, _register_test_ops):
        """Should run tap without affecting main result."""
        result = await thunk_tap(
            operation="test.echo",
            inputs={"value": "main"},
            tap_thunk={"operation": "test.noop", "inputs": {}},
        )
        assert result["success"] is True
        assert result["output"]["echoed"] == "main"
        assert result["tap_executed"] is True
        assert result["tap_success"] is True

    @pytest.mark.asyncio
    async def test_tap_not_run_on_error_by_default(self, _register_test_ops):
        """Should not run tap on error by default."""
        result = await thunk_tap(
            operation="test.fail",
            inputs={"message": "boom"},
            tap_thunk={"operation": "test.noop", "inputs": {}},
        )
        assert result["success"] is False
        assert result["tap_executed"] is False

    @pytest.mark.asyncio
    async def test_tap_run_on_error_when_configured(self, _register_test_ops):
        """Should run tap on error when tap_on_error=True."""
        result = await thunk_tap(
            operation="test.fail",
            inputs={"message": "boom"},
            tap_thunk={"operation": "test.noop", "inputs": {}},
            tap_on_error=True,
        )
        assert result["success"] is False
        assert result["tap_executed"] is True

    @pytest.mark.asyncio
    async def test_tap_failure_ignored_by_default(self, _register_test_ops):
        """Should ignore tap failures by default."""
        result = await thunk_tap(
            operation="test.echo",
            inputs={"value": "main"},
            tap_thunk={"operation": "test.fail", "inputs": {"message": "tap failed"}},
        )
        assert result["success"] is True
        assert result["output"]["echoed"] == "main"
        assert result["tap_executed"] is True
        assert result["tap_success"] is False
        assert "tap_error" not in result  # Ignored


class TestThunkValidate:
    """Tests for thunk.validate combinator."""

    @pytest.mark.asyncio
    async def test_validate_passes_valid_output(self, _register_test_ops):
        """Should pass through valid output."""
        result = await thunk_validate(
            operation="test.echo",
            inputs={"value": "hello"},
            required_keys=["echoed"],
        )
        assert result["success"] is True
        assert result["valid"] is True
        assert result["output"]["echoed"] == "hello"

    @pytest.mark.asyncio
    async def test_validate_fails_missing_keys(self, _register_test_ops):
        """Should fail when required keys are missing."""
        result = await thunk_validate(
            operation="test.echo",
            inputs={"value": "hello"},
            required_keys=["missing_key"],
            on_invalid="error",
        )
        assert result["success"] is False
        assert result["valid"] is False
        assert "validation_errors" in result

    @pytest.mark.asyncio
    async def test_validate_warns_instead_of_error(self, _register_test_ops):
        """Should warn instead of fail when on_invalid='warn'."""
        result = await thunk_validate(
            operation="test.echo",
            inputs={"value": "hello"},
            required_keys=["missing_key"],
            on_invalid="warn",
        )
        assert result["success"] is True  # Still succeeds
        assert result["valid"] is False
        assert "validation_warnings" in result

    @pytest.mark.asyncio
    async def test_validate_with_schema(self, _register_test_ops):
        """Should validate against JSON schema."""
        result = await thunk_validate(
            operation="test.echo",
            inputs={"value": "hello"},
            schema={
                "type": "object",
                "required": ["echoed"],
                "properties": {"echoed": {"type": "string"}},
            },
        )
        assert result["success"] is True
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_schema_type_mismatch(self, _register_test_ops):
        """Should fail when schema type doesn't match."""
        result = await thunk_validate(
            operation="test.add",
            inputs={"a": 1, "b": 2},
            schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},  # Actually int
            },
        )
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_with_custom_validator(self, _register_test_ops):
        """Should run custom validator thunk."""
        result = await thunk_validate(
            operation="test.echo",
            inputs={"value": "hello"},
            validator_thunk={
                "operation": "test.validator",
                "inputs": {"require_key": "echoed"},
            },
        )
        assert result["success"] is True
        assert result["valid"] is True


class TestThunkZip:
    """Tests for thunk.zip combinator."""

    @pytest.mark.asyncio
    async def test_zip_multiple_thunks(self, _register_test_ops):
        """Should zip results from multiple thunks."""
        result = await thunk_zip(
            thunks=[
                {"operation": "test.get_user", "inputs": {"id": 1}},
                {"operation": "test.get_settings", "inputs": {"id": 1}},
            ]
        )
        assert result["success"] is True
        assert result["total"] == 2
        assert result["succeeded"] == 2
        # Without keys, results are in a list
        assert isinstance(result["zipped"], list)
        assert len(result["zipped"]) == 2

    @pytest.mark.asyncio
    async def test_zip_with_keys(self, _register_test_ops):
        """Should zip results into dict with keys."""
        result = await thunk_zip(
            thunks=[
                {"operation": "test.get_user", "inputs": {"id": 1}},
                {"operation": "test.get_settings", "inputs": {"id": 1}},
            ],
            keys=["user", "settings"],
        )
        assert result["success"] is True
        assert isinstance(result["zipped"], dict)
        assert "user" in result["zipped"]
        assert "settings" in result["zipped"]
        assert result["zipped"]["user"]["name"] == "User1"

    @pytest.mark.asyncio
    async def test_zip_fails_on_error(self, _register_test_ops):
        """Should fail when any thunk fails (fail_fast=True)."""
        result = await thunk_zip(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "ok"}},
                {"operation": "test.fail", "inputs": {"message": "boom"}},
            ],
            fail_fast=True,
        )
        assert result["success"] is False
        assert result["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_zip_keys_length_mismatch(self, _register_test_ops):
        """Should error when keys length doesn't match thunks."""
        result = await thunk_zip(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "a"}},
                {"operation": "test.echo", "inputs": {"value": "b"}},
            ],
            keys=["only_one"],
        )
        assert result["success"] is False
        assert "length" in result["error"].lower()


class TestThunkFanout:
    """Tests for thunk.fanout combinator."""

    @pytest.mark.asyncio
    async def test_fanout_broadcasts_input(self, _register_test_ops):
        """Should broadcast same input to multiple thunks."""
        result = await thunk_fanout(
            input_data={"text": "hello"},
            thunks=[
                {"operation": "test.transform", "inputs": {"prefix": "a"}},
                {"operation": "test.transform", "inputs": {"prefix": "b"}},
            ],
            input_key="data",
        )
        assert result["all_success"] is True
        assert result["total"] == 2
        assert result["results"][0]["output"]["prefix"] == "a"
        assert result["results"][1]["output"]["prefix"] == "b"
        # Both should have received the same input
        assert result["results"][0]["output"]["original"] == {"text": "hello"}
        assert result["results"][1]["output"]["original"] == {"text": "hello"}

    @pytest.mark.asyncio
    async def test_fanout_with_merge(self, _register_test_ops):
        """Should optionally merge all results."""
        result = await thunk_fanout(
            input_data={"x": 1},
            thunks=[
                {"operation": "test.config_a"},
                {"operation": "test.config_b"},
            ],
            input_key="data",
            merge_results=True,
        )
        assert result["all_success"] is True
        assert "merged" in result
        # Merged should have keys from both configs
        assert "setting1" in result["merged"]
        assert "setting2" in result["merged"]

    @pytest.mark.asyncio
    async def test_fanout_partial_failure(self, _register_test_ops):
        """Should continue even if some thunks fail."""
        result = await thunk_fanout(
            input_data={"x": 1},
            thunks=[
                {"operation": "test.config_a"},  # Will succeed
                {"operation": "test.fail", "inputs": {"message": "boom"}},
            ],
            input_key="data",
        )
        assert result["all_success"] is False
        assert result["succeeded"] == 1
        assert result["failed"] == 1


class TestThunkMerge:
    """Tests for thunk.merge combinator."""

    @pytest.mark.asyncio
    async def test_merge_multiple_dicts(self, _register_test_ops):
        """Should merge dicts from multiple thunks."""
        result = await thunk_merge(
            thunks=[
                {"operation": "test.config_a"},
                {"operation": "test.config_b"},
            ]
        )
        assert result["success"] is True
        assert "setting1" in result["merged"]
        assert "setting2" in result["merged"]

    @pytest.mark.asyncio
    async def test_merge_last_wins(self, _register_test_ops):
        """Should use last value for conflicts (default)."""
        result = await thunk_merge(
            thunks=[
                {"operation": "test.config_a"},  # has shared: "from_a"
                {"operation": "test.config_b"},  # has shared: "from_b"
            ],
            conflict_strategy="last_wins",
        )
        assert result["success"] is True
        assert result["merged"]["shared"] == "from_b"
        assert "shared" in result["conflicts"]

    @pytest.mark.asyncio
    async def test_merge_first_wins(self, _register_test_ops):
        """Should keep first value for conflicts."""
        result = await thunk_merge(
            thunks=[
                {"operation": "test.config_a"},  # has shared: "from_a"
                {"operation": "test.config_b"},  # has shared: "from_b"
            ],
            conflict_strategy="first_wins",
        )
        assert result["success"] is True
        assert result["merged"]["shared"] == "from_a"

    @pytest.mark.asyncio
    async def test_merge_collect_strategy(self, _register_test_ops):
        """Should collect conflicting values into list."""
        result = await thunk_merge(
            thunks=[
                {"operation": "test.config_a"},
                {"operation": "test.config_b"},
            ],
            conflict_strategy="collect",
        )
        assert result["success"] is True
        assert result["merged"]["shared"] == ["from_a", "from_b"]


class TestThunkPartition:
    """Tests for thunk.partition combinator."""

    @pytest.mark.asyncio
    async def test_partition_by_predicate(self, _register_test_ops):
        """Should partition items by predicate."""
        result = await thunk_partition(
            items=[1, 2, 3, 4, 5, 6],
            predicate={"operation": "test.is_even", "inputs": {}},
            item_key="item",
        )
        assert result["success"] is True
        assert result["passed"] == [2, 4, 6]
        assert result["failed"] == [1, 3, 5]
        assert result["passed_count"] == 3
        assert result["failed_count"] == 3

    @pytest.mark.asyncio
    async def test_partition_empty_list(self, _register_test_ops):
        """Should handle empty list."""
        result = await thunk_partition(
            items=[],
            predicate={"operation": "test.is_even", "inputs": {}},
        )
        assert result["success"] is True
        assert result["passed"] == []
        assert result["failed"] == []

    @pytest.mark.asyncio
    async def test_partition_all_pass(self, _register_test_ops):
        """Should handle all items passing."""
        result = await thunk_partition(
            items=[2, 4, 6],
            predicate={"operation": "test.is_even", "inputs": {}},
            item_key="item",
        )
        assert result["success"] is True
        assert result["passed"] == [2, 4, 6]
        assert result["failed"] == []

    @pytest.mark.asyncio
    async def test_partition_missing_predicate_operation(self, _register_test_ops):
        """Should error when predicate has no operation."""
        result = await thunk_partition(
            items=[1, 2, 3],
            predicate={"inputs": {}},  # Missing operation
        )
        assert result["success"] is False
        assert "operation" in result["error"].lower()


class TestThunkPipe:
    """Tests for thunk.pipe combinator."""

    @pytest.mark.asyncio
    async def test_pipe_chains_output_to_input(self, _register_test_ops):
        """Should pass output of each stage to next stage."""
        result = await thunk_pipe(
            thunks=[
                {"operation": "test.double", "inputs": {}},
                {"operation": "test.add_one", "inputs": {}},
            ],
            initial_input={"value": 5},
        )
        assert result["success"] is True
        # 5 * 2 = 10, then 10 + 1 = 11
        assert result["output"] == 11
        assert result["stages"] == 2

    @pytest.mark.asyncio
    async def test_pipe_with_multiple_stages(self, _register_test_ops):
        """Should handle multiple pipeline stages."""
        result = await thunk_pipe(
            thunks=[
                {"operation": "test.double"},
                {"operation": "test.add_one"},
                {"operation": "test.double"},
            ],
            initial_input={"value": 2},
        )
        assert result["success"] is True
        # 2 * 2 = 4, 4 + 1 = 5, 5 * 2 = 10
        assert result["output"] == 10

    @pytest.mark.asyncio
    async def test_pipe_stops_on_failure(self, _register_test_ops):
        """Should stop pipeline on failure."""
        result = await thunk_pipe(
            thunks=[
                {"operation": "test.echo", "inputs": {"value": "ok"}},
                {"operation": "test.fail", "inputs": {"message": "boom"}},
                {"operation": "test.echo", "inputs": {"value": "never"}},
            ],
            initial_input={},
        )
        assert result["success"] is False
        assert result["failed_at"] == 1
        assert len(result["history"]) == 2

    @pytest.mark.asyncio
    async def test_pipe_records_history(self, _register_test_ops):
        """Should record history of all stages."""
        result = await thunk_pipe(
            thunks=[
                {"operation": "test.double"},
                {"operation": "test.add_one"},
            ],
            initial_input={"value": 3},
        )
        assert result["success"] is True
        assert len(result["history"]) == 2
        assert result["history"][0]["output"]["result"] == 6
        assert result["history"][1]["output"]["result"] == 7


class TestThunkFilter:
    """Tests for thunk.filter combinator."""

    @pytest.mark.asyncio
    async def test_filter_keeps_passing_items(self, _register_test_ops):
        """Should keep only items that pass predicate."""
        result = await thunk_filter(
            items=[1, 2, 3, 4, 5, 6],
            predicate={"operation": "test.is_even"},
            item_key="item",
        )
        assert result["success"] is True
        assert result["items"] == [2, 4, 6]
        assert result["count"] == 3
        assert result["filtered_out"] == 3

    @pytest.mark.asyncio
    async def test_filter_empty_result(self, _register_test_ops):
        """Should handle no items passing."""
        result = await thunk_filter(
            items=[1, 3, 5, 7],
            predicate={"operation": "test.is_even"},
            item_key="item",
        )
        assert result["success"] is True
        assert result["items"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_filter_all_pass(self, _register_test_ops):
        """Should handle all items passing."""
        result = await thunk_filter(
            items=[2, 4, 6],
            predicate={"operation": "test.is_even"},
            item_key="item",
        )
        assert result["success"] is True
        assert result["items"] == [2, 4, 6]
        assert result["filtered_out"] == 0


class TestThunkRepeat:
    """Tests for thunk.repeat combinator."""

    @pytest.mark.asyncio
    async def test_repeat_runs_n_times(self, _register_test_ops):
        """Should run operation specified number of times."""
        result = await thunk_repeat(
            operation="test.echo",
            inputs={"value": "hello"},
            times=3,
        )
        assert result["success"] is True
        assert result["total"] == 3
        assert result["completed"] == 3
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_repeat_parallel(self, _register_test_ops):
        """Should run iterations in parallel."""
        result = await thunk_repeat(
            operation="test.echo",
            inputs={"value": "parallel"},
            times=5,
            parallel=True,
        )
        assert result["success"] is True
        assert result["completed"] == 5

    @pytest.mark.asyncio
    async def test_repeat_with_index(self, _register_test_ops):
        """Should include iteration index when requested."""
        result = await thunk_repeat(
            operation="test.noop",
            inputs={},
            times=3,
            include_index=True,
            index_key="iteration",
        )
        assert result["success"] is True
        # Each call should have received the iteration index
        for i, r in enumerate(result["results"]):
            assert r["output"]["received"]["iteration"] == i

    @pytest.mark.asyncio
    async def test_repeat_stops_on_failure(self, _register_test_ops):
        """Should stop on failure when configured."""
        result = await thunk_repeat(
            operation="test.fail",
            inputs={"message": "always fails"},
            times=5,
            stop_on_failure=True,
        )
        assert result["success"] is False
        assert result["completed"] == 1  # Stopped after first failure


class TestThunkWhile:
    """Tests for thunk.while combinator."""

    @pytest.mark.asyncio
    async def test_while_loops_until_condition_false(self, _register_test_ops):
        """Should loop while condition is true."""
        # Reset counter first
        await thunk_repeat(
            operation="test.counter_reset",
            inputs={},
            times=1,
        )

        result = await thunk_while(
            condition={
                "operation": "test.should_continue",
                "inputs": {"limit": 3},
            },
            body={"operation": "test.counter_increment"},
            max_iterations=10,
            pass_output_to_condition=True,
        )
        assert result["success"] is True
        assert result["iterations"] == 3
        assert result["max_reached"] is False

    @pytest.mark.asyncio
    async def test_while_respects_max_iterations(self, _register_test_ops):
        """Should stop at max_iterations."""
        result = await thunk_while(
            condition={
                "operation": "test.check_flag",
                "inputs": {"flag": True},  # Always true
            },
            body={"operation": "test.noop"},
            max_iterations=5,
        )
        assert result["success"] is True
        assert result["iterations"] == 5
        assert result["max_reached"] is True

    @pytest.mark.asyncio
    async def test_while_zero_iterations(self, _register_test_ops):
        """Should handle condition false from start."""
        result = await thunk_while(
            condition={
                "operation": "test.check_flag",
                "inputs": {"flag": False},  # Immediately false
            },
            body={"operation": "test.noop"},
        )
        assert result["success"] is True
        assert result["iterations"] == 0


class TestThunkUntil:
    """Tests for thunk.until combinator."""

    @pytest.mark.asyncio
    async def test_until_loops_until_condition_true(self, _register_test_ops):
        """Should loop until condition becomes true."""
        # Reset counter
        await thunk_repeat(operation="test.counter_reset", inputs={}, times=1)

        result = await thunk_until(
            condition={
                "operation": "test.counter_check",
                "inputs": {"target": 3},
            },
            body={"operation": "test.counter_increment"},
            max_iterations=10,
            pass_output_to_condition=True,
        )
        assert result["success"] is True
        assert result["condition_met"] is True
        assert result["iterations"] == 3

    @pytest.mark.asyncio
    async def test_until_runs_at_least_once(self, _register_test_ops):
        """Do-until should run body at least once."""
        result = await thunk_until(
            condition={
                "operation": "test.check_flag",
                "inputs": {"flag": True},  # Immediately true (done)
            },
            body={"operation": "test.echo", "inputs": {"value": "ran"}},
            check_before=False,  # do-until (default)
        )
        assert result["success"] is True
        assert result["iterations"] == 1  # Ran once before checking

    @pytest.mark.asyncio
    async def test_until_check_before(self, _register_test_ops):
        """While-not should check before running body."""
        result = await thunk_until(
            condition={
                "operation": "test.check_flag",
                "inputs": {"flag": True},  # Already done
            },
            body={"operation": "test.echo", "inputs": {"value": "never"}},
            check_before=True,  # while-not style
        )
        assert result["success"] is True
        assert result["iterations"] == 0  # Never ran body

    @pytest.mark.asyncio
    async def test_until_max_iterations(self, _register_test_ops):
        """Should fail when max iterations reached without condition met."""
        result = await thunk_until(
            condition={
                "operation": "test.check_flag",
                "inputs": {"flag": False},  # Never done
            },
            body={"operation": "test.noop"},
            max_iterations=3,
            pass_output_to_condition=False,  # Don't pass body output to condition
        )
        assert result["success"] is False
        assert result["condition_met"] is False
        assert result["max_reached"] is True


class TestCombinatorRegistration:
    """Test that combinators are properly registered."""

    @pytest.mark.asyncio
    async def test_all_combinators_registered(self):
        """All thunk.* combinators should be discoverable."""
        from crows_nest.core.introspection import system_list_operations

        result = await system_list_operations(namespace="thunk")
        ops = result["operations"].get("thunk", [])
        op_names = {op["name"] for op in ops}

        expected = {
            "thunk.sequence",
            "thunk.parallel",
            "thunk.race",
            "thunk.retry",
            "thunk.timeout",
            "thunk.fallback",
            "thunk.conditional",
            "thunk.map",
            "thunk.reduce",
            "thunk.catch",
            "thunk.ensure",
            "thunk.tap",
            "thunk.validate",
            "thunk.zip",
            "thunk.fanout",
            "thunk.merge",
            "thunk.partition",
            "thunk.pipe",
            "thunk.filter",
            "thunk.repeat",
            "thunk.while",
            "thunk.until",
        }
        assert expected.issubset(op_names)

    @pytest.mark.asyncio
    async def test_combinators_require_no_capabilities(self):
        """Combinators should require no capabilities (checks happen per-thunk)."""
        from crows_nest.core.introspection import system_list_operations

        result = await system_list_operations(namespace="thunk")
        ops = result["operations"].get("thunk", [])

        for op in ops:
            assert op["required_capabilities"] == [], f"{op['name']} should require no capabilities"
