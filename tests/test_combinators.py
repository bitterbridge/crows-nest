"""Tests for thunk combinators."""

from __future__ import annotations

import pytest

from crows_nest.core.combinators import (
    thunk_catch,
    thunk_conditional,
    thunk_ensure,
    thunk_fallback,
    thunk_map,
    thunk_parallel,
    thunk_race,
    thunk_reduce,
    thunk_retry,
    thunk_sequence,
    thunk_tap,
    thunk_timeout,
    thunk_validate,
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
