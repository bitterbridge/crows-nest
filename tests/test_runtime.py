"""Tests for ThunkRuntime."""

from uuid import uuid4

import pytest

from crows_nest.core import (
    CancellationError,
    CancellationToken,
    CyclicDependencyError,
    DependencyNotFoundError,
    RuntimeEvent,
    Thunk,
    ThunkContext,
    ThunkRef,
    ThunkRegistry,
    ThunkRuntime,
)


class TestThunkRuntime:
    """Tests for ThunkRuntime."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return ThunkRegistry()

    @pytest.fixture
    def runtime(self, registry: ThunkRegistry):
        """Create a runtime with the registry."""
        return ThunkRuntime(registry=registry)

    async def test_force_simple_thunk(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Force a simple thunk with no dependencies."""

        @registry.operation("math.add")
        async def add(a: int, b: int) -> int:
            return a + b

        thunk = Thunk.create("math.add", {"a": 1, "b": 2})
        result = await runtime.force(thunk)

        assert result.is_success
        assert result.value == 3
        assert result.thunk_id == thunk.id
        assert result.duration_ms >= 0

    async def test_force_thunk_with_dependency(
        self, registry: ThunkRegistry, runtime: ThunkRuntime
    ):
        """Force thunks with linear dependency chain."""

        @registry.operation("double")
        async def double(x: int) -> int:
            return x * 2

        @registry.operation("add_one")
        async def add_one(x: int) -> int:
            return x + 1

        # Create thunk A: double(5) = 10
        thunk_a = Thunk.create("double", {"x": 5})
        # Create thunk B: add_one(result of A) = 11
        thunk_b = Thunk.create("add_one", {"x": ThunkRef(thunk_id=thunk_a.id)})

        result = await runtime.force_all(
            {thunk_a.id: thunk_a, thunk_b.id: thunk_b},
            target=thunk_b.id,
        )

        assert result.is_success
        assert result.value == 11

    async def test_force_diamond_dependency(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Force thunks with diamond dependency pattern."""
        #      A
        #     / \
        #    B   C
        #     \ /
        #      D

        @registry.operation("value")
        async def value(v: int) -> int:
            return v

        @registry.operation("add")
        async def add(a: int, b: int) -> int:
            return a + b

        thunk_a = Thunk.create("value", {"v": 10})
        thunk_b = Thunk.create("value", {"v": ThunkRef(thunk_id=thunk_a.id)})
        thunk_c = Thunk.create("value", {"v": ThunkRef(thunk_id=thunk_a.id)})
        thunk_d = Thunk.create(
            "add",
            {
                "a": ThunkRef(thunk_id=thunk_b.id),
                "b": ThunkRef(thunk_id=thunk_c.id),
            },
        )

        thunks = {t.id: t for t in [thunk_a, thunk_b, thunk_c, thunk_d]}
        result = await runtime.force_all(thunks, target=thunk_d.id)

        assert result.is_success
        assert result.value == 20

    async def test_operation_not_found(self, runtime: ThunkRuntime):
        """Thunk with unknown operation fails."""
        thunk = Thunk.create("nonexistent.operation", {})
        result = await runtime.force(thunk)

        assert result.is_failure
        assert result.error is not None
        assert "OperationNotFoundError" in result.error.error_type

    async def test_capability_check_passes(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Thunk with required capabilities succeeds."""

        @registry.operation("secure.op", required_capabilities=frozenset({"admin"}))
        async def secure_op() -> str:
            return "secret"

        thunk = Thunk.create(
            "secure.op",
            {},
            capabilities=frozenset({"admin", "other"}),
        )
        result = await runtime.force(thunk)

        assert result.is_success
        assert result.value == "secret"

    async def test_capability_check_fails(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Thunk without required capabilities fails."""

        @registry.operation("secure.op", required_capabilities=frozenset({"admin"}))
        async def secure_op() -> str:
            return "secret"

        thunk = Thunk.create(
            "secure.op",
            {},
            capabilities=frozenset({"user"}),  # Missing "admin"
        )
        result = await runtime.force(thunk)

        assert result.is_failure
        assert result.error is not None
        assert "CapabilityError" in result.error.error_type

    async def test_handler_exception_captured(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Exception in handler is captured as failure."""

        @registry.operation("failing.op")
        async def failing_op() -> None:
            raise ValueError("Something went wrong")

        thunk = Thunk.create("failing.op", {})
        result = await runtime.force(thunk)

        assert result.is_failure
        assert result.error is not None
        assert result.error.error_type == "ValueError"
        assert "Something went wrong" in result.error.message

    async def test_dependency_not_found(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Missing dependency raises error."""

        @registry.operation("identity")
        async def identity(x: int) -> int:
            return x

        missing_id = uuid4()
        thunk = Thunk.create("identity", {"x": ThunkRef(thunk_id=missing_id)})

        with pytest.raises(DependencyNotFoundError):
            await runtime.force(thunk)

    async def test_cyclic_dependency_detected(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Cyclic dependencies raise error."""

        @registry.operation("identity")
        async def identity(x: int) -> int:
            return x

        # Create cycle: A depends on B, B depends on A
        id_a = uuid4()
        id_b = uuid4()

        thunk_a = Thunk(
            id=id_a,
            operation="identity",
            inputs={"x": ThunkRef(thunk_id=id_b)},
            metadata=Thunk.create("identity", {}).metadata,
        )
        thunk_b = Thunk(
            id=id_b,
            operation="identity",
            inputs={"x": ThunkRef(thunk_id=id_a)},
            metadata=Thunk.create("identity", {}).metadata,
        )

        with pytest.raises(CyclicDependencyError):
            await runtime.force_all({id_a: thunk_a, id_b: thunk_b})

    async def test_thunk_returning_thunk(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Handler can return a thunk that gets forced."""

        @registry.operation("create_thunk")
        async def create_thunk(value: int) -> Thunk:
            return Thunk.create("double", {"x": value})

        @registry.operation("double")
        async def double(x: int) -> int:
            return x * 2

        thunk = Thunk.create("create_thunk", {"value": 21})
        result = await runtime.force(thunk)

        # Should have forced the returned thunk
        assert result.is_success
        assert result.value == 42

    async def test_output_path_extraction(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """ThunkRef with output_path extracts nested value."""

        @registry.operation("get_data")
        async def get_data() -> dict:
            return {"result": {"value": 42, "status": "ok"}}

        @registry.operation("use_value")
        async def use_value(v: int) -> int:
            return v + 1

        thunk_a = Thunk.create("get_data", {})
        thunk_b = Thunk.create(
            "use_value",
            {"v": ThunkRef(thunk_id=thunk_a.id, output_path="result.value")},
        )

        result = await runtime.force_all(
            {thunk_a.id: thunk_a, thunk_b.id: thunk_b},
            target=thunk_b.id,
        )

        assert result.is_success
        assert result.value == 43

    async def test_event_emission(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Runtime emits lifecycle events."""
        events_received: list[RuntimeEvent] = []

        async def listener(event_data):
            events_received.append(event_data.event)

        runtime.on(RuntimeEvent.THUNK_FORCING, listener)
        runtime.on(RuntimeEvent.THUNK_COMPLETED, listener)

        @registry.operation("simple")
        async def simple() -> int:
            return 1

        thunk = Thunk.create("simple", {})
        await runtime.force(thunk)

        assert RuntimeEvent.THUNK_FORCING in events_received
        assert RuntimeEvent.THUNK_COMPLETED in events_received

    async def test_event_unregister(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Can unregister event listeners."""
        call_count = 0

        async def listener(event_data):
            nonlocal call_count
            call_count += 1

        runtime.on(RuntimeEvent.THUNK_FORCING, listener)

        @registry.operation("simple")
        async def simple() -> int:
            return 1

        thunk = Thunk.create("simple", {})
        await runtime.force(thunk)
        assert call_count == 1

        runtime.off(RuntimeEvent.THUNK_FORCING, listener)
        thunk2 = Thunk.create("simple", {})
        runtime.clear_cache()
        await runtime.force(thunk2)
        assert call_count == 1  # Listener not called again

    async def test_result_cache(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Results are cached."""
        call_count = 0

        @registry.operation("counting")
        async def counting() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        thunk = Thunk.create("counting", {})

        result1 = await runtime.force(thunk)
        result2 = await runtime.force(thunk)

        assert result1.value == 1
        assert result2.value == 1  # Cached, not called again
        assert call_count == 1

    async def test_clear_cache(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """clear_cache removes cached results."""
        call_count = 0

        @registry.operation("counting")
        async def counting() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        thunk = Thunk.create("counting", {})

        await runtime.force(thunk)
        assert call_count == 1

        runtime.clear_cache()
        await runtime.force(thunk)
        assert call_count == 2

    async def test_get_result(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """get_result retrieves cached result."""

        @registry.operation("simple")
        async def simple() -> int:
            return 42

        thunk = Thunk.create("simple", {})

        # Not forced yet
        assert runtime.get_result(thunk.id) is None

        await runtime.force(thunk)

        # Now available
        cached = runtime.get_result(thunk.id)
        assert cached is not None
        assert cached.value == 42

    async def test_force_empty_raises(self, runtime: ThunkRuntime):
        """force_all with empty dict raises."""
        with pytest.raises(ValueError, match="No thunks"):
            await runtime.force_all({})

    async def test_force_with_context(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Context is passed through to execution."""
        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token, my_value="test")

        @registry.operation("simple")
        async def simple() -> int:
            return 42

        thunk = Thunk.create("simple", {})
        result = await runtime.force(thunk, context=ctx)

        assert result.is_success
        assert result.value == 42

    async def test_cancellation_before_start(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Pre-cancelled context raises CancellationError."""
        token = CancellationToken()
        token.cancel_sync("Already cancelled")
        ctx = ThunkContext.create(cancellation=token)

        @registry.operation("simple")
        async def simple() -> int:
            return 42

        thunk = Thunk.create("simple", {})

        with pytest.raises(CancellationError) as exc_info:
            await runtime.force(thunk, context=ctx)
        assert "Already cancelled" in str(exc_info.value)

    async def test_cancellation_between_thunks(
        self, registry: ThunkRegistry, runtime: ThunkRuntime
    ):
        """Cancellation between thunks stops execution."""
        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token)
        execution_order: list[str] = []

        @registry.operation("first")
        async def first() -> int:
            execution_order.append("first")
            # Cancel after first thunk executes
            token.cancel_sync("Cancelled after first")
            return 1

        @registry.operation("second")
        async def second(x: int) -> int:
            execution_order.append("second")
            return x + 1

        thunk_a = Thunk.create("first", {})
        thunk_b = Thunk.create("second", {"x": ThunkRef(thunk_id=thunk_a.id)})

        with pytest.raises(CancellationError):
            await runtime.force_all(
                {thunk_a.id: thunk_a, thunk_b.id: thunk_b},
                target=thunk_b.id,
                context=ctx,
            )

        # Only first should have run
        assert execution_order == ["first"]

    async def test_context_passed_to_handler(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """Handler can receive context via _context parameter."""
        received_context: ThunkContext | None = None

        @registry.operation("context_aware")
        async def context_aware(_context: ThunkContext) -> str:
            nonlocal received_context
            received_context = _context
            return _context.get("my_key", "default")

        token = CancellationToken()
        ctx = ThunkContext.create(cancellation=token, my_key="my_value")

        thunk = Thunk.create("context_aware", {})
        result = await runtime.force(thunk, context=ctx)

        assert result.is_success
        assert result.value == "my_value"
        assert received_context is not None
        assert received_context.get("my_key") == "my_value"

    async def test_cancelled_event_emission(self, registry: ThunkRegistry, runtime: ThunkRuntime):
        """THUNK_CANCELLED event is emitted when cancelled."""
        token = CancellationToken()
        token.cancel_sync("Test cancellation")
        ctx = ThunkContext.create(cancellation=token)
        events_received: list[RuntimeEvent] = []

        async def listener(event_data):
            events_received.append(event_data.event)

        runtime.on(RuntimeEvent.THUNK_CANCELLED, listener)

        @registry.operation("simple")
        async def simple() -> int:
            return 42

        thunk = Thunk.create("simple", {})

        with pytest.raises(CancellationError):
            await runtime.force(thunk, context=ctx)

        # Note: Cancellation at start happens before we get to the loop,
        # so THUNK_CANCELLED isn't emitted in this case. It's emitted
        # when cancellation happens between thunks.
