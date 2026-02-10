"""Tests for ThunkRegistry."""

import pytest

from crows_nest.core import (
    InvalidHandlerError,
    OperationAlreadyRegisteredError,
    OperationNotFoundError,
    ThunkRegistry,
)


class TestThunkRegistry:
    """Tests for ThunkRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ThunkRegistry()

    async def test_register_and_get(self, registry: ThunkRegistry):
        """Register an operation and retrieve it."""

        async def my_handler(a: int, b: int) -> int:
            return a + b

        registry.register("math.add", my_handler)
        info = registry.get("math.add")
        assert info.name == "math.add"
        assert info.handler is my_handler

    async def test_decorator_registration(self, registry: ThunkRegistry):
        """Register using decorator syntax."""

        @registry.operation("math.multiply")
        async def multiply(x: int, y: int) -> int:
            return x * y

        info = registry.get("math.multiply")
        assert info.handler is multiply
        # Handler still works
        result = await multiply(3, 4)
        assert result == 12

    async def test_get_handler(self, registry: ThunkRegistry):
        """get_handler returns the handler function directly."""

        async def handler(x: int) -> int:
            return x * 2

        registry.register("double", handler)
        got = registry.get_handler("double")
        assert got is handler
        assert await got(5) == 10

    def test_operation_not_found(self, registry: ThunkRegistry):
        """Getting nonexistent operation raises error."""
        with pytest.raises(OperationNotFoundError) as exc_info:
            registry.get("does.not.exist")
        assert "does.not.exist" in str(exc_info.value)

    async def test_duplicate_registration_raises(self, registry: ThunkRegistry):
        """Registering same operation twice raises error."""

        async def handler1():
            pass

        async def handler2():
            pass

        registry.register("test.op", handler1)
        with pytest.raises(OperationAlreadyRegisteredError):
            registry.register("test.op", handler2)

    async def test_replace_allows_duplicate(self, registry: ThunkRegistry):
        """With replace=True, can overwrite existing registration."""

        async def handler1():
            return 1

        async def handler2():
            return 2

        registry.register("test.op", handler1)
        registry.register("test.op", handler2, replace=True)

        got = registry.get_handler("test.op")
        assert await got() == 2

    def test_non_async_handler_rejected(self, registry: ThunkRegistry):
        """Non-async handlers are rejected."""

        def sync_handler():
            pass

        with pytest.raises(InvalidHandlerError) as exc_info:
            registry.register("test.op", sync_handler)
        assert "async" in str(exc_info.value).lower()

    def test_non_callable_rejected(self, registry: ThunkRegistry):
        """Non-callable handlers are rejected."""
        with pytest.raises(InvalidHandlerError):
            registry.register("test.op", "not a function")  # type: ignore[arg-type]

    async def test_has_operation(self, registry: ThunkRegistry):
        """has() checks if operation exists."""

        async def handler():
            pass

        assert registry.has("test.op") is False
        registry.register("test.op", handler)
        assert registry.has("test.op") is True

    async def test_contains_operator(self, registry: ThunkRegistry):
        """'in' operator works on registry."""

        async def handler():
            pass

        assert "test.op" not in registry
        registry.register("test.op", handler)
        assert "test.op" in registry

    async def test_list_operations(self, registry: ThunkRegistry):
        """list_operations returns sorted operation names."""

        async def handler():
            pass

        registry.register("z.op", handler)
        registry.register("a.op", handler)
        registry.register("m.op", handler)

        ops = registry.list_operations()
        assert ops == ["a.op", "m.op", "z.op"]

    async def test_unregister(self, registry: ThunkRegistry):
        """unregister removes an operation."""

        async def handler():
            pass

        registry.register("test.op", handler)
        assert registry.has("test.op")

        result = registry.unregister("test.op")
        assert result is True
        assert registry.has("test.op") is False

        # Unregistering nonexistent returns False
        result = registry.unregister("test.op")
        assert result is False

    async def test_clear(self, registry: ThunkRegistry):
        """clear removes all operations."""

        async def handler():
            pass

        registry.register("op1", handler)
        registry.register("op2", handler)
        assert len(registry) == 2

        registry.clear()
        assert len(registry) == 0

    async def test_len(self, registry: ThunkRegistry):
        """len() returns number of registered operations."""

        async def handler():
            pass

        assert len(registry) == 0
        registry.register("op1", handler)
        assert len(registry) == 1
        registry.register("op2", handler)
        assert len(registry) == 2

    async def test_description_from_docstring(self, registry: ThunkRegistry):
        """Description defaults to handler docstring."""

        @registry.operation("documented")
        async def handler_with_docs():
            """This is the docstring description."""
            pass

        info = registry.get("documented")
        assert "docstring description" in info.description

    async def test_explicit_description(self, registry: ThunkRegistry):
        """Explicit description overrides docstring."""

        @registry.operation("custom", description="Custom description")
        async def handler():
            """Docstring that will be ignored."""
            pass

        info = registry.get("custom")
        assert info.description == "Custom description"

    async def test_required_capabilities(self, registry: ThunkRegistry):
        """Required capabilities are stored in operation info."""

        @registry.operation("secure", required_capabilities=frozenset({"admin", "write"}))
        async def secure_handler():
            pass

        info = registry.get("secure")
        assert info.required_capabilities == frozenset({"admin", "write"})

    async def test_parameter_extraction(self, registry: ThunkRegistry):
        """Parameters are extracted from handler signature."""

        @registry.operation("typed")
        async def typed_handler(
            required_param: str,
            optional_param: int = 42,
            *,
            keyword_only: bool = True,
        ) -> dict:
            return {}

        info = registry.get("typed")
        params = info.parameters

        assert "required_param" in params
        assert params["required_param"].required is True
        assert params["required_param"].annotation == "str"

        assert "optional_param" in params
        assert params["optional_param"].required is False
        assert params["optional_param"].default == 42

        assert "keyword_only" in params
        assert params["keyword_only"].default is True

    async def test_list_operations_info(self, registry: ThunkRegistry):
        """list_operations_info returns full operation info."""

        @registry.operation("op1", description="First")
        async def handler1():
            pass

        @registry.operation("op2", description="Second")
        async def handler2():
            pass

        infos = registry.list_operations_info()
        assert len(infos) == 2
        assert infos[0].name == "op1"
        assert infos[1].name == "op2"
