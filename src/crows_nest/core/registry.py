"""Thunk operation registry.

The registry maps operation names to handler functions. Handlers are async
functions that receive resolved inputs and return a result.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

# Type alias for operation handlers
OperationHandler = Callable[..., Awaitable[Any]]


class RegistryError(Exception):
    """Base exception for registry errors."""


class OperationNotFoundError(RegistryError):
    """Raised when an operation is not found in the registry."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"Operation not found: {operation}")


class OperationAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register an operation that already exists."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"Operation already registered: {operation}")


class InvalidHandlerError(RegistryError):
    """Raised when a handler doesn't meet requirements."""


@dataclass(frozen=True, slots=True)
class OperationInfo:
    """Information about a registered operation.

    Attributes:
        name: The operation name (e.g., "shell.run").
        handler: The async handler function.
        description: Human-readable description.
        required_capabilities: Capabilities required to execute this operation.
        parameters: Parameter information extracted from handler signature.
    """

    name: str
    handler: OperationHandler
    description: str
    required_capabilities: frozenset[str]
    parameters: dict[str, ParameterInfo]


@dataclass(frozen=True, slots=True)
class ParameterInfo:
    """Information about an operation parameter.

    Attributes:
        name: Parameter name.
        annotation: Type annotation as string, if present.
        required: Whether the parameter is required (no default).
        default: Default value if not required.
    """

    name: str
    annotation: str | None
    required: bool
    default: Any = None


def _extract_parameters(handler: OperationHandler) -> dict[str, ParameterInfo]:
    """Extract parameter information from a handler's signature."""
    sig = inspect.signature(handler)
    params: dict[str, ParameterInfo] = {}

    for name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = None
        if param.annotation is not inspect.Parameter.empty:
            annotation = (
                param.annotation.__name__
                if hasattr(param.annotation, "__name__")
                else str(param.annotation)
            )

        required = param.default is inspect.Parameter.empty
        default = None if required else param.default

        params[name] = ParameterInfo(
            name=name,
            annotation=annotation,
            required=required,
            default=default,
        )

    return params


def _validate_handler(handler: OperationHandler, operation: str) -> None:
    """Validate that a handler meets requirements."""
    if not callable(handler):
        raise InvalidHandlerError(f"Handler for '{operation}' is not callable")

    if not inspect.iscoroutinefunction(handler):
        raise InvalidHandlerError(f"Handler for '{operation}' must be an async function")


@dataclass
class ThunkRegistry:
    """Registry of thunk operations.

    Maps operation names to their handler functions. Supports decorator-based
    registration and introspection.

    Example:
        registry = ThunkRegistry()

        @registry.operation("math.add")
        async def add(a: int, b: int) -> int:
            return a + b

        # Or register explicitly
        registry.register("math.subtract", subtract_handler)

        # Execute
        handler = registry.get("math.add")
        result = await handler(a=1, b=2)
    """

    _operations: dict[str, OperationInfo] = field(default_factory=dict)

    def register(
        self,
        name: str,
        handler: OperationHandler,
        *,
        description: str | None = None,
        required_capabilities: frozenset[str] | None = None,
        replace: bool = False,
    ) -> None:
        """Register an operation handler.

        Args:
            name: Operation name (e.g., "shell.run").
            handler: Async function to handle the operation.
            description: Human-readable description. Defaults to handler docstring.
            required_capabilities: Capabilities required to execute. Defaults to empty.
            replace: If True, replace existing registration. Otherwise raise error.

        Raises:
            OperationAlreadyRegisteredError: If operation exists and replace=False.
            InvalidHandlerError: If handler doesn't meet requirements.
        """
        _validate_handler(handler, name)

        if name in self._operations and not replace:
            raise OperationAlreadyRegisteredError(name)

        desc = description or handler.__doc__ or f"Operation: {name}"
        # Clean up docstring formatting
        desc = inspect.cleandoc(desc)

        self._operations[name] = OperationInfo(
            name=name,
            handler=handler,
            description=desc,
            required_capabilities=required_capabilities or frozenset(),
            parameters=_extract_parameters(handler),
        )

    def operation(
        self,
        name: str,
        *,
        description: str | None = None,
        required_capabilities: frozenset[str] | None = None,
    ) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
        """Decorator to register an operation handler.

        Example:
            @registry.operation("shell.run", required_capabilities=frozenset({"shell"}))
            async def shell_run(command: str, args: list[str]) -> ShellResult:
                ...
        """

        def decorator(handler: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
            self.register(
                name,
                handler,
                description=description,
                required_capabilities=required_capabilities,
            )
            return handler

        return decorator

    def get(self, name: str) -> OperationInfo:
        """Get operation info by name.

        Args:
            name: The operation name.

        Returns:
            OperationInfo for the operation.

        Raises:
            OperationNotFoundError: If operation is not registered.
        """
        if name not in self._operations:
            raise OperationNotFoundError(name)
        return self._operations[name]

    def get_handler(self, name: str) -> OperationHandler:
        """Get the handler function for an operation.

        Args:
            name: The operation name.

        Returns:
            The handler function.

        Raises:
            OperationNotFoundError: If operation is not registered.
        """
        return self.get(name).handler

    def has(self, name: str) -> bool:
        """Check if an operation is registered."""
        return name in self._operations

    def list_operations(self) -> list[str]:
        """List all registered operation names."""
        return sorted(self._operations.keys())

    def list_operations_info(self) -> list[OperationInfo]:
        """List all registered operations with full info."""
        return [self._operations[name] for name in sorted(self._operations.keys())]

    def unregister(self, name: str) -> bool:
        """Unregister an operation.

        Args:
            name: The operation name.

        Returns:
            True if operation was unregistered, False if not found.
        """
        if name in self._operations:
            del self._operations[name]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()

    def __len__(self) -> int:
        """Return number of registered operations."""
        return len(self._operations)

    def __contains__(self, name: str) -> bool:
        """Check if operation is registered."""
        return self.has(name)


# Global registry instance
_global_registry: ThunkRegistry | None = None


def get_global_registry() -> ThunkRegistry:
    """Get the global registry instance, creating it if needed."""
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = ThunkRegistry()
    return _global_registry


def thunk_operation(
    name: str,
    *,
    description: str | None = None,
    required_capabilities: frozenset[str] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to register an operation with the global registry.

    Example:
        @thunk_operation("shell.run", required_capabilities=frozenset({"shell"}))
        async def shell_run(command: str, args: list[str]) -> ShellResult:
            ...
    """
    return get_global_registry().operation(
        name,
        description=description,
        required_capabilities=required_capabilities,
    )
