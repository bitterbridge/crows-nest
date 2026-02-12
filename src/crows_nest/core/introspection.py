"""System introspection operations.

Provides operations for agents to discover available capabilities,
operations, plugins, and how to use the introspection system itself.
"""

from __future__ import annotations

from typing import Any

from crows_nest.core.registry import get_global_registry, thunk_operation

# The introspection operations themselves - used for self-documentation
INTROSPECTION_OPERATIONS = {
    "system.help": "Get help on how to introspect the system and discover available operations.",
    "system.list_operations": "List all registered operations with their descriptions.",
    "system.describe_operation": "Get detailed information about a specific operation.",
    "system.list_plugins": "List all loaded plugins and their operations.",
    "system.list_capabilities": "List all capabilities used by registered operations.",
}


@thunk_operation(
    name="system.help",
    description="Get help on how to introspect the system and discover available operations.",
    required_capabilities=frozenset(),  # No capabilities needed - always available
)
async def system_help() -> dict[str, Any]:
    """Get help on system introspection.

    Returns information about how to discover available operations,
    plugins, and capabilities in the system.

    Returns:
        Dict with introspection guide and available introspection operations.
    """
    return {
        "message": "Welcome to Crow's Nest! Use these operations to discover what's available.",
        "introspection_operations": dict(INTROSPECTION_OPERATIONS),
        "quick_start": [
            "1. Call 'system.list_operations' to see all available operations",
            "2. Call 'system.describe_operation' with an operation name for details",
            "3. Call 'system.list_plugins' to see loaded plugins",
            "4. Call 'system.list_capabilities' to see what capabilities exist",
        ],
        "tips": [
            "Operations are namespaced (e.g., 'shell.run', 'memory.recall')",
            "Each operation requires specific capabilities to execute",
            "Use 'system.describe_operation' to see required capabilities",
            "Plugins can add new operations dynamically",
        ],
    }


@thunk_operation(
    name="system.list_operations",
    description="List all registered operations with their descriptions.",
    required_capabilities=frozenset(),  # No capabilities needed
)
async def system_list_operations(
    namespace: str | None = None,
    include_parameters: bool = False,
) -> dict[str, Any]:
    """List all registered operations.

    Args:
        namespace: Optional namespace filter (e.g., "shell", "memory").
        include_parameters: Whether to include parameter info (default False).

    Returns:
        Dict with list of operations grouped by namespace.
    """
    registry = get_global_registry()
    operations = registry.list_operations_info()

    # Group by namespace
    grouped: dict[str, list[dict[str, Any]]] = {}
    for op in operations:
        # Extract namespace from operation name (e.g., "shell" from "shell.run")
        parts = op.name.split(".")
        ns = parts[0] if len(parts) > 1 else "default"

        # Filter by namespace if specified
        if namespace and ns != namespace:
            continue

        op_info: dict[str, Any] = {
            "name": op.name,
            "description": op.description,
            "required_capabilities": sorted(op.required_capabilities),
        }

        if include_parameters:
            op_info["parameters"] = {
                name: {
                    "type": (param.annotation or "Any"),
                    "required": param.required,
                    "default": param.default if not param.required else None,
                }
                for name, param in op.parameters.items()
            }

        if ns not in grouped:
            grouped[ns] = []
        grouped[ns].append(op_info)

    # Sort operations within each namespace
    for _ns, ops in grouped.items():
        ops.sort(key=lambda x: x["name"])

    return {
        "total_operations": sum(len(ops) for ops in grouped.values()),
        "namespaces": sorted(grouped.keys()),
        "operations": grouped,
    }


@thunk_operation(
    name="system.describe_operation",
    description="Get detailed information about a specific operation.",
    required_capabilities=frozenset(),  # No capabilities needed
)
async def system_describe_operation(operation_name: str) -> dict[str, Any]:
    """Get detailed information about an operation.

    Args:
        operation_name: The fully qualified operation name (e.g., "shell.run").

    Returns:
        Dict with operation details including parameters and capabilities.
    """
    registry = get_global_registry()

    if operation_name not in registry:
        return {
            "found": False,
            "error": f"Operation '{operation_name}' not found",
            "suggestion": "Use 'system.list_operations' to see available operations",
        }

    op_info = registry.get(operation_name)
    return {
        "found": True,
        "operation": {
            "name": op_info.name,
            "description": op_info.description,
            "required_capabilities": sorted(op_info.required_capabilities),
            "parameters": {
                name: {
                    "type": (param.annotation or "Any"),
                    "required": param.required,
                    "default": None if param.required else param.default,
                    "description": (
                        f"{'Required' if param.required else 'Optional'} "
                        f"{(param.annotation or 'Any')} parameter"
                    ),
                }
                for name, param in op_info.parameters.items()
            },
        },
        "usage_example": _generate_usage_example(op_info),
    }


def _generate_usage_example(op_info: Any) -> dict[str, Any]:
    """Generate a usage example for an operation."""
    example_inputs: dict[str, Any] = {}

    for name, param in op_info.parameters.items():
        if param.required:
            # Generate example values based on type
            if "str" in (param.annotation or "Any").lower():
                example_inputs[name] = f"<{name}>"
            elif "int" in (param.annotation or "Any").lower():
                example_inputs[name] = 0
            elif "bool" in (param.annotation or "Any").lower():
                example_inputs[name] = True
            elif "list" in (param.annotation or "Any").lower():
                example_inputs[name] = []
            elif "dict" in (param.annotation or "Any").lower():
                example_inputs[name] = {}
            else:
                example_inputs[name] = f"<{name}>"

    return {
        "operation": op_info.name,
        "inputs": example_inputs,
    }


@thunk_operation(
    name="system.list_plugins",
    description="List all loaded plugins and their operations.",
    required_capabilities=frozenset(),  # No capabilities needed
)
async def system_list_plugins() -> dict[str, Any]:
    """List all loaded plugins.

    Returns:
        Dict with plugin information including their operations.
    """
    from crows_nest.plugins.loader import get_plugin_loader

    loader = get_plugin_loader()
    plugins = loader.plugins

    plugin_list = []
    for plugin in plugins:
        plugin_list.append(
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "author": plugin.author,
                "source": plugin.source.value,
                "loaded": plugin.loaded,
                "operations": plugin.operations,
                "load_error": plugin.load_error,
            }
        )

    return {
        "total_plugins": len(plugins),
        "loaded_plugins": sum(1 for p in plugins if p.loaded),
        "failed_plugins": sum(1 for p in plugins if not p.loaded),
        "plugins": sorted(plugin_list, key=lambda x: x["name"]),
    }


@thunk_operation(
    name="system.list_capabilities",
    description="List all capabilities used by registered operations.",
    required_capabilities=frozenset(),  # No capabilities needed
)
async def system_list_capabilities() -> dict[str, Any]:
    """List all capabilities used by operations.

    Returns:
        Dict with capabilities and which operations require them.
    """
    registry = get_global_registry()
    operations = registry.list_operations_info()

    # Collect capabilities and their operations
    capabilities: dict[str, list[str]] = {}
    for op in operations:
        for cap in op.required_capabilities:
            if cap not in capabilities:
                capabilities[cap] = []
            capabilities[cap].append(op.name)

    # Sort operations for each capability
    for _cap, ops in capabilities.items():
        ops.sort()

    # Group capabilities by namespace
    grouped: dict[str, dict[str, list[str]]] = {}
    for cap, ops in capabilities.items():
        parts = cap.split(".")
        ns = parts[0] if len(parts) > 1 else "general"
        if ns not in grouped:
            grouped[ns] = {}
        grouped[ns][cap] = ops

    return {
        "total_capabilities": len(capabilities),
        "capability_namespaces": sorted(grouped.keys()),
        "capabilities": dict(sorted(capabilities.items())),
        "by_namespace": {ns: dict(sorted(caps.items())) for ns, caps in sorted(grouped.items())},
    }


@thunk_operation(
    name="system.search_operations",
    description="Search for operations by keyword in name or description.",
    required_capabilities=frozenset(),  # No capabilities needed
)
async def system_search_operations(
    query: str,
    search_descriptions: bool = True,
) -> dict[str, Any]:
    """Search for operations matching a query.

    Args:
        query: Search term to match against operation names and descriptions.
        search_descriptions: Whether to search in descriptions (default True).

    Returns:
        Dict with matching operations.
    """
    registry = get_global_registry()
    operations = registry.list_operations_info()

    query_lower = query.lower()
    matches = []

    for op in operations:
        name_match = query_lower in op.name.lower()
        desc_match = search_descriptions and query_lower in (op.description or "").lower()

        if name_match or desc_match:
            matches.append(
                {
                    "name": op.name,
                    "description": op.description,
                    "required_capabilities": sorted(op.required_capabilities),
                    "match_type": "name" if name_match else "description",
                }
            )

    return {
        "query": query,
        "total_matches": len(matches),
        "matches": sorted(matches, key=lambda x: x["name"]),
    }
