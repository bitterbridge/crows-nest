"""Hello World Plugin - Simplest possible Crow's Nest plugin.

This plugin demonstrates the basics of creating a custom operation.

Usage:
    1. Copy this file to ~/.crows-nest/plugins/
    2. Restart the server
    3. The operation 'hello.greet' will be available

Example thunk:
    {
        "operation": "hello.greet",
        "inputs": {"name": "World"},
        "capabilities": ["hello.greet"]
    }
"""

from crows_nest.core.registry import thunk_operation


@thunk_operation(
    name="hello.greet",
    description="Return a greeting message",
    required_capabilities=frozenset({"hello.greet"}),
)
async def greet(name: str = "World") -> str:
    """Generate a greeting for the given name.

    Args:
        name: The name to greet (default: "World")

    Returns:
        A greeting string
    """
    return f"Hello, {name}! Welcome to Crow's Nest."


@thunk_operation(
    name="hello.farewell",
    description="Return a farewell message",
    required_capabilities=frozenset({"hello.greet"}),
)
async def farewell(name: str = "friend") -> str:
    """Generate a farewell message.

    Args:
        name: The name to bid farewell (default: "friend")

    Returns:
        A farewell string
    """
    return f"Goodbye, {name}! Until next time."


# Plugin metadata (optional but recommended)
PLUGIN_NAME = "hello_world"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "A simple greeting plugin demonstrating Crow's Nest operations"
PLUGIN_AUTHOR = "Crow's Nest Examples"
