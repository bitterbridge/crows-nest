"""Operation registration module.

Imports all operation modules to trigger @thunk_operation decorator registration.
Import this module when you need all operations to be available.
"""

from __future__ import annotations


def register_all_operations() -> None:
    """Register all built-in operations.

    This function imports all modules that define operations using the
    @thunk_operation decorator, causing them to be registered with the
    global registry.

    Call this at application startup to ensure all operations are available.
    """
    # Core operations (system.*, thunk.*)
    from crows_nest.core import combinators as _combinators  # noqa: F401
    from crows_nest.core import introspection as _introspection  # noqa: F401

    # Shell operations (shell.*, verify.*)
    from crows_nest.shell import operations as _shell_ops  # noqa: F401
    from crows_nest.shell import verification as _verification  # noqa: F401

    # Memory operations (memory.*)
    from crows_nest.memory import operations as _memory_ops  # noqa: F401

    # Agent operations (agent.*, agent.hierarchy.*)
    from crows_nest.agents import operations as _agent_ops  # noqa: F401
    from crows_nest.agents import spawning as _spawning  # noqa: F401

    # Message queue operations (queue.*)
    from crows_nest.messaging import operations as _queue_ops  # noqa: F401

    # Orchestration operations (orchestrate.*)
    from crows_nest.orchestrator import planner as _planner  # noqa: F401
