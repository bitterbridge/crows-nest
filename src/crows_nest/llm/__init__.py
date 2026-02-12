"""LLM integration via atomic_agents adapter.

Bridges atomic_agents with the thunk system, providing:
- Provider configuration and instantiation
- Thunk operations as atomic_agents tools
- Context propagation (cancellation, progress)
- Mock providers for testing and exploration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# These don't depend on atomic_agents - always available
from crows_nest.llm.mock import (
    ConversationRecorder,
    MockAgent,
)
from crows_nest.llm.providers import (
    AnthropicConfig,
    LiteLLMConfig,
    MockConfig,
    OpenAIConfig,
    ProviderConfig,
)

if TYPE_CHECKING:
    from crows_nest.llm.adapter import (
        AgentAdapter,
        ThunkTool,
        create_client,
        tools_from_registry,
    )

__all__ = [
    "AgentAdapter",
    "AnthropicConfig",
    "ConversationRecorder",
    "LiteLLMConfig",
    "MockAgent",
    "MockConfig",
    "OpenAIConfig",
    "ProviderConfig",
    "ThunkTool",
    "create_client",
    "tools_from_registry",
]

# Lazy imports for adapter module (depends on atomic_agents)
_ADAPTER_EXPORTS = {"AgentAdapter", "ThunkTool", "create_client", "tools_from_registry"}


def __getattr__(name: str) -> Any:
    """Lazy import adapter exports to avoid atomic_agents import at module load."""
    if name in _ADAPTER_EXPORTS:
        from crows_nest.llm import adapter

        return getattr(adapter, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
