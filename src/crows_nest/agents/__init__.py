"""Agent factory and operations."""

from crows_nest.agents.factory import (
    Agent,
    AgentContext,
    AgentFactory,
    AgentSpec,
    ToolSpec,
)
from crows_nest.agents.llm import (
    AtomicAgentsProvider,
    ChatMessage,
    CompletionResult,
    LLMConfig,
    LLMProvider,
    LLMProviderType,
    MockLLMProviderAdapter,
    create_provider,
)
from crows_nest.agents.operations import (
    AgentCreateInput,
    AgentCreateOutput,
    AgentRunInput,
    AgentRunOutput,
    TextResponse,
    agent_create,
    agent_run,
    agent_spawn,
    get_output_schema,
    register_output_schema,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentCreateInput",
    "AgentCreateOutput",
    "AgentFactory",
    "AgentRunInput",
    "AgentRunOutput",
    "AgentSpec",
    "AtomicAgentsProvider",
    "ChatMessage",
    "CompletionResult",
    "LLMConfig",
    "LLMProvider",
    "LLMProviderType",
    "MockLLMProviderAdapter",
    "TextResponse",
    "ToolSpec",
    "agent_create",
    "agent_run",
    "agent_spawn",
    "create_provider",
    "get_output_schema",
    "register_output_schema",
]
