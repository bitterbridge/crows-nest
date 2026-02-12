"""Agent factory and operations."""

# Import spawning to register operations
from crows_nest.agents import spawning as _spawning  # noqa: F401
from crows_nest.agents.budget import (
    AgentResources,
    BudgetError,
    CapabilityBudget,
    CapabilityNotAllowedError,
    InsufficientCapabilitiesError,
    QuotaExceededError,
    ResourceQuota,
)
from crows_nest.agents.committee import (
    EvaluatorSpec,
    EvaluatorVote,
    HiringCommittee,
    SpawnDecision,
    SpawnDecisionStatus,
    SpawnRequest,
    SpawnVote,
    get_hiring_committee,
    reset_hiring_committee,
    set_hiring_committee,
)
from crows_nest.agents.factory import (
    Agent,
    AgentContext,
    AgentFactory,
    AgentSpec,
    ToolSpec,
)
from crows_nest.agents.hierarchy import (
    AgentHierarchyError,
    AgentLineage,
    AgentNotFoundError,
    AgentRegistry,
    MaxChildrenExceededError,
    MaxDepthExceededError,
    get_agent_registry,
    reset_agent_registry,
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
    # Factory
    "Agent",
    "AgentContext",
    # Operations
    "AgentCreateInput",
    "AgentCreateOutput",
    "AgentFactory",
    # Hierarchy
    "AgentHierarchyError",
    "AgentLineage",
    "AgentNotFoundError",
    "AgentRegistry",
    # Budget
    "AgentResources",
    "AgentRunInput",
    "AgentRunOutput",
    "AgentSpec",
    # LLM
    "AtomicAgentsProvider",
    "BudgetError",
    "CapabilityBudget",
    "CapabilityNotAllowedError",
    "ChatMessage",
    "CompletionResult",
    # Committee
    "EvaluatorSpec",
    "EvaluatorVote",
    "HiringCommittee",
    "InsufficientCapabilitiesError",
    "LLMConfig",
    "LLMProvider",
    "LLMProviderType",
    "MaxChildrenExceededError",
    "MaxDepthExceededError",
    "MockLLMProviderAdapter",
    "QuotaExceededError",
    "ResourceQuota",
    "SpawnDecision",
    "SpawnDecisionStatus",
    "SpawnRequest",
    "SpawnVote",
    "TextResponse",
    "ToolSpec",
    "agent_create",
    "agent_run",
    "agent_spawn",
    "create_provider",
    "get_agent_registry",
    "get_hiring_committee",
    "get_output_schema",
    "register_output_schema",
    "reset_agent_registry",
    "reset_hiring_committee",
    "set_hiring_committee",
]
