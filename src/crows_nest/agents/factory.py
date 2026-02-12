"""Agent factory for dynamic agent creation.

Provides AgentFactory which builds agents from configuration, supporting
capability inheritance where child agents receive a subset of parent capabilities.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from crows_nest.agents.llm import LLMProvider, create_provider
from crows_nest.llm.providers import MockConfig, ProviderConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crows_nest.core.thunk import ThunkMetadata


class ToolSpec(BaseModel):
    """Specification for an agent tool.

    Attributes:
        name: Tool name (used in prompts and responses).
        description: Human-readable description.
        input_schema: JSON Schema for tool input validation.
        required_capabilities: Capabilities needed to use this tool.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for inputs")
    required_capabilities: list[str] = Field(
        default_factory=list, description="Required capabilities"
    )


class AgentSpec(BaseModel):
    """Specification defining an agent's behavior.

    AgentSpecs are immutable definitions that describe what an agent does.
    They can be serialized and stored, then used to create Agent instances.

    Attributes:
        name: Unique name for this agent type.
        description: Human-readable description.
        system_prompt: Base system prompt for the agent.
        output_schema: Pydantic model class name for structured output.
        tools: Available tools for this agent.
        required_capabilities: Capabilities needed to run this agent.
        max_iterations: Maximum iterations for agent loops.
    """

    name: str = Field(description="Agent type name")
    description: str = Field(description="Agent description")
    system_prompt: str = Field(description="System prompt template")
    output_schema: str | None = Field(default=None, description="Output schema class name")
    tools: list[ToolSpec] = Field(default_factory=list, description="Available tools")
    required_capabilities: list[str] = Field(
        default_factory=list, description="Required capabilities"
    )
    max_iterations: int = Field(default=10, description="Max iterations")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Deserialize from dict."""
        return cls.model_validate(data)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Runtime context for an agent execution.

    Agents are stateless; all state is passed through context.

    Attributes:
        agent_id: Unique ID for this agent instance.
        parent_id: ID of the agent that created this one (None for root).
        capabilities: Capabilities this agent is allowed to use.
        trace_id: Correlation ID for distributed tracing.
        variables: Dynamic variables for prompt templating.
    """

    agent_id: UUID
    parent_id: UUID | None
    capabilities: frozenset[str]
    trace_id: str
    variables: dict[str, Any] = field(default_factory=dict)

    def with_restricted_capabilities(self, allowed: frozenset[str]) -> AgentContext:
        """Create new context with capabilities restricted to allowed set."""
        return AgentContext(
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            capabilities=self.capabilities & allowed,
            trace_id=self.trace_id,
            variables=self.variables,
        )

    def for_child(
        self,
        child_id: UUID,
        *,
        restrict_capabilities: frozenset[str] | None = None,
    ) -> AgentContext:
        """Create context for a child agent with inherited capabilities."""
        caps = self.capabilities
        if restrict_capabilities is not None:
            caps = caps & restrict_capabilities
        return AgentContext(
            agent_id=child_id,
            parent_id=self.agent_id,
            capabilities=caps,
            trace_id=self.trace_id,
            variables=self.variables.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "agent_id": str(self.agent_id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "capabilities": sorted(self.capabilities),
            "trace_id": self.trace_id,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentContext:
        """Deserialize from dict."""
        return cls(
            agent_id=UUID(data["agent_id"]),
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
            capabilities=frozenset(data.get("capabilities", [])),
            trace_id=data["trace_id"],
            variables=dict(data.get("variables", {})),
        )

    @classmethod
    def create(
        cls,
        *,
        capabilities: frozenset[str] | None = None,
        trace_id: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Create a new root context."""
        return cls(
            agent_id=uuid4(),
            parent_id=None,
            capabilities=capabilities or frozenset(),
            trace_id=trace_id or str(uuid4()),
            variables=variables or {},
        )

    @classmethod
    def from_thunk_metadata(cls, metadata: ThunkMetadata) -> AgentContext:
        """Create context from thunk metadata."""
        parent_id = None
        if metadata.created_by and metadata.created_by != "user":
            with contextlib.suppress(ValueError):
                parent_id = UUID(metadata.created_by)

        return cls(
            agent_id=uuid4(),
            parent_id=parent_id,
            capabilities=metadata.capabilities,
            trace_id=metadata.trace_id,
            variables=dict(metadata.tags),
        )


@dataclass
class Agent:
    """An executable agent instance.

    Agents are stateless executors - they take a context and messages,
    and produce a response. All state management is external.

    Attributes:
        spec: The agent specification.
        provider: LLM provider for completions.
        context: Runtime context with capabilities and variables.
    """

    spec: AgentSpec
    provider: LLMProvider
    context: AgentContext

    def can_use_tool(self, tool: ToolSpec) -> bool:
        """Check if this agent has capabilities to use a tool."""
        required = frozenset(tool.required_capabilities)
        return required <= self.context.capabilities

    def available_tools(self) -> list[ToolSpec]:
        """Get tools this agent can use based on capabilities."""
        return [t for t in self.spec.tools if self.can_use_tool(t)]

    def render_system_prompt(self) -> str:
        """Render the system prompt with context variables."""
        prompt = self.spec.system_prompt
        for key, value in self.context.variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        return prompt

    def create_child_context(
        self,
        restrict_capabilities: frozenset[str] | None = None,
    ) -> AgentContext:
        """Create a context for spawning a child agent."""
        return self.context.for_child(
            uuid4(),
            restrict_capabilities=restrict_capabilities,
        )


class AgentFactory:
    """Factory for creating agents from specifications.

    The factory manages agent specs and creates Agent instances with
    appropriate LLM providers and contexts.

    Example:
        factory = AgentFactory(provider_config=MockConfig())
        factory.register(AgentSpec(name="assistant", ...))

        context = AgentContext.create(capabilities=frozenset(["read", "write"]))
        agent = factory.create("assistant", context)
    """

    def __init__(
        self,
        provider_config: ProviderConfig | None = None,
        specs: Sequence[AgentSpec] | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            provider_config: Default provider configuration for created agents.
            specs: Initial agent specifications to register.
        """
        self._provider_config = provider_config or MockConfig()
        self._specs: dict[str, AgentSpec] = {}
        self._provider: LLMProvider | None = None

        if specs:
            for spec in specs:
                self.register(spec)

    @property
    def provider(self) -> LLMProvider:
        """Get or create the LLM provider (lazy initialization)."""
        if self._provider is None:
            self._provider = create_provider(self._provider_config)
        return self._provider

    def register(self, spec: AgentSpec) -> None:
        """Register an agent specification.

        Args:
            spec: The agent spec to register.

        Raises:
            ValueError: If a spec with the same name is already registered.
        """
        if spec.name in self._specs:
            msg = f"Agent spec '{spec.name}' is already registered"
            raise ValueError(msg)
        self._specs[spec.name] = spec

    def get_spec(self, name: str) -> AgentSpec:
        """Get a registered agent specification.

        Args:
            name: The spec name.

        Returns:
            The agent spec.

        Raises:
            KeyError: If no spec with that name is registered.
        """
        if name not in self._specs:
            msg = f"No agent spec registered with name '{name}'"
            raise KeyError(msg)
        return self._specs[name]

    def list_specs(self) -> list[str]:
        """List all registered spec names."""
        return list(self._specs.keys())

    def create(
        self,
        spec_name: str,
        context: AgentContext,
        *,
        provider: LLMProvider | None = None,
    ) -> Agent:
        """Create an agent instance.

        Args:
            spec_name: Name of the registered spec to use.
            context: Runtime context for the agent.
            provider: Optional override LLM provider.

        Returns:
            A new Agent instance.

        Raises:
            KeyError: If spec_name is not registered.
            PermissionError: If context lacks required capabilities.
        """
        spec = self.get_spec(spec_name)

        # Check capability requirements
        required = frozenset(spec.required_capabilities)
        if not required <= context.capabilities:
            missing = required - context.capabilities
            msg = f"Missing capabilities for agent '{spec_name}': {missing}"
            raise PermissionError(msg)

        return Agent(
            spec=spec,
            provider=provider or self.provider,
            context=context,
        )

    def create_from_spec(
        self,
        spec: AgentSpec,
        context: AgentContext,
        *,
        provider: LLMProvider | None = None,
    ) -> Agent:
        """Create an agent from an unregistered spec.

        Useful for dynamic agent creation where specs aren't pre-registered.

        Args:
            spec: The agent spec to use.
            context: Runtime context for the agent.
            provider: Optional override LLM provider.

        Returns:
            A new Agent instance.

        Raises:
            PermissionError: If context lacks required capabilities.
        """
        required = frozenset(spec.required_capabilities)
        if not required <= context.capabilities:
            missing = required - context.capabilities
            msg = f"Missing capabilities for agent '{spec.name}': {missing}"
            raise PermissionError(msg)

        return Agent(
            spec=spec,
            provider=provider or self.provider,
            context=context,
        )

    def create_child(
        self,
        parent: Agent,
        spec_name: str,
        *,
        restrict_capabilities: frozenset[str] | None = None,
    ) -> Agent:
        """Create a child agent from a parent agent.

        The child inherits capabilities from the parent (restricted by
        the inherit-and-restrict model).

        Args:
            parent: The parent agent.
            spec_name: Name of the spec for the child.
            restrict_capabilities: Optional further capability restrictions.

        Returns:
            A new child Agent instance.
        """
        child_context = parent.create_child_context(restrict_capabilities)
        return self.create(spec_name, child_context)
