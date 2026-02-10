"""Agent thunk operations.

Provides thunk operations for creating and running agents. These operations
can be composed in the thunk graph like any other computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crows_nest.agents.factory import (
    AgentContext,
    AgentFactory,
    AgentSpec,
)
from crows_nest.agents.llm import ChatMessage, LLMConfig, LLMProviderType, create_provider
from crows_nest.core.registry import thunk_operation

if TYPE_CHECKING:
    from collections.abc import Sequence


class AgentCreateInput(BaseModel):
    """Input schema for agent.create operation."""

    spec: dict[str, Any] = Field(description="Agent specification as dict")
    context: dict[str, Any] = Field(description="Agent context as dict")
    llm_config: dict[str, Any] | None = Field(
        default=None, description="Optional LLM config override"
    )


class AgentCreateOutput(BaseModel):
    """Output schema for agent.create operation."""

    agent_id: str = Field(description="ID of the created agent")
    spec_name: str = Field(description="Name of the agent spec")
    capabilities: list[str] = Field(description="Agent capabilities")
    available_tools: list[str] = Field(description="Tools available to this agent")


class AgentRunInput(BaseModel):
    """Input schema for agent.run operation."""

    spec: dict[str, Any] = Field(description="Agent specification")
    context: dict[str, Any] = Field(description="Agent context")
    messages: list[dict[str, str]] = Field(description="Conversation messages")
    output_schema_name: str = Field(default="TextResponse", description="Name of output schema")
    llm_config: dict[str, Any] | None = Field(
        default=None, description="Optional LLM config override"
    )


class TextResponse(BaseModel):
    """Default text response schema for agent runs."""

    content: str = Field(description="The response content")
    reasoning: str | None = Field(default=None, description="Optional reasoning")


class AgentRunOutput(BaseModel):
    """Output schema for agent.run operation."""

    agent_id: str = Field(description="ID of the agent that ran")
    response: dict[str, Any] = Field(description="The agent's response")
    model: str | None = Field(default=None, description="Model used")


# Registry of output schemas that can be used by agents
_output_schemas: dict[str, type[BaseModel]] = {
    "TextResponse": TextResponse,
}


def register_output_schema(name: str, schema: type[BaseModel]) -> None:
    """Register an output schema for use by agents.

    Args:
        name: Schema name for reference in operations.
        schema: Pydantic model class.
    """
    _output_schemas[name] = schema


def get_output_schema(name: str) -> type[BaseModel]:
    """Get a registered output schema by name.

    Args:
        name: The schema name.

    Returns:
        The Pydantic model class.

    Raises:
        KeyError: If schema is not registered.
    """
    if name not in _output_schemas:
        msg = f"Output schema '{name}' not registered"
        raise KeyError(msg)
    return _output_schemas[name]


@thunk_operation(
    "agent.create",
    description="Create an agent instance from specification and context",
    required_capabilities=frozenset({"agent.create"}),
)
async def agent_create(
    spec: dict[str, Any],
    context: dict[str, Any],
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent instance.

    This operation creates an agent from a specification and context,
    returning information about the created agent.

    Args:
        spec: Agent specification dict (will be parsed as AgentSpec).
        context: Agent context dict (will be parsed as AgentContext).
        llm_config: Optional LLM configuration override.

    Returns:
        Dict with agent information (agent_id, spec_name, capabilities, tools).
    """
    agent_spec = AgentSpec.model_validate(spec)
    agent_context = AgentContext.from_dict(context)

    # Create LLM config if provided
    config = None
    if llm_config:
        provider_type = LLMProviderType(llm_config.get("provider", "mock"))
        config = LLMConfig(
            provider=provider_type,
            model=llm_config.get("model", "claude-sonnet-4-20250514"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 1000),
            temperature=llm_config.get("temperature", 0.7),
            extra=llm_config.get("extra", {}),
        )

    # Create factory and agent
    factory = AgentFactory(llm_config=config)
    agent = factory.create_from_spec(agent_spec, agent_context)

    return AgentCreateOutput(
        agent_id=str(agent.context.agent_id),
        spec_name=agent.spec.name,
        capabilities=sorted(agent.context.capabilities),
        available_tools=[t.name for t in agent.available_tools()],
    ).model_dump()


@thunk_operation(
    "agent.run",
    description="Run an agent with messages and get a response",
    required_capabilities=frozenset({"agent.run"}),
)
async def agent_run(
    spec: dict[str, Any],
    context: dict[str, Any],
    messages: Sequence[dict[str, str]],
    output_schema_name: str = "TextResponse",
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an agent and get a response.

    This operation creates an agent, sends it messages, and returns
    the structured response.

    Args:
        spec: Agent specification dict.
        context: Agent context dict.
        messages: List of message dicts with 'role' and 'content'.
        output_schema_name: Name of the output schema to use.
        llm_config: Optional LLM configuration override.

    Returns:
        Dict with agent_id, response content, and model used.
    """
    agent_spec = AgentSpec.model_validate(spec)
    agent_context = AgentContext.from_dict(context)
    output_schema = get_output_schema(output_schema_name)

    # Build LLM config
    config = LLMConfig.mock()
    if llm_config:
        provider_type = LLMProviderType(llm_config.get("provider", "mock"))
        config = LLMConfig(
            provider=provider_type,
            model=llm_config.get("model", "claude-sonnet-4-20250514"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 1000),
            temperature=llm_config.get("temperature", 0.7),
            extra=llm_config.get("extra", {}),
        )

    # Create provider and agent
    provider = create_provider(config)
    factory = AgentFactory(llm_config=config)
    agent = factory.create_from_spec(agent_spec, agent_context, provider=provider)

    # Convert messages to ChatMessage objects
    chat_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

    # Get the system prompt
    system_prompt = agent.render_system_prompt()

    # Run completion
    result = await provider.complete_async(
        chat_messages,
        output_schema,
        system_prompt=system_prompt,
    )

    # Extract response content
    response_content = result.content
    if hasattr(result.content, "model_dump"):
        response_content = result.content.model_dump()

    return AgentRunOutput(
        agent_id=str(agent.context.agent_id),
        response=response_content,
        model=result.model,
    ).model_dump()


@thunk_operation(
    "agent.spawn",
    description="Create a child agent from a parent agent context",
    required_capabilities=frozenset({"agent.spawn"}),
)
async def agent_spawn(
    parent_context: dict[str, Any],
    child_spec: dict[str, Any],
    restrict_capabilities: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn a child agent from a parent context.

    Creates a new agent that inherits capabilities from the parent,
    optionally restricted to a subset.

    Args:
        parent_context: Parent agent's context dict.
        child_spec: Specification for the child agent.
        restrict_capabilities: Optional list of capabilities to restrict to.
        llm_config: Optional LLM configuration override.

    Returns:
        Dict with child agent information.
    """
    parent_ctx = AgentContext.from_dict(parent_context)
    child_spec_obj = AgentSpec.model_validate(child_spec)

    # Create child context with inherited capabilities
    restrict = frozenset(restrict_capabilities) if restrict_capabilities else None
    child_ctx = parent_ctx.for_child(uuid4(), restrict_capabilities=restrict)

    # Build LLM config
    config = LLMConfig.mock()
    if llm_config:
        provider_type = LLMProviderType(llm_config.get("provider", "mock"))
        config = LLMConfig(
            provider=provider_type,
            model=llm_config.get("model", "claude-sonnet-4-20250514"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 1000),
            temperature=llm_config.get("temperature", 0.7),
            extra=llm_config.get("extra", {}),
        )

    # Create the child agent
    factory = AgentFactory(llm_config=config)
    child_agent = factory.create_from_spec(child_spec_obj, child_ctx)

    return AgentCreateOutput(
        agent_id=str(child_agent.context.agent_id),
        spec_name=child_agent.spec.name,
        capabilities=sorted(child_agent.context.capabilities),
        available_tools=[t.name for t in child_agent.available_tools()],
    ).model_dump()
