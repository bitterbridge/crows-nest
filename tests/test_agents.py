"""Tests for agent factory and operations."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from crows_nest.agents import (
    Agent,
    AgentContext,
    AgentFactory,
    AgentSpec,
    LLMConfig,
    TextResponse,
    ToolSpec,
    agent_create,
    agent_run,
    agent_spawn,
    get_output_schema,
    register_output_schema,
)
from crows_nest.core.thunk import ThunkMetadata


class CustomResponse(BaseModel):
    """Custom response schema for tests."""

    result: str
    score: float


class TestToolSpec:
    """Tests for ToolSpec."""

    def test_create_tool_spec(self):
        """Should create tool spec."""
        spec = ToolSpec(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            required_capabilities=["web_access"],
        )

        assert spec.name == "search"
        assert spec.description == "Search the web"
        assert spec.required_capabilities == ["web_access"]

    def test_default_values(self):
        """Should have sensible defaults."""
        spec = ToolSpec(name="test", description="Test tool")

        assert spec.input_schema == {}
        assert spec.required_capabilities == []


class TestAgentSpec:
    """Tests for AgentSpec."""

    def test_create_agent_spec(self):
        """Should create agent spec."""
        spec = AgentSpec(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a helpful assistant.",
        )

        assert spec.name == "assistant"
        assert spec.description == "A helpful assistant"
        assert spec.max_iterations == 10

    def test_spec_with_tools(self):
        """Should accept tools list."""
        tool = ToolSpec(name="search", description="Search")
        spec = AgentSpec(
            name="researcher",
            description="Research agent",
            system_prompt="Research things.",
            tools=[tool],
        )

        assert len(spec.tools) == 1
        assert spec.tools[0].name == "search"

    def test_to_dict(self):
        """Should serialize to dict."""
        spec = AgentSpec(
            name="test",
            description="Test",
            system_prompt="Test prompt",
            required_capabilities=["read"],
        )

        data = spec.to_dict()

        assert data["name"] == "test"
        assert data["required_capabilities"] == ["read"]

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "name": "test",
            "description": "Test",
            "system_prompt": "Hello",
            "required_capabilities": ["write"],
        }

        spec = AgentSpec.from_dict(data)

        assert spec.name == "test"
        assert spec.required_capabilities == ["write"]


class TestAgentContext:
    """Tests for AgentContext."""

    def test_create_root_context(self):
        """Should create root context."""
        ctx = AgentContext.create(
            capabilities=frozenset(["read", "write"]),
            trace_id="trace-123",
        )

        assert ctx.parent_id is None
        assert ctx.capabilities == frozenset(["read", "write"])
        assert ctx.trace_id == "trace-123"

    def test_create_generates_ids(self):
        """Should generate agent_id and trace_id if not provided."""
        ctx = AgentContext.create()

        assert ctx.agent_id is not None
        assert ctx.trace_id is not None

    def test_with_restricted_capabilities(self):
        """Should restrict capabilities."""
        ctx = AgentContext.create(capabilities=frozenset(["read", "write", "delete"]))

        restricted = ctx.with_restricted_capabilities(frozenset(["read", "write"]))

        assert restricted.capabilities == frozenset(["read", "write"])
        assert ctx.capabilities == frozenset(["read", "write", "delete"])  # Original unchanged

    def test_for_child(self):
        """Should create child context with inherited capabilities."""
        parent = AgentContext.create(capabilities=frozenset(["read", "write"]))
        child_id = uuid4()

        child = parent.for_child(child_id)

        assert child.agent_id == child_id
        assert child.parent_id == parent.agent_id
        assert child.capabilities == parent.capabilities
        assert child.trace_id == parent.trace_id

    def test_for_child_with_restrictions(self):
        """Should restrict child capabilities."""
        parent = AgentContext.create(capabilities=frozenset(["read", "write", "delete"]))
        child_id = uuid4()

        child = parent.for_child(child_id, restrict_capabilities=frozenset(["read"]))

        assert child.capabilities == frozenset(["read"])

    def test_to_dict(self):
        """Should serialize to dict."""
        ctx = AgentContext.create(capabilities=frozenset(["read"]))

        data = ctx.to_dict()

        assert "agent_id" in data
        assert "capabilities" in data
        assert "read" in data["capabilities"]

    def test_from_dict(self):
        """Should deserialize from dict."""
        agent_id = str(uuid4())
        data = {
            "agent_id": agent_id,
            "parent_id": None,
            "capabilities": ["read", "write"],
            "trace_id": "trace-1",
            "variables": {"key": "value"},
        }

        ctx = AgentContext.from_dict(data)

        assert str(ctx.agent_id) == agent_id
        assert ctx.capabilities == frozenset(["read", "write"])
        assert ctx.variables == {"key": "value"}

    def test_from_thunk_metadata(self):
        """Should create from ThunkMetadata."""
        from datetime import UTC, datetime

        metadata = ThunkMetadata(
            created_at=datetime.now(UTC),
            created_by="user",
            trace_id="trace-123",
            capabilities=frozenset(["read"]),
            tags=frozenset([("env", "test")]),
        )

        ctx = AgentContext.from_thunk_metadata(metadata)

        assert ctx.capabilities == frozenset(["read"])
        assert ctx.trace_id == "trace-123"
        assert ctx.parent_id is None  # "user" is not a UUID

    def test_from_thunk_metadata_with_parent(self):
        """Should parse parent_id from created_by if valid UUID."""
        from datetime import UTC, datetime

        parent_id = uuid4()
        metadata = ThunkMetadata(
            created_at=datetime.now(UTC),
            created_by=str(parent_id),
            trace_id="trace-123",
            capabilities=frozenset(["read"]),
        )

        ctx = AgentContext.from_thunk_metadata(metadata)

        assert ctx.parent_id == parent_id


class TestAgent:
    """Tests for Agent."""

    @pytest.fixture
    def basic_spec(self):
        """Create a basic agent spec."""
        return AgentSpec(
            name="assistant",
            description="Test assistant",
            system_prompt="You are helpful. {{name}} is the user.",
            tools=[
                ToolSpec(
                    name="search",
                    description="Search",
                    required_capabilities=["web"],
                ),
                ToolSpec(
                    name="write",
                    description="Write files",
                    required_capabilities=["file_write"],
                ),
            ],
        )

    @pytest.fixture
    def basic_context(self):
        """Create a basic context."""
        return AgentContext.create(
            capabilities=frozenset(["web"]),
            variables={"name": "Alice"},
        )

    @pytest.fixture
    def basic_agent(self, basic_spec, basic_context):
        """Create a basic agent."""
        from crows_nest.agents.llm import create_provider

        config = LLMConfig.mock(seed=42)
        provider = create_provider(config)
        return Agent(spec=basic_spec, provider=provider, context=basic_context)

    def test_can_use_tool(self, basic_agent):
        """Should check if agent can use tool."""
        search_tool = basic_agent.spec.tools[0]  # requires "web"
        write_tool = basic_agent.spec.tools[1]  # requires "file_write"

        assert basic_agent.can_use_tool(search_tool)
        assert not basic_agent.can_use_tool(write_tool)

    def test_available_tools(self, basic_agent):
        """Should filter tools by capabilities."""
        available = basic_agent.available_tools()

        assert len(available) == 1
        assert available[0].name == "search"

    def test_render_system_prompt(self, basic_agent):
        """Should render system prompt with variables."""
        prompt = basic_agent.render_system_prompt()

        assert "Alice" in prompt
        assert "{{name}}" not in prompt

    def test_create_child_context(self, basic_agent):
        """Should create child context."""
        child_ctx = basic_agent.create_child_context()

        assert child_ctx.parent_id == basic_agent.context.agent_id
        assert child_ctx.capabilities == basic_agent.context.capabilities

    def test_create_child_context_restricted(self, basic_agent):
        """Should restrict child context capabilities."""
        child_ctx = basic_agent.create_child_context(restrict_capabilities=frozenset(["read"]))

        # Intersection of {"web"} and {"read"} = {}
        assert child_ctx.capabilities == frozenset()


class TestAgentFactory:
    """Tests for AgentFactory."""

    @pytest.fixture
    def factory(self):
        """Create a factory with mock provider."""
        return AgentFactory(llm_config=LLMConfig.mock(seed=42))

    @pytest.fixture
    def assistant_spec(self):
        """Create an assistant spec."""
        return AgentSpec(
            name="assistant",
            description="Test assistant",
            system_prompt="You are helpful.",
            required_capabilities=["read"],
        )

    def test_register_spec(self, factory, assistant_spec):
        """Should register agent spec."""
        factory.register(assistant_spec)

        assert "assistant" in factory.list_specs()

    def test_register_duplicate_raises(self, factory, assistant_spec):
        """Should raise on duplicate registration."""
        factory.register(assistant_spec)

        with pytest.raises(ValueError, match="already registered"):
            factory.register(assistant_spec)

    def test_get_spec(self, factory, assistant_spec):
        """Should get registered spec."""
        factory.register(assistant_spec)

        spec = factory.get_spec("assistant")

        assert spec.name == "assistant"

    def test_get_spec_not_found(self, factory):
        """Should raise for unknown spec."""
        with pytest.raises(KeyError, match="No agent spec registered"):
            factory.get_spec("unknown")

    def test_list_specs(self, factory, assistant_spec):
        """Should list all specs."""
        factory.register(assistant_spec)
        factory.register(AgentSpec(name="researcher", description="R", system_prompt="P"))

        specs = factory.list_specs()

        assert "assistant" in specs
        assert "researcher" in specs

    def test_create_agent(self, factory, assistant_spec):
        """Should create agent instance."""
        factory.register(assistant_spec)
        ctx = AgentContext.create(capabilities=frozenset(["read", "write"]))

        agent = factory.create("assistant", ctx)

        assert agent.spec.name == "assistant"
        assert agent.context.capabilities == frozenset(["read", "write"])

    def test_create_agent_missing_capabilities(self, factory, assistant_spec):
        """Should raise if context lacks required capabilities."""
        factory.register(assistant_spec)
        ctx = AgentContext.create(capabilities=frozenset())  # No capabilities

        with pytest.raises(PermissionError, match="Missing capabilities"):
            factory.create("assistant", ctx)

    def test_create_from_spec(self, factory, assistant_spec):
        """Should create agent from unregistered spec."""
        ctx = AgentContext.create(capabilities=frozenset(["read"]))

        agent = factory.create_from_spec(assistant_spec, ctx)

        assert agent.spec.name == "assistant"

    def test_create_child(self, factory, assistant_spec):
        """Should create child agent."""
        factory.register(assistant_spec)
        factory.register(AgentSpec(name="helper", description="H", system_prompt="P"))
        ctx = AgentContext.create(capabilities=frozenset(["read"]))
        parent = factory.create("assistant", ctx)

        child = factory.create_child(parent, "helper")

        assert child.context.parent_id == parent.context.agent_id
        assert child.context.capabilities == parent.context.capabilities


class TestOutputSchemaRegistry:
    """Tests for output schema registry."""

    def test_get_default_schema(self):
        """Should have TextResponse registered by default."""
        schema = get_output_schema("TextResponse")

        assert schema == TextResponse

    def test_register_custom_schema(self):
        """Should register custom schemas."""
        register_output_schema("CustomResponse", CustomResponse)

        schema = get_output_schema("CustomResponse")

        assert schema == CustomResponse

    def test_get_unknown_schema_raises(self):
        """Should raise for unknown schema."""
        with pytest.raises(KeyError, match="not registered"):
            get_output_schema("UnknownSchema")


class TestAgentOperations:
    """Tests for agent thunk operations."""

    @pytest.fixture
    def basic_spec_dict(self):
        """Create a basic spec dict."""
        return {
            "name": "test_agent",
            "description": "Test agent",
            "system_prompt": "You are helpful.",
            "required_capabilities": [],
        }

    @pytest.fixture
    def basic_context_dict(self):
        """Create a basic context dict."""
        return {
            "agent_id": str(uuid4()),
            "parent_id": None,
            "capabilities": ["read", "write"],
            "trace_id": "trace-123",
            "variables": {},
        }

    @pytest.mark.asyncio
    async def test_agent_create(self, basic_spec_dict, basic_context_dict):
        """Should create agent and return info."""
        result = await agent_create(
            spec=basic_spec_dict,
            context=basic_context_dict,
        )

        assert "agent_id" in result
        assert result["spec_name"] == "test_agent"
        assert "read" in result["capabilities"]
        assert "write" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_agent_create_with_llm_config(self, basic_spec_dict, basic_context_dict):
        """Should accept LLM config override."""
        result = await agent_create(
            spec=basic_spec_dict,
            context=basic_context_dict,
            llm_config={"provider": "mock", "model": "test-model"},
        )

        assert "agent_id" in result

    @pytest.mark.asyncio
    async def test_agent_run(self, basic_spec_dict, basic_context_dict):
        """Should run agent and return response."""
        result = await agent_run(
            spec=basic_spec_dict,
            context=basic_context_dict,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert "agent_id" in result
        assert "response" in result
        # Mock provider generates random content
        assert "content" in result["response"]

    @pytest.mark.asyncio
    async def test_agent_spawn(self, basic_spec_dict, basic_context_dict):
        """Should spawn child agent."""
        result = await agent_spawn(
            parent_context=basic_context_dict,
            child_spec=basic_spec_dict,
        )

        assert "agent_id" in result
        assert result["spec_name"] == "test_agent"
        assert "read" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_agent_spawn_with_restrictions(self, basic_spec_dict, basic_context_dict):
        """Should restrict child capabilities."""
        result = await agent_spawn(
            parent_context=basic_context_dict,
            child_spec=basic_spec_dict,
            restrict_capabilities=["read"],
        )

        # Intersection of ["read", "write"] and ["read"] = ["read"]
        assert result["capabilities"] == ["read"]
