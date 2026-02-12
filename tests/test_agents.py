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
    TextResponse,
    ToolSpec,
    agent_create,
    agent_run,
    agent_spawn,
    get_output_schema,
    register_output_schema,
)
from crows_nest.core.thunk import ThunkMetadata
from crows_nest.llm import MockConfig


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

        config = MockConfig()
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
        return AgentFactory(provider_config=MockConfig())

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


# =============================================================================
# Stage 8: Agent Hierarchy and Spawning Tests
# =============================================================================


class TestAgentLineage:
    """Tests for AgentLineage dataclass."""

    def test_create_lineage(self):
        """Should create agent lineage."""
        from crows_nest.agents import AgentLineage

        agent_id = uuid4()
        lineage = AgentLineage(agent_id=agent_id)

        assert lineage.agent_id == agent_id
        assert lineage.parent_id is None
        assert lineage.children == []
        assert lineage.depth == 0
        assert lineage.is_active
        assert lineage.is_root

    def test_lineage_with_parent(self):
        """Should track parent relationship."""
        from crows_nest.agents import AgentLineage

        parent_id = uuid4()
        child_id = uuid4()
        lineage = AgentLineage(agent_id=child_id, parent_id=parent_id, depth=1)

        assert lineage.parent_id == parent_id
        assert lineage.depth == 1
        assert not lineage.is_root

    def test_lineage_terminated(self):
        """Should track termination status."""
        from datetime import UTC, datetime

        from crows_nest.agents import AgentLineage

        lineage = AgentLineage(agent_id=uuid4(), terminated_at=datetime.now(UTC))

        assert not lineage.is_active

    def test_lineage_to_dict(self):
        """Should serialize to dictionary."""
        from crows_nest.agents import AgentLineage

        lineage = AgentLineage(
            agent_id=uuid4(),
            metadata={"role": "worker"},
        )

        data = lineage.to_dict()

        assert "agent_id" in data
        assert data["metadata"]["role"] == "worker"
        assert data["parent_id"] is None

    def test_lineage_from_dict(self):
        """Should deserialize from dictionary."""
        from crows_nest.agents import AgentLineage

        agent_id = uuid4()
        data = {
            "agent_id": str(agent_id),
            "parent_id": None,
            "children": [],
            "depth": 0,
            "created_at": "2025-01-01T00:00:00+00:00",
            "terminated_at": None,
            "metadata": {"key": "value"},
        }

        lineage = AgentLineage.from_dict(data)

        assert lineage.agent_id == agent_id
        assert lineage.metadata == {"key": "value"}


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    @pytest.fixture
    async def registry(self):
        """Create a fresh registry."""
        from crows_nest.agents import AgentRegistry

        reg = AgentRegistry(max_depth=5, max_children=3)
        await reg.initialize()
        yield reg
        await reg.close()

    @pytest.mark.asyncio
    async def test_register_root_agent(self, registry):
        """Should register root agent."""
        lineage = await registry.register()

        assert lineage.is_root
        assert lineage.depth == 0

    @pytest.mark.asyncio
    async def test_register_with_custom_id(self, registry):
        """Should accept custom agent ID."""
        agent_id = uuid4()
        lineage = await registry.register(agent_id=agent_id)

        assert lineage.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_register_child_agent(self, registry):
        """Should register child agent."""
        parent = await registry.register()
        child = await registry.register(parent_id=parent.agent_id)

        assert child.parent_id == parent.agent_id
        assert child.depth == 1

        # Parent should have child in children list
        updated_parent = await registry.get_lineage(parent.agent_id)
        assert child.agent_id in updated_parent.children

    @pytest.mark.asyncio
    async def test_register_max_depth_exceeded(self, registry):
        """Should raise when max depth exceeded."""
        from crows_nest.agents import MaxDepthExceededError

        # Create chain up to max depth
        current = await registry.register()
        for _ in range(5):  # max_depth=5
            current = await registry.register(parent_id=current.agent_id)

        # Next registration should fail
        with pytest.raises(MaxDepthExceededError):
            await registry.register(parent_id=current.agent_id)

    @pytest.mark.asyncio
    async def test_register_max_children_exceeded(self, registry):
        """Should raise when max children exceeded."""
        from crows_nest.agents import MaxChildrenExceededError

        parent = await registry.register()

        # Create children up to max
        for _ in range(3):  # max_children=3
            await registry.register(parent_id=parent.agent_id)

        # Next child should fail
        with pytest.raises(MaxChildrenExceededError):
            await registry.register(parent_id=parent.agent_id)

    @pytest.mark.asyncio
    async def test_get_lineage(self, registry):
        """Should get lineage by ID."""
        lineage = await registry.register()
        retrieved = await registry.get_lineage(lineage.agent_id)

        assert retrieved.agent_id == lineage.agent_id

    @pytest.mark.asyncio
    async def test_get_lineage_not_found(self, registry):
        """Should raise for unknown agent."""
        from crows_nest.agents import AgentNotFoundError

        with pytest.raises(AgentNotFoundError):
            await registry.get_lineage(uuid4())

    @pytest.mark.asyncio
    async def test_get_children(self, registry):
        """Should get all children."""
        parent = await registry.register()
        child1 = await registry.register(parent_id=parent.agent_id)
        child2 = await registry.register(parent_id=parent.agent_id)

        children = await registry.get_children(parent.agent_id)

        assert len(children) == 2
        child_ids = [c.agent_id for c in children]
        assert child1.agent_id in child_ids
        assert child2.agent_id in child_ids

    @pytest.mark.asyncio
    async def test_get_ancestors(self, registry):
        """Should get all ancestors."""
        grandparent = await registry.register()
        parent = await registry.register(parent_id=grandparent.agent_id)
        child = await registry.register(parent_id=parent.agent_id)

        ancestors = await registry.get_ancestors(child.agent_id)

        assert len(ancestors) == 2
        assert ancestors[0].agent_id == parent.agent_id
        assert ancestors[1].agent_id == grandparent.agent_id

    @pytest.mark.asyncio
    async def test_get_descendants(self, registry):
        """Should get all descendants."""
        root = await registry.register()
        child1 = await registry.register(parent_id=root.agent_id)
        child2 = await registry.register(parent_id=root.agent_id)
        grandchild = await registry.register(parent_id=child1.agent_id)

        descendants = await registry.get_descendants(root.agent_id)

        assert len(descendants) == 3
        desc_ids = [d.agent_id for d in descendants]
        assert child1.agent_id in desc_ids
        assert child2.agent_id in desc_ids
        assert grandchild.agent_id in desc_ids

    @pytest.mark.asyncio
    async def test_terminate_agent(self, registry):
        """Should terminate agent."""
        lineage = await registry.register()
        terminated = await registry.terminate(lineage.agent_id)

        assert lineage.agent_id in terminated

        updated = await registry.get_lineage(lineage.agent_id)
        assert not updated.is_active

    @pytest.mark.asyncio
    async def test_terminate_cascades(self, registry):
        """Should cascade termination to descendants."""
        root = await registry.register()
        child = await registry.register(parent_id=root.agent_id)
        grandchild = await registry.register(parent_id=child.agent_id)

        terminated = await registry.terminate(root.agent_id, cascade=True)

        assert len(terminated) == 3
        assert root.agent_id in terminated
        assert child.agent_id in terminated
        assert grandchild.agent_id in terminated

    @pytest.mark.asyncio
    async def test_terminate_no_cascade(self, registry):
        """Should orphan children when not cascading."""
        root = await registry.register()
        child = await registry.register(parent_id=root.agent_id)

        await registry.terminate(root.agent_id, cascade=False)

        updated_child = await registry.get_lineage(child.agent_id)
        assert updated_child.parent_id is None
        assert updated_child.depth == 0
        assert updated_child.is_active

    @pytest.mark.asyncio
    async def test_list_agents(self, registry):
        """Should list all active agents."""
        _agent1 = await registry.register()
        agent2 = await registry.register()
        await registry.terminate(agent2.agent_id)

        active_agents = registry.list_agents(include_terminated=False)
        all_agents = registry.list_agents(include_terminated=True)

        assert len(active_agents) == 1
        assert len(all_agents) == 2

    @pytest.mark.asyncio
    async def test_list_root_agents(self, registry):
        """Should list only root agents."""
        root1 = await registry.register()
        _root2 = await registry.register()
        await registry.register(parent_id=root1.agent_id)

        roots = registry.list_root_agents()

        assert len(roots) == 2


class TestCapabilityBudget:
    """Tests for CapabilityBudget."""

    def test_create_budget(self):
        """Should create capability budget."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write", "delete"]))

        assert budget.total == frozenset(["read", "write", "delete"])
        assert budget.available == frozenset(["read", "write", "delete"])
        assert budget.allocated == {}

    def test_can_allocate(self):
        """Should check if capabilities can be allocated."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write"]))

        assert budget.can_allocate(frozenset(["read"]))
        assert budget.can_allocate(frozenset(["read", "write"]))
        assert not budget.can_allocate(frozenset(["delete"]))

    def test_allocate_capabilities(self):
        """Should allocate capabilities to child."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write"]))
        child_id = uuid4()

        allocated = budget.allocate(child_id, frozenset(["read"]))

        assert allocated == frozenset(["read"])
        assert budget.allocated[child_id] == frozenset(["read"])

    def test_allocate_insufficient_raises(self):
        """Should raise when allocating unavailable capabilities."""
        from crows_nest.agents import CapabilityBudget, InsufficientCapabilitiesError

        budget = CapabilityBudget(total=frozenset(["read"]))

        with pytest.raises(InsufficientCapabilitiesError):
            budget.allocate(uuid4(), frozenset(["write"]))

    def test_deallocate_capabilities(self):
        """Should deallocate capabilities."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write"]))
        child_id = uuid4()
        budget.allocate(child_id, frozenset(["read"]))

        deallocated = budget.deallocate(child_id)

        assert deallocated == frozenset(["read"])
        assert child_id not in budget.allocated

    def test_reserve_capabilities(self):
        """Should reserve capabilities from allocation."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write", "admin"]))
        budget.reserve(frozenset(["admin"]))

        assert budget.available == frozenset(["read", "write"])
        assert not budget.can_allocate(frozenset(["admin"]))

    def test_create_child_budget(self):
        """Should create budget for child."""
        from crows_nest.agents import CapabilityBudget

        budget = CapabilityBudget(total=frozenset(["read", "write"]))
        child_id = uuid4()

        child_budget = budget.create_child_budget(child_id, frozenset(["read"]))

        assert child_budget.total == frozenset(["read"])
        assert budget.allocated[child_id] == frozenset(["read"])


class TestResourceQuota:
    """Tests for ResourceQuota."""

    def test_create_quota(self):
        """Should create resource quota."""
        from crows_nest.agents import ResourceQuota

        quota = ResourceQuota(max_children=5, max_depth=3)

        assert quota.max_children == 5
        assert quota.max_depth == 3
        assert quota.children_spawned == 0

    def test_can_spawn(self):
        """Should check if spawn is allowed."""
        from crows_nest.agents import ResourceQuota

        quota = ResourceQuota(max_children=2)

        assert quota.can_spawn()

        quota.record_spawn()
        assert quota.can_spawn()

        quota.record_spawn()
        assert not quota.can_spawn()

    def test_check_spawn_raises(self):
        """Should raise when spawn not allowed."""
        from crows_nest.agents import QuotaExceededError, ResourceQuota

        quota = ResourceQuota(max_children=1, children_spawned=1)

        with pytest.raises(QuotaExceededError):
            quota.check_spawn()

    def test_token_usage(self):
        """Should track token usage."""
        from crows_nest.agents import ResourceQuota

        quota = ResourceQuota(max_total_tokens=1000)

        assert quota.can_use_tokens(500)
        quota.use_tokens(500)
        assert quota.total_tokens_used == 500

        assert quota.can_use_tokens(500)
        assert not quota.can_use_tokens(501)

    def test_create_child_quota(self):
        """Should create constrained quota for child."""
        from crows_nest.agents import ResourceQuota

        quota = ResourceQuota(max_children=10, max_depth=5, max_tokens_per_child=5000)

        child_quota = quota.create_child_quota()

        assert child_quota.max_children == 10
        assert child_quota.max_depth == 4  # Decremented
        assert child_quota.max_total_tokens == 5000


class TestAgentResources:
    """Tests for AgentResources."""

    def test_create_resources(self):
        """Should create agent resources."""
        from crows_nest.agents import AgentResources, CapabilityBudget

        resources = AgentResources(
            agent_id=uuid4(),
            capabilities=CapabilityBudget(total=frozenset(["read", "write"])),
        )

        assert resources.capabilities.total == frozenset(["read", "write"])

    def test_can_spawn_with(self):
        """Should check if spawn is possible."""
        from crows_nest.agents import AgentResources, CapabilityBudget

        resources = AgentResources(
            agent_id=uuid4(),
            capabilities=CapabilityBudget(total=frozenset(["read", "write"])),
        )

        assert resources.can_spawn_with(frozenset(["read"]))
        assert not resources.can_spawn_with(frozenset(["delete"]))

    def test_spawn_child(self):
        """Should spawn child with resources."""
        from crows_nest.agents import AgentResources, CapabilityBudget

        parent = AgentResources(
            agent_id=uuid4(),
            capabilities=CapabilityBudget(total=frozenset(["read", "write"])),
        )
        child_id = uuid4()

        child_resources = parent.spawn_child(child_id, frozenset(["read"]))

        assert child_resources.agent_id == child_id
        assert child_resources.capabilities.total == frozenset(["read"])
        assert parent.quota.children_spawned == 1

    def test_terminate_child(self):
        """Should clean up child resources."""
        from crows_nest.agents import AgentResources, CapabilityBudget

        parent = AgentResources(
            agent_id=uuid4(),
            capabilities=CapabilityBudget(total=frozenset(["read", "write"])),
        )
        child_id = uuid4()
        parent.spawn_child(child_id, frozenset(["read"]))

        parent.terminate_child(child_id)

        assert child_id not in parent.capabilities.allocated
        assert parent.quota.children_spawned == 0


class TestHiringCommittee:
    """Tests for HiringCommittee."""

    @pytest.fixture
    def basic_request(self):
        """Create a basic spawn request."""
        from crows_nest.agents import SpawnRequest

        return SpawnRequest(
            parent_id=uuid4(),
            requested_capabilities=frozenset(["read", "write"]),
            justification="Need to process data",
        )

    @pytest.fixture
    def evaluators(self):
        """Create test evaluators."""
        from crows_nest.agents import EvaluatorSpec

        return [
            EvaluatorSpec(evaluator_id=uuid4(), name="Security", criteria="Security check"),
            EvaluatorSpec(evaluator_id=uuid4(), name="Resource", criteria="Resource check"),
            EvaluatorSpec(evaluator_id=uuid4(), name="Necessity", criteria="Necessity check"),
        ]

    def test_create_committee(self, evaluators):
        """Should create committee."""
        from crows_nest.agents import HiringCommittee

        committee = HiringCommittee(evaluators=evaluators, consensus_threshold=0.6)

        assert len(committee.evaluators) == 3
        assert committee.consensus_threshold == 0.6

    def test_quorum_size(self, evaluators):
        """Should calculate quorum size."""
        from crows_nest.agents import HiringCommittee

        committee = HiringCommittee(evaluators=evaluators, quorum_fraction=0.5)

        assert committee.quorum_size == 1  # int(50% of 3) = int(1.5) = 1

    @pytest.mark.asyncio
    async def test_evaluate_approved(self, evaluators, basic_request):
        """Should approve with sufficient votes."""
        from crows_nest.agents import (
            EvaluatorVote,
            HiringCommittee,
            SpawnDecisionStatus,
            SpawnVote,
        )

        committee = HiringCommittee(evaluators=evaluators, consensus_threshold=0.6)
        votes = [
            EvaluatorVote(evaluator_id=evaluators[0].evaluator_id, vote=SpawnVote.APPROVE),
            EvaluatorVote(evaluator_id=evaluators[1].evaluator_id, vote=SpawnVote.APPROVE),
            EvaluatorVote(evaluator_id=evaluators[2].evaluator_id, vote=SpawnVote.REJECT),
        ]

        decision = await committee.evaluate(basic_request, votes=votes)

        assert decision.status == SpawnDecisionStatus.APPROVED
        assert decision.is_approved
        assert decision.granted_capabilities == basic_request.requested_capabilities

    @pytest.mark.asyncio
    async def test_evaluate_rejected(self, evaluators, basic_request):
        """Should reject with insufficient votes."""
        from crows_nest.agents import (
            EvaluatorVote,
            HiringCommittee,
            SpawnDecisionStatus,
            SpawnVote,
        )

        committee = HiringCommittee(evaluators=evaluators, consensus_threshold=0.6)
        votes = [
            EvaluatorVote(evaluator_id=evaluators[0].evaluator_id, vote=SpawnVote.APPROVE),
            EvaluatorVote(evaluator_id=evaluators[1].evaluator_id, vote=SpawnVote.REJECT),
            EvaluatorVote(evaluator_id=evaluators[2].evaluator_id, vote=SpawnVote.REJECT),
        ]

        decision = await committee.evaluate(basic_request, votes=votes)

        assert decision.status == SpawnDecisionStatus.REJECTED
        assert not decision.is_approved
        assert decision.granted_capabilities is None

    @pytest.mark.asyncio
    async def test_evaluate_pending_no_quorum(self, evaluators, basic_request):
        """Should stay pending without quorum."""
        from crows_nest.agents import (
            EvaluatorVote,
            HiringCommittee,
            SpawnDecisionStatus,
            SpawnVote,
        )

        committee = HiringCommittee(
            evaluators=evaluators,
            require_quorum=True,
            quorum_fraction=0.5,
        )
        votes = [
            EvaluatorVote(evaluator_id=evaluators[0].evaluator_id, vote=SpawnVote.ABSTAIN),
        ]

        decision = await committee.evaluate(basic_request, votes=votes)

        assert decision.status == SpawnDecisionStatus.PENDING

    @pytest.mark.asyncio
    async def test_evaluate_auto_approve_no_evaluators(self, basic_request):
        """Should auto-approve when no evaluators configured."""
        from crows_nest.agents import HiringCommittee, SpawnDecisionStatus

        committee = HiringCommittee(evaluators=[])

        decision = await committee.evaluate_with_auto_approve(basic_request)

        assert decision.status == SpawnDecisionStatus.APPROVED

    def test_approval_rate_excludes_abstentions(self, evaluators, basic_request):
        """Should calculate approval rate excluding abstentions."""
        from crows_nest.agents import EvaluatorVote, SpawnDecision, SpawnVote

        decision = SpawnDecision(request=basic_request)
        decision.add_vote(
            EvaluatorVote(evaluator_id=evaluators[0].evaluator_id, vote=SpawnVote.APPROVE)
        )
        decision.add_vote(
            EvaluatorVote(evaluator_id=evaluators[1].evaluator_id, vote=SpawnVote.ABSTAIN)
        )
        decision.add_vote(
            EvaluatorVote(evaluator_id=evaluators[2].evaluator_id, vote=SpawnVote.REJECT)
        )

        # 1 approve, 1 reject = 50%
        assert decision.approval_rate == 0.5


class TestSpawningOperations:
    """Tests for spawning thunk operations."""

    @pytest.fixture
    async def cleanup(self):
        """Clean up global state after tests."""
        yield
        from crows_nest.agents import reset_agent_registry, reset_hiring_committee
        from crows_nest.agents.spawning import reset_agent_resources

        await reset_agent_registry()
        reset_hiring_committee()
        reset_agent_resources()

    @pytest.mark.asyncio
    async def test_register_root_agent(self, cleanup):
        """Should register root agent with resources."""
        from crows_nest.agents.spawning import register_root_agent

        result = await register_root_agent(
            capabilities=["read", "write", "agent.spawn"],
            max_children=5,
            max_depth=3,
        )

        assert "agent_id" in result
        assert result["is_root"]
        assert "read" in result["capabilities"]
        assert result["quota"]["max_children"] == 5

    @pytest.mark.asyncio
    async def test_spawn_agent(self, cleanup):
        """Should spawn child agent."""
        from crows_nest.agents.spawning import register_root_agent, spawn_agent

        # First register a root
        root = await register_root_agent(
            capabilities=["read", "write", "agent.spawn", "agent.terminate"],
        )

        # Then spawn a child
        child = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read"],
        )

        assert "child_id" in child
        assert child["parent_id"] == root["agent_id"]
        assert child["capabilities"] == ["read"]
        assert child["depth"] == 1

    @pytest.mark.asyncio
    async def test_spawn_agent_capability_restriction(self, cleanup):
        """Should reject spawning with capabilities parent doesn't have."""
        from crows_nest.agents import InsufficientCapabilitiesError
        from crows_nest.agents.spawning import register_root_agent, spawn_agent

        root = await register_root_agent(
            capabilities=["read", "agent.spawn"],
        )

        with pytest.raises(InsufficientCapabilitiesError):
            await spawn_agent(
                parent_id=root["agent_id"],
                capabilities=["write"],  # Parent doesn't have this
            )

    @pytest.mark.asyncio
    async def test_terminate_agent(self, cleanup):
        """Should terminate agent."""
        from crows_nest.agents.spawning import (
            register_root_agent,
            spawn_agent,
            terminate_agent,
        )

        root = await register_root_agent(
            capabilities=["read", "agent.spawn", "agent.terminate"],
        )
        child = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read"],
        )

        result = await terminate_agent(agent_id=child["child_id"])

        assert result["agent_id"] == child["child_id"]
        assert child["child_id"] in result["terminated"]

    @pytest.mark.asyncio
    async def test_terminate_cascade(self, cleanup):
        """Should cascade termination."""
        from crows_nest.agents.spawning import (
            register_root_agent,
            spawn_agent,
            terminate_agent,
        )

        root = await register_root_agent(
            capabilities=["read", "agent.spawn", "agent.terminate"],
        )
        child = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read", "agent.spawn"],
        )
        grandchild = await spawn_agent(
            parent_id=child["child_id"],
            capabilities=["read"],
        )

        result = await terminate_agent(agent_id=child["child_id"], cascade=True)

        assert result["count"] == 2
        assert child["child_id"] in result["terminated"]
        assert grandchild["child_id"] in result["terminated"]

    @pytest.mark.asyncio
    async def test_list_children(self, cleanup):
        """Should list agent children."""
        from crows_nest.agents.spawning import (
            list_children,
            register_root_agent,
            spawn_agent,
        )

        root = await register_root_agent(
            capabilities=["read", "agent.spawn", "agent.read"],
        )
        child1 = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read"],
        )
        child2 = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read"],
        )

        result = await list_children(agent_id=root["agent_id"])

        assert result["count"] == 2
        child_ids = [c["child_id"] for c in result["children"]]
        assert child1["child_id"] in child_ids
        assert child2["child_id"] in child_ids

    @pytest.mark.asyncio
    async def test_get_lineage(self, cleanup):
        """Should get agent lineage."""
        from crows_nest.agents.spawning import (
            get_lineage,
            register_root_agent,
            spawn_agent,
        )

        root = await register_root_agent(
            capabilities=["read", "agent.spawn", "agent.read"],
        )
        child = await spawn_agent(
            parent_id=root["agent_id"],
            capabilities=["read"],
        )

        result = await get_lineage(agent_id=child["child_id"])

        assert result["parent_id"] == root["agent_id"]
        assert result["depth"] == 1
        assert len(result["ancestors"]) == 1

    @pytest.mark.asyncio
    async def test_get_resources(self, cleanup):
        """Should get agent resources."""
        from crows_nest.agents.spawning import get_resources, register_root_agent

        root = await register_root_agent(
            capabilities=["read", "write", "agent.read"],
        )

        result = await get_resources(agent_id=root["agent_id"])

        assert "read" in result["capabilities"]["total"]
        assert "write" in result["capabilities"]["total"]
        assert result["quota"]["max_children"] == 10
