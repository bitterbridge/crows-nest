"""Tests for system introspection operations."""

from __future__ import annotations

import pytest

from crows_nest.core.introspection import (
    system_describe_operation,
    system_help,
    system_list_capabilities,
    system_list_operations,
    system_list_plugins,
    system_search_operations,
)
from crows_nest.core.registry import get_global_registry, thunk_operation


@pytest.fixture
def _register_test_operations():
    """Register test operations for introspection testing."""
    registry = get_global_registry()

    # Register a test operation
    @thunk_operation(
        name="test.example",
        description="A test operation for introspection.",
        required_capabilities=frozenset({"test.cap1", "test.cap2"}),
    )
    async def test_example_op(name: str, count: int = 5) -> dict:
        return {"name": name, "count": count}

    yield

    # Cleanup
    registry.unregister("test.example")


class TestSystemHelp:
    """Tests for system.help operation."""

    @pytest.mark.asyncio
    async def test_returns_welcome_message(self):
        """Should return welcome message."""
        result = await system_help()
        assert "message" in result
        assert "Welcome to Crow's Nest" in result["message"]

    @pytest.mark.asyncio
    async def test_includes_introspection_operations(self):
        """Should list all introspection operations."""
        result = await system_help()
        assert "introspection_operations" in result
        ops = result["introspection_operations"]
        assert "system.help" in ops
        assert "system.list_operations" in ops
        assert "system.describe_operation" in ops
        assert "system.list_plugins" in ops
        assert "system.list_capabilities" in ops

    @pytest.mark.asyncio
    async def test_includes_quick_start_guide(self):
        """Should include quick start guide."""
        result = await system_help()
        assert "quick_start" in result
        assert isinstance(result["quick_start"], list)
        assert len(result["quick_start"]) > 0

    @pytest.mark.asyncio
    async def test_includes_tips(self):
        """Should include usage tips."""
        result = await system_help()
        assert "tips" in result
        assert isinstance(result["tips"], list)


class TestSystemListOperations:
    """Tests for system.list_operations operation."""

    @pytest.mark.asyncio
    async def test_returns_total_count(self, _register_test_operations):
        """Should return total operation count."""
        result = await system_list_operations()
        assert "total_operations" in result
        assert result["total_operations"] > 0

    @pytest.mark.asyncio
    async def test_returns_namespaces(self, _register_test_operations):
        """Should return list of namespaces."""
        result = await system_list_operations()
        assert "namespaces" in result
        assert "system" in result["namespaces"]
        assert "test" in result["namespaces"]

    @pytest.mark.asyncio
    async def test_groups_by_namespace(self, _register_test_operations):
        """Should group operations by namespace."""
        result = await system_list_operations()
        assert "operations" in result
        assert "system" in result["operations"]
        assert "test" in result["operations"]

    @pytest.mark.asyncio
    async def test_filter_by_namespace(self, _register_test_operations):
        """Should filter operations by namespace."""
        result = await system_list_operations(namespace="test")
        assert "operations" in result
        assert "test" in result["operations"]
        assert "system" not in result["operations"]

    @pytest.mark.asyncio
    async def test_includes_capabilities(self, _register_test_operations):
        """Should include required capabilities."""
        result = await system_list_operations(namespace="test")
        ops = result["operations"]["test"]
        test_op = next(op for op in ops if op["name"] == "test.example")
        assert "required_capabilities" in test_op
        assert "test.cap1" in test_op["required_capabilities"]
        assert "test.cap2" in test_op["required_capabilities"]

    @pytest.mark.asyncio
    async def test_include_parameters_false(self, _register_test_operations):
        """Should not include parameters by default."""
        result = await system_list_operations(namespace="test")
        ops = result["operations"]["test"]
        test_op = next(op for op in ops if op["name"] == "test.example")
        assert "parameters" not in test_op

    @pytest.mark.asyncio
    async def test_include_parameters_true(self, _register_test_operations):
        """Should include parameters when requested."""
        result = await system_list_operations(namespace="test", include_parameters=True)
        ops = result["operations"]["test"]
        test_op = next(op for op in ops if op["name"] == "test.example")
        assert "parameters" in test_op
        assert "name" in test_op["parameters"]
        assert "count" in test_op["parameters"]


class TestSystemDescribeOperation:
    """Tests for system.describe_operation operation."""

    @pytest.mark.asyncio
    async def test_describe_existing_operation(self, _register_test_operations):
        """Should describe an existing operation."""
        result = await system_describe_operation("test.example")
        assert result["found"] is True
        assert "operation" in result
        op = result["operation"]
        assert op["name"] == "test.example"
        assert op["description"] == "A test operation for introspection."

    @pytest.mark.asyncio
    async def test_describe_nonexistent_operation(self):
        """Should return error for nonexistent operation."""
        result = await system_describe_operation("nonexistent.operation")
        assert result["found"] is False
        assert "error" in result
        assert "suggestion" in result

    @pytest.mark.asyncio
    async def test_includes_parameters(self, _register_test_operations):
        """Should include parameter details."""
        result = await system_describe_operation("test.example")
        params = result["operation"]["parameters"]
        assert "name" in params
        assert params["name"]["required"] is True
        assert "count" in params
        assert params["count"]["required"] is False
        assert params["count"]["default"] == 5

    @pytest.mark.asyncio
    async def test_includes_capabilities(self, _register_test_operations):
        """Should include required capabilities."""
        result = await system_describe_operation("test.example")
        caps = result["operation"]["required_capabilities"]
        assert "test.cap1" in caps
        assert "test.cap2" in caps

    @pytest.mark.asyncio
    async def test_includes_usage_example(self, _register_test_operations):
        """Should include usage example."""
        result = await system_describe_operation("test.example")
        assert "usage_example" in result
        example = result["usage_example"]
        assert example["operation"] == "test.example"
        assert "inputs" in example


class TestSystemListPlugins:
    """Tests for system.list_plugins operation."""

    @pytest.mark.asyncio
    async def test_returns_plugin_counts(self):
        """Should return plugin counts."""
        result = await system_list_plugins()
        assert "total_plugins" in result
        assert "loaded_plugins" in result
        assert "failed_plugins" in result

    @pytest.mark.asyncio
    async def test_returns_plugin_list(self):
        """Should return plugin list."""
        result = await system_list_plugins()
        assert "plugins" in result
        assert isinstance(result["plugins"], list)


class TestSystemListCapabilities:
    """Tests for system.list_capabilities operation."""

    @pytest.mark.asyncio
    async def test_returns_total_count(self, _register_test_operations):
        """Should return total capability count."""
        result = await system_list_capabilities()
        assert "total_capabilities" in result
        assert result["total_capabilities"] > 0

    @pytest.mark.asyncio
    async def test_returns_capability_namespaces(self, _register_test_operations):
        """Should return capability namespaces."""
        result = await system_list_capabilities()
        assert "capability_namespaces" in result
        assert "test" in result["capability_namespaces"]

    @pytest.mark.asyncio
    async def test_maps_capabilities_to_operations(self, _register_test_operations):
        """Should map capabilities to operations."""
        result = await system_list_capabilities()
        caps = result["capabilities"]
        assert "test.cap1" in caps
        assert "test.example" in caps["test.cap1"]

    @pytest.mark.asyncio
    async def test_groups_by_namespace(self, _register_test_operations):
        """Should group capabilities by namespace."""
        result = await system_list_capabilities()
        by_ns = result["by_namespace"]
        assert "test" in by_ns
        assert "test.cap1" in by_ns["test"]


class TestSystemSearchOperations:
    """Tests for system.search_operations operation."""

    @pytest.mark.asyncio
    async def test_search_by_name(self, _register_test_operations):
        """Should find operations by name."""
        result = await system_search_operations("test.example")
        assert result["total_matches"] >= 1
        matches = result["matches"]
        assert any(m["name"] == "test.example" for m in matches)

    @pytest.mark.asyncio
    async def test_search_by_partial_name(self, _register_test_operations):
        """Should find operations by partial name."""
        result = await system_search_operations("example")
        assert result["total_matches"] >= 1
        matches = result["matches"]
        assert any(m["name"] == "test.example" for m in matches)

    @pytest.mark.asyncio
    async def test_search_by_description(self, _register_test_operations):
        """Should find operations by description."""
        result = await system_search_operations("introspection")
        assert result["total_matches"] >= 1
        matches = result["matches"]
        assert any(m["name"] == "test.example" for m in matches)

    @pytest.mark.asyncio
    async def test_search_disabled_descriptions(self, _register_test_operations):
        """Should not search descriptions when disabled."""
        result = await system_search_operations("introspection", search_descriptions=False)
        # Should not find test.example since "introspection" is only in description
        matches = result["matches"]
        assert not any(m["name"] == "test.example" for m in matches)

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, _register_test_operations):
        """Should search case-insensitively."""
        result = await system_search_operations("TEST.EXAMPLE")
        assert result["total_matches"] >= 1
        matches = result["matches"]
        assert any(m["name"] == "test.example" for m in matches)

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        """Should return empty results for no matches."""
        result = await system_search_operations("xyznonexistent123")
        assert result["total_matches"] == 0
        assert len(result["matches"]) == 0

    @pytest.mark.asyncio
    async def test_search_includes_match_type(self, _register_test_operations):
        """Should indicate match type."""
        result = await system_search_operations("test.example")
        matches = result["matches"]
        match = next(m for m in matches if m["name"] == "test.example")
        assert match["match_type"] == "name"


class TestIntrospectionIsIntrospectable:
    """Test that introspection operations are themselves discoverable."""

    @pytest.mark.asyncio
    async def test_system_operations_are_listed(self):
        """All system.* operations should be discoverable."""
        result = await system_list_operations(namespace="system")
        ops = result["operations"]["system"]
        op_names = {op["name"] for op in ops}

        assert "system.help" in op_names
        assert "system.list_operations" in op_names
        assert "system.describe_operation" in op_names
        assert "system.list_plugins" in op_names
        assert "system.list_capabilities" in op_names
        assert "system.search_operations" in op_names

    @pytest.mark.asyncio
    async def test_system_operations_have_no_capabilities(self):
        """System operations should require no capabilities."""
        result = await system_list_operations(namespace="system")
        ops = result["operations"]["system"]

        for op in ops:
            assert op["required_capabilities"] == [], f"{op['name']} should require no capabilities"

    @pytest.mark.asyncio
    async def test_can_describe_system_help(self):
        """Should be able to describe system.help itself."""
        result = await system_describe_operation("system.help")
        assert result["found"] is True
        assert result["operation"]["name"] == "system.help"

    @pytest.mark.asyncio
    async def test_can_search_for_system_operations(self):
        """Should be able to search for system operations."""
        result = await system_search_operations("system")
        assert result["total_matches"] >= 6  # At least our 6 system operations
