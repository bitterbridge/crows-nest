"""Tests for plugin generation operation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from crows_nest.agents.llm import CompletionResult, LLMConfig
from crows_nest.plugins.generation import (
    GeneratedOperation,
    GeneratedParameter,
    GeneratedPlugin,
    configure_llm_provider,
    get_llm_provider,
    plugin_generate,
    reset_llm_provider,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_provider() -> Generator[None, None, None]:
    """Reset LLM provider before and after each test."""
    reset_llm_provider()
    yield
    reset_llm_provider()


class TestLLMProviderConfiguration:
    """Tests for LLM provider configuration."""

    def test_get_provider_without_config_raises(self) -> None:
        """Getting provider without configuration raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not configured"):
            get_llm_provider()

    def test_configure_provider(self) -> None:
        """Configuring provider allows getting it."""
        config = LLMConfig.mock()
        configure_llm_provider(config)
        provider = get_llm_provider()
        assert provider is not None

    def test_reset_provider(self) -> None:
        """Resetting provider clears configuration."""
        config = LLMConfig.mock()
        configure_llm_provider(config)
        reset_llm_provider()
        with pytest.raises(RuntimeError, match="not configured"):
            get_llm_provider()


class TestPluginGenerate:
    """Tests for plugin_generate operation."""

    @pytest.mark.asyncio
    async def test_generate_without_provider_returns_error(self) -> None:
        """Generating without provider configured returns error."""
        result = await plugin_generate(
            spec="Parse JSON data",
            target_operations=["json.parse"],
        )

        assert result["success"] is False
        assert result["error_type"] == "configuration_error"
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_with_mock_provider_returns_error_for_complex_schema(
        self,
    ) -> None:
        """Mock provider can't generate valid nested Pydantic models.

        This is expected - the mock uses Hypothesis which generates random
        data that may not satisfy nested model constraints. Real providers
        with actual LLMs handle this properly.
        """
        # Configure mock provider
        config = LLMConfig.mock(seed=42)
        configure_llm_provider(config)

        result = await plugin_generate(
            spec="Parse JSON data",
            target_operations=["json.parse"],
            agent_id=str(uuid4()),
        )

        # Mock provider generates invalid nested structures
        # This is expected behavior - it's testing that errors are handled
        assert result["success"] is False
        assert "error_type" in result

    @pytest.mark.asyncio
    async def test_generate_with_custom_mock_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generating with custom mock returns expected artifact."""
        # Create a mock provider that returns specific content
        mock_provider = MagicMock()
        mock_generated = GeneratedPlugin(
            name="json_utils",
            description="JSON parsing utilities",
            source_code='''"""JSON utilities."""
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="json.parse",
    description="Parse JSON string",
    required_capabilities=frozenset(),
)
async def json_parse(data: str) -> dict:
    import json
    return {"parsed": json.loads(data)}
''',
            operations=[
                GeneratedOperation(
                    name="json.parse",
                    description="Parse JSON string",
                    parameters=[
                        GeneratedParameter(
                            name="data",
                            type_annotation="str",
                            description="JSON string to parse",
                            required=True,
                        )
                    ],
                    return_type="dict",
                    required_capabilities=[],
                )
            ],
            requested_capabilities=[],
        )

        mock_provider.complete_async = AsyncMock(
            return_value=CompletionResult(
                content=mock_generated,
                raw_response={"mock": True},
            )
        )

        # Patch get_llm_provider to return our mock
        monkeypatch.setattr(
            "crows_nest.plugins.generation.get_llm_provider",
            lambda: mock_provider,
        )

        agent_id = str(uuid4())
        result = await plugin_generate(
            spec="Parse JSON data with custom handling",
            target_operations=["json.parse"],
            example_inputs=[{"data": '{"key": "value"}'}],
            example_outputs=[{"parsed": {"key": "value"}}],
            agent_id=agent_id,
            context={"reason": "Need JSON parsing for API responses"},
        )

        assert result["success"] is True
        artifact = result["artifact"]
        assert artifact["name"] == "json_utils"
        assert "json.parse" in artifact["source_code"]
        assert len(artifact["operations"]) == 1
        assert artifact["operations"][0]["name"] == "json.parse"

        # Check generation context was captured
        assert artifact["generation_context"]["spec"] == "Parse JSON data with custom handling"
        assert artifact["generation_context"]["reason"] == "Need JSON parsing for API responses"

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generation handles LLM errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.complete_async = AsyncMock(side_effect=ValueError("LLM error"))

        monkeypatch.setattr(
            "crows_nest.plugins.generation.get_llm_provider",
            lambda: mock_provider,
        )

        result = await plugin_generate(
            spec="Some spec",
            target_operations=["test.op"],
        )

        assert result["success"] is False
        assert result["error_type"] == "ValueError"
        assert "LLM error" in result["error"]


class TestGeneratedModels:
    """Tests for Pydantic models used in generation."""

    def test_generated_parameter_defaults(self) -> None:
        """GeneratedParameter has sensible defaults."""
        param = GeneratedParameter(
            name="data",
            type_annotation="str",
            description="Input data",
        )
        assert param.required is True
        assert param.default_value is None

    def test_generated_operation_defaults(self) -> None:
        """GeneratedOperation has sensible defaults."""
        op = GeneratedOperation(
            name="test.op",
            description="Test operation",
        )
        assert op.parameters == []
        assert op.return_type == "dict[str, Any]"
        assert op.required_capabilities == []

    def test_generated_plugin_structure(self) -> None:
        """GeneratedPlugin captures full plugin structure."""
        plugin = GeneratedPlugin(
            name="test_plugin",
            description="A test plugin",
            source_code="# code here",
            operations=[
                GeneratedOperation(
                    name="test.greet",
                    description="Greet someone",
                    parameters=[
                        GeneratedParameter(
                            name="name",
                            type_annotation="str",
                            description="Name to greet",
                        )
                    ],
                )
            ],
            requested_capabilities=["user.prompt"],
        )
        assert plugin.name == "test_plugin"
        assert len(plugin.operations) == 1
        assert plugin.operations[0].name == "test.greet"
        assert plugin.requested_capabilities == ["user.prompt"]
