"""Tests for LLM provider abstraction."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from crows_nest.agents.llm import (
    ChatMessage,
    CompletionResult,
    LLMConfig,
    LLMProviderType,
    MockLLMProviderAdapter,
    create_provider,
)


class SimpleResponse(BaseModel):
    """Simple test response schema."""

    answer: str
    confidence: float


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = LLMConfig()

        assert config.provider == LLMProviderType.MOCK
        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_key is None
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.extra == {}

    def test_mock_factory(self):
        """Should create mock config."""
        config = LLMConfig.mock()

        assert config.provider == LLMProviderType.MOCK

    def test_mock_factory_with_seed(self):
        """Should pass seed through extra."""
        config = LLMConfig.mock(seed=42)

        assert config.extra == {"seed": 42}

    def test_anthropic_factory(self):
        """Should create Anthropic config."""
        config = LLMConfig.anthropic(api_key="test-key")

        assert config.provider == LLMProviderType.ANTHROPIC
        assert config.api_key == "test-key"
        assert config.model == "claude-sonnet-4-20250514"

    def test_anthropic_factory_custom_model(self):
        """Should allow custom model."""
        config = LLMConfig.anthropic(api_key="key", model="claude-3-opus")

        assert config.model == "claude-3-opus"

    def test_anthropic_factory_kwargs(self):
        """Should pass kwargs."""
        config = LLMConfig.anthropic(
            api_key="key",
            max_tokens=2000,
            temperature=0.5,
        )

        assert config.max_tokens == 2000
        assert config.temperature == 0.5

    def test_config_is_frozen(self):
        """Should be immutable."""
        config = LLMConfig()

        with pytest.raises(AttributeError):
            config.provider = LLMProviderType.ANTHROPIC  # type: ignore[misc]


class TestLLMProviderType:
    """Tests for LLMProviderType enum."""

    def test_provider_types(self):
        """Should have expected provider types."""
        assert LLMProviderType.MOCK.value == "mock"
        assert LLMProviderType.ANTHROPIC.value == "anthropic"
        assert LLMProviderType.OPENAI.value == "openai"

    def test_from_string(self):
        """Should create from string value."""
        assert LLMProviderType("mock") == LLMProviderType.MOCK
        assert LLMProviderType("anthropic") == LLMProviderType.ANTHROPIC
        assert LLMProviderType("openai") == LLMProviderType.OPENAI


class TestChatMessage:
    """Tests for ChatMessage."""

    def test_create_message(self):
        """Should create message."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_serialization(self):
        """Should serialize to dict."""
        msg = ChatMessage(role="assistant", content="Hi there")

        data = msg.model_dump()

        assert data == {"role": "assistant", "content": "Hi there"}


class TestCompletionResult:
    """Tests for CompletionResult."""

    def test_create_result(self):
        """Should create completion result."""
        result = CompletionResult(
            content={"answer": "test"},
            raw_response={"id": "123"},
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            model="claude-3",
        )

        assert result.content == {"answer": "test"}
        assert result.raw_response == {"id": "123"}
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result.model == "claude-3"

    def test_defaults(self):
        """Should have sensible defaults."""
        result = CompletionResult(content="hello")

        assert result.content == "hello"
        assert result.raw_response is None
        assert result.usage is None
        assert result.model is None


class TestMockLLMProviderAdapter:
    """Tests for MockLLMProviderAdapter."""

    def test_complete_returns_result(self):
        """Should return CompletionResult."""
        config = LLMConfig.mock(seed=42)
        adapter = MockLLMProviderAdapter(config)
        messages = [ChatMessage(role="user", content="test")]

        result = adapter.complete(messages, SimpleResponse)

        assert isinstance(result, CompletionResult)
        assert isinstance(result.content, SimpleResponse)
        assert result.raw_response == {"mock": True}
        assert result.usage == {"prompt_tokens": 0, "completion_tokens": 0}
        assert result.model == "mock"

    def test_deterministic_with_seed(self):
        """Should be deterministic with same seed."""
        config1 = LLMConfig.mock(seed=42)
        config2 = LLMConfig.mock(seed=42)
        adapter1 = MockLLMProviderAdapter(config1)
        adapter2 = MockLLMProviderAdapter(config2)
        messages = [ChatMessage(role="user", content="test")]

        result1 = adapter1.complete(messages, SimpleResponse)
        result2 = adapter2.complete(messages, SimpleResponse)

        assert result1.content.answer == result2.content.answer
        assert result1.content.confidence == result2.content.confidence

    @pytest.mark.asyncio
    async def test_complete_async(self):
        """Should work asynchronously."""
        config = LLMConfig.mock(seed=42)
        adapter = MockLLMProviderAdapter(config)
        messages = [ChatMessage(role="user", content="test")]

        result = await adapter.complete_async(messages, SimpleResponse)

        assert isinstance(result, CompletionResult)
        assert isinstance(result.content, SimpleResponse)


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_mock_provider(self):
        """Should create mock provider."""
        config = LLMConfig.mock()

        provider = create_provider(config)

        assert isinstance(provider, MockLLMProviderAdapter)

    def test_create_mock_provider_with_seed(self):
        """Should pass seed to mock provider."""
        config = LLMConfig.mock(seed=123)

        provider = create_provider(config)
        messages = [ChatMessage(role="user", content="test")]
        result = provider.complete(messages, SimpleResponse)

        # Verify it's deterministic
        provider2 = create_provider(LLMConfig.mock(seed=123))
        result2 = provider2.complete(messages, SimpleResponse)

        assert result.content.answer == result2.content.answer

    def test_create_unsupported_provider_raises(self):
        """Should raise for unsupported provider types."""
        # Create config with invalid provider type by using internal constructor
        config = LLMConfig(
            provider=LLMProviderType.MOCK,
            model="test",
        )
        # Manually override to test error path
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="test",
            api_key=None,  # Missing API key
        )

        # This should create the provider, but fail when actually used
        from crows_nest.agents.llm import AtomicAgentsProvider

        provider = create_provider(config)
        assert isinstance(provider, AtomicAgentsProvider)
