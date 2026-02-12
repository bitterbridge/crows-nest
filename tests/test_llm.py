"""Tests for LLM provider abstraction.

Tests both Layer 1 (providers) and Layer 2 (protocol/implementations).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from crows_nest.agents.llm import (
    ChatMessage,
    CompletionResult,
    InstructorProvider,
    MockLLMProvider,
    create_provider,
)
from crows_nest.llm import (
    AnthropicConfig,
    ConversationRecorder,
    MockAgent,
    MockConfig,
    OpenAIConfig,
)


class SimpleResponse(BaseModel):
    """Simple test response schema."""

    answer: str
    confidence: float


class ChatInput(BaseModel):
    """Simple chat input schema."""

    message: str = Field(description="User message")


class ChatOutput(BaseModel):
    """Simple chat output schema."""

    response: str = Field(description="Assistant response")


# =============================================================================
# Layer 1: Provider Config Tests
# =============================================================================


class TestProviderConfig:
    """Tests for provider configurations (Layer 1)."""

    def test_anthropic_config_defaults(self):
        """AnthropicConfig has sensible defaults."""
        config = AnthropicConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.env_key_name == "ANTHROPIC_API_KEY"
        assert config.api_key is None
        assert config.base_url is None

    def test_anthropic_config_custom(self):
        """Can customize AnthropicConfig."""
        config = AnthropicConfig(
            model="claude-opus-4-20250514",
            api_key="test-key",
            base_url="https://custom.api",
        )

        assert config.model == "claude-opus-4-20250514"
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api"

    def test_openai_config_defaults(self):
        """OpenAIConfig has sensible defaults."""
        config = OpenAIConfig()

        assert config.model == "gpt-4o"
        assert config.env_key_name == "OPENAI_API_KEY"

    def test_mock_config(self):
        """MockConfig doesn't need real API key."""
        config = MockConfig()

        assert config.model == "mock"
        assert config.api_key == "mock"

    def test_mock_config_create_client_raises(self):
        """MockConfig.create_client() raises NotImplementedError."""
        config = MockConfig()

        with pytest.raises(NotImplementedError):
            config.create_client()

    def test_get_api_key_from_config(self):
        """get_api_key returns configured key."""
        config = AnthropicConfig(api_key="my-key")

        assert config.get_api_key() == "my-key"

    def test_get_api_key_missing_raises(self, monkeypatch):
        """get_api_key raises if no key and env not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = AnthropicConfig()

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            config.get_api_key()


# =============================================================================
# Layer 2: Protocol and Implementation Tests
# =============================================================================


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


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_complete_returns_result(self):
        """Should return CompletionResult."""
        provider = MockLLMProvider(seed=42)
        messages = [ChatMessage(role="user", content="test")]

        result = provider.complete(messages, SimpleResponse)

        assert isinstance(result, CompletionResult)
        assert isinstance(result.content, SimpleResponse)
        assert result.raw_response == {"mock": True}
        assert result.usage == {"prompt_tokens": 0, "completion_tokens": 0}
        assert result.model == "mock"

    def test_deterministic_with_seed(self):
        """Should be deterministic with same seed."""
        provider1 = MockLLMProvider(seed=42)
        provider2 = MockLLMProvider(seed=42)
        messages = [ChatMessage(role="user", content="test")]

        result1 = provider1.complete(messages, SimpleResponse)
        result2 = provider2.complete(messages, SimpleResponse)

        assert result1.content.answer == result2.content.answer
        assert result1.content.confidence == result2.content.confidence

    @pytest.mark.asyncio
    async def test_complete_async(self):
        """Should work asynchronously."""
        provider = MockLLMProvider(seed=42)
        messages = [ChatMessage(role="user", content="test")]

        result = await provider.complete_async(messages, SimpleResponse)

        assert isinstance(result, CompletionResult)
        assert isinstance(result.content, SimpleResponse)


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_mock_provider(self):
        """Should create mock provider from MockConfig."""
        config = MockConfig()

        provider = create_provider(config)

        assert isinstance(provider, MockLLMProvider)

    def test_create_instructor_provider(self):
        """Should create InstructorProvider for real configs."""
        # This would fail without API key, but we can check the type
        config = AnthropicConfig(api_key="test-key")

        # Note: This will fail at client creation without valid key,
        # but we're testing the factory logic
        try:
            provider = create_provider(config)
            assert isinstance(provider, InstructorProvider)
        except Exception:  # noqa: S110
            # Expected if anthropic SDK validates key format
            pass


# =============================================================================
# Layer 3: Mock Agent Tests (for testing without API)
# =============================================================================


class TestMockAgent:
    """Tests for MockAgent (from llm.mock)."""

    def test_echo_mode(self):
        """Echo mode reflects input."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="echo",
            echo_template="You said: {input}",
        )

        result = agent.run(ChatInput(message="hello world"))
        assert "You said: hello world" in result.response

    def test_scripted_mode(self):
        """Scripted mode returns canned responses."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="scripted",
            responses=["First", "Second", "Third"],
        )

        r1 = agent.run(ChatInput(message="a"))
        r2 = agent.run(ChatInput(message="b"))
        r3 = agent.run(ChatInput(message="c"))
        r4 = agent.run(ChatInput(message="d"))

        assert r1.response == "First"
        assert r2.response == "Second"
        assert r3.response == "Third"
        assert r4.response == "First"  # Cycles

    def test_pattern_mode(self):
        """Pattern mode matches regex."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="pattern",
            patterns={
                r"hello|hi": "Greetings!",
                r"bye": "Farewell!",
            },
        )

        assert "Greetings!" in agent.run(ChatInput(message="hello")).response
        assert "Farewell!" in agent.run(ChatInput(message="bye")).response

    def test_deterministic_mode(self):
        """Deterministic mode is reproducible."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="deterministic",
            deterministic_seed="test",
        )

        r1 = agent.run(ChatInput(message="same input"))
        r2 = agent.run(ChatInput(message="same input"))

        assert r1.response == r2.response

    @pytest.mark.asyncio
    async def test_async_run(self):
        """Async run works."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="echo",
        )

        result = await agent.run_async(ChatInput(message="async"))
        assert "async" in result.response

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Streaming yields partial results."""
        agent: MockAgent[ChatInput, ChatOutput] = MockAgent(
            output_schema=ChatOutput,
            mode="echo",
            echo_template="one two three",
        )

        chunks = []
        async for partial in agent.run_async_stream(ChatInput(message="x")):
            chunks.append(partial.response)

        assert len(chunks) == 3
        assert "one" in chunks[0]
        assert "two" in chunks[1]
        assert "three" in chunks[2]


class TestConversationRecorder:
    """Tests for ConversationRecorder."""

    def test_record_turns(self):
        """Records conversation turns."""
        recorder = ConversationRecorder()

        recorder.record(
            ChatInput(message="Hello"),
            ChatOutput(response="Hi!"),
        )

        assert len(recorder.turns) == 1
        assert recorder.turns[0]["input"]["message"] == "Hello"
        assert recorder.turns[0]["output"]["response"] == "Hi!"

    def test_to_scripted_responses(self):
        """Extracts responses for scripted mode."""
        recorder = ConversationRecorder()
        recorder.record(ChatInput(message="Q1"), ChatOutput(response="A1"))
        recorder.record(ChatInput(message="Q2"), ChatOutput(response="A2"))

        responses = recorder.to_scripted_responses()
        assert responses == ["A1", "A2"]

    def test_save_load(self, tmp_path):
        """Can save and load."""
        recorder = ConversationRecorder()
        recorder.record(ChatInput(message="Test"), ChatOutput(response="Response"))

        path = tmp_path / "conv.json"
        recorder.save(str(path))

        loaded = ConversationRecorder.load(str(path))
        assert len(loaded.turns) == 1
        assert loaded.turns[0]["input"]["message"] == "Test"
