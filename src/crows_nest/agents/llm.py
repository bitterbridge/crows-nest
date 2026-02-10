"""LLM provider abstraction and implementations.

Provides a unified interface for different LLM backends, including
mock providers for testing and real providers via atomic-agents.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence


class LLMProviderType(Enum):
    """Supported LLM provider types."""

    MOCK = "mock"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider.

    Attributes:
        provider: The provider type to use.
        model: Model identifier (e.g., "claude-sonnet-4-20250514").
        api_key: API key (can be None for mock provider).
        max_tokens: Maximum tokens for responses.
        temperature: Temperature for generation.
        extra: Additional provider-specific configuration.
    """

    provider: LLMProviderType = LLMProviderType.MOCK
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.7
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def mock(cls, seed: int | None = None) -> LLMConfig:
        """Create a mock provider config."""
        return cls(
            provider=LLMProviderType.MOCK,
            extra={"seed": seed} if seed is not None else {},
        )

    @classmethod
    def anthropic(
        cls,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ) -> LLMConfig:
        """Create an Anthropic provider config."""
        return cls(
            provider=LLMProviderType.ANTHROPIC,
            model=model,
            api_key=api_key,
            **kwargs,
        )


class ChatMessage(BaseModel):
    """A chat message."""

    role: str = Field(description="Message role: system, user, or assistant")
    content: str = Field(description="Message content")


class CompletionResult(BaseModel):
    """Result of an LLM completion."""

    content: Any = Field(description="The completion content (structured or text)")
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw response from provider"
    )
    usage: dict[str, int] | None = Field(default=None, description="Token usage statistics")
    model: str | None = Field(default=None, description="Model used for completion")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All providers must implement both sync and async completion methods.
    """

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a completion synchronously.

        Args:
            messages: Conversation history.
            output_schema: Pydantic model for structured output.
            system_prompt: Optional system prompt.

        Returns:
            Completion result with structured content.
        """
        ...

    @abstractmethod
    async def complete_async(
        self,
        messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a completion asynchronously.

        Args:
            messages: Conversation history.
            output_schema: Pydantic model for structured output.
            system_prompt: Optional system prompt.

        Returns:
            Completion result with structured content.
        """
        ...


class MockLLMProviderAdapter:
    """Adapter to make MockLLMProvider conform to LLMProvider protocol."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with config."""
        from crows_nest.mocks import MockLLMProvider

        seed = config.extra.get("seed")
        self._provider = MockLLMProvider(seed=seed)
        self._config = config

    def complete(
        self,
        _messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        _system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a mock completion."""
        result = self._provider.generate(output_schema)
        return CompletionResult(
            content=result,
            raw_response={"mock": True},
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            model="mock",
        )

    async def complete_async(
        self,
        messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a mock completion (async)."""
        return self.complete(messages, output_schema, _system_prompt=system_prompt)


class AtomicAgentsProvider:
    """LLM provider using atomic-agents library.

    Wraps atomic-agents to provide structured completions via
    Anthropic, OpenAI, or other supported backends.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with config."""
        self._config = config
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create the instructor client based on provider type."""
        import instructor

        if self._config.provider == LLMProviderType.ANTHROPIC:
            import anthropic

            return instructor.from_anthropic(anthropic.Anthropic(api_key=self._config.api_key))
        if self._config.provider == LLMProviderType.OPENAI:
            import openai

            return instructor.from_openai(openai.OpenAI(api_key=self._config.api_key))

        msg = f"Unsupported provider type: {self._config.provider}"
        raise ValueError(msg)

    def _create_async_client(self) -> Any:
        """Create an async instructor client."""
        import instructor

        if self._config.provider == LLMProviderType.ANTHROPIC:
            import anthropic

            return instructor.from_anthropic(anthropic.AsyncAnthropic(api_key=self._config.api_key))
        if self._config.provider == LLMProviderType.OPENAI:
            import openai

            return instructor.from_openai(openai.AsyncOpenAI(api_key=self._config.api_key))

        msg = f"Unsupported provider type: {self._config.provider}"
        raise ValueError(msg)

    def _build_messages(
        self,
        messages: Sequence[ChatMessage],
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """Build message list for API call."""
        result: list[dict[str, str]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    def complete(
        self,
        messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a completion using atomic-agents/instructor."""
        api_messages = self._build_messages(messages, system_prompt)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=api_messages,
            response_model=output_schema,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        return CompletionResult(
            content=response,
            model=self._config.model,
        )

    async def complete_async(
        self,
        messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        """Generate a completion asynchronously."""
        api_messages = self._build_messages(messages, system_prompt)
        async_client = self._create_async_client()

        response = await async_client.chat.completions.create(
            model=self._config.model,
            messages=api_messages,
            response_model=output_schema,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        return CompletionResult(
            content=response,
            model=self._config.model,
        )


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create an LLM provider from config.

    Args:
        config: LLM configuration.

    Returns:
        An LLM provider instance.

    Raises:
        ValueError: If provider type is not supported.
    """
    if config.provider == LLMProviderType.MOCK:
        return MockLLMProviderAdapter(config)  # type: ignore[return-value]

    if config.provider in (LLMProviderType.ANTHROPIC, LLMProviderType.OPENAI):
        return AtomicAgentsProvider(config)

    msg = f"Unsupported provider type: {config.provider}"
    raise ValueError(msg)
