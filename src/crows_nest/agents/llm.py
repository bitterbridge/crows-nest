"""LLM provider abstraction and implementations.

Layer 2: Provider protocol and completion interface.
Uses crows_nest.llm.providers (Layer 1) for client creation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crows_nest.llm.providers import ProviderConfig


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


class InstructorProvider:
    """LLM provider using instructor for structured outputs.

    Works with any ProviderConfig from crows_nest.llm.providers.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize with a provider configuration.

        Args:
            config: Provider configuration (AnthropicConfig, OpenAIConfig, etc.)
        """
        self._config = config
        self._client = config.create_client()

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
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> CompletionResult:
        """Generate a completion using instructor."""
        api_messages = self._build_messages(messages, system_prompt)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=api_messages,
            response_model=output_schema,
            max_tokens=max_tokens,
            temperature=temperature,
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
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> CompletionResult:
        """Generate a completion asynchronously."""
        api_messages = self._build_messages(messages, system_prompt)
        async_client = self._config.create_async_client()

        response = await async_client.chat.completions.create(
            model=self._config.model,
            messages=api_messages,
            response_model=output_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return CompletionResult(
            content=response,
            model=self._config.model,
        )


class MockLLMProvider:
    """Mock LLM provider for testing.

    Uses crows_nest.mocks.MockLLMProvider for deterministic output generation.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize mock provider.

        Args:
            seed: Optional seed for deterministic generation.
        """
        from crows_nest.mocks import MockLLMProvider as MockGenerator

        self._generator = MockGenerator(seed=seed)

    def complete(
        self,
        _messages: Sequence[ChatMessage],
        output_schema: type[BaseModel],
        *,
        _system_prompt: str | None = None,
        **_kwargs: Any,
    ) -> CompletionResult:
        """Generate a mock completion."""
        result = self._generator.generate(output_schema)
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
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a mock completion (async)."""
        return self.complete(messages, output_schema, _system_prompt=system_prompt, **kwargs)


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Factory function to create an LLM provider from config.

    Args:
        config: Provider configuration from crows_nest.llm.providers.

    Returns:
        An LLM provider instance.
    """
    from crows_nest.llm.providers import MockConfig

    if isinstance(config, MockConfig):
        return MockLLMProvider()  # type: ignore[return-value]

    return InstructorProvider(config)
