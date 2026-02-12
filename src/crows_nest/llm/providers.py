"""Provider configurations for LLM clients.

Thin wrappers around instructor's provider factories, with
sensible defaults and environment-based configuration.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import instructor


@dataclass
class ProviderConfig(ABC):
    """Base configuration for LLM providers."""

    model: str
    """Default model to use."""

    api_key: str | None = None
    """API key. Falls back to environment variable if not provided."""

    base_url: str | None = None
    """Optional base URL override."""

    extra_params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters passed to the client."""

    @abstractmethod
    def create_client(self) -> instructor.Instructor:
        """Create an instructor client for this provider."""

    @abstractmethod
    def create_async_client(self) -> instructor.Instructor:
        """Create an async instructor client for this provider."""

    @property
    @abstractmethod
    def env_key_name(self) -> str:
        """Environment variable name for API key."""

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        key = os.environ.get(self.env_key_name)
        if not key:
            raise ValueError(f"No API key provided and {self.env_key_name} not set in environment")
        return key


@dataclass
class AnthropicConfig(ProviderConfig):
    """Configuration for Anthropic's Claude models."""

    model: str = "claude-sonnet-4-20250514"

    @property
    def env_key_name(self) -> str:
        return "ANTHROPIC_API_KEY"

    def create_client(self) -> instructor.Instructor:
        import anthropic
        import instructor

        client = anthropic.Anthropic(
            api_key=self.get_api_key(),
            base_url=self.base_url,
            **self.extra_params,
        )
        return instructor.from_anthropic(client)

    def create_async_client(self) -> instructor.Instructor:
        import anthropic
        import instructor

        client = anthropic.AsyncAnthropic(
            api_key=self.get_api_key(),
            base_url=self.base_url,
            **self.extra_params,
        )
        return instructor.from_anthropic(client)


@dataclass
class OpenAIConfig(ProviderConfig):
    """Configuration for OpenAI models."""

    model: str = "gpt-4o"

    @property
    def env_key_name(self) -> str:
        return "OPENAI_API_KEY"

    def create_client(self) -> instructor.Instructor:
        import instructor
        import openai

        client = openai.OpenAI(
            api_key=self.get_api_key(),
            base_url=self.base_url,
            **self.extra_params,
        )
        return instructor.from_openai(client)

    def create_async_client(self) -> instructor.Instructor:
        import instructor
        import openai

        client = openai.AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.base_url,
            **self.extra_params,
        )
        return instructor.from_openai(client)


@dataclass
class LiteLLMConfig(ProviderConfig):
    """Configuration for LiteLLM (supports Ollama, many others).

    LiteLLM provides a unified interface to many providers.
    For local models via Ollama, use model="ollama/llama3.2".
    """

    model: str = "ollama/llama3.2"
    api_key: str | None = "dummy"  # LiteLLM requires a key even for local

    @property
    def env_key_name(self) -> str:
        return "LITELLM_API_KEY"

    def create_client(self) -> instructor.Instructor:
        import instructor
        import litellm  # type: ignore[import-not-found]

        # LiteLLM uses a different pattern - we patch it
        return instructor.from_litellm(litellm.completion)

    def create_async_client(self) -> instructor.Instructor:
        import instructor
        import litellm

        # LiteLLM async uses acompletion
        return instructor.from_litellm(litellm.acompletion)


@dataclass
class MockConfig(ProviderConfig):
    """Configuration for mock/testing provider.

    Useful for deterministic testing and exploration without API calls.
    """

    model: str = "mock"
    api_key: str | None = "mock"

    responses: list[str] = field(default_factory=list)
    """Canned responses to return in order. Cycles when exhausted."""

    echo: bool = False
    """If True, echo back the user's input with reflection."""

    @property
    def env_key_name(self) -> str:
        return "MOCK_API_KEY"

    def create_client(self) -> instructor.Instructor:
        # Mock doesn't use a real instructor client
        raise NotImplementedError(
            "MockConfig doesn't create a real instructor client. "
            "Use MockAgent directly for testing."
        )

    def create_async_client(self) -> instructor.Instructor:
        # Mock doesn't use a real instructor client
        raise NotImplementedError(
            "MockConfig doesn't create a real instructor client. "
            "Use MockAgent directly for testing."
        )
