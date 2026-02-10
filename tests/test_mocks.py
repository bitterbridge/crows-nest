"""Tests for mock LLM provider."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from crows_nest.mocks import MockChatProvider, MockLLMProvider


class SimpleOutput(BaseModel):
    """Simple test output schema."""

    message: str
    count: int


class OutputWithOptional(BaseModel):
    """Output schema with optional field."""

    required_field: str
    optional_field: str | None = None


class OutputWithEnum(BaseModel):
    """Output schema with enum constraint."""

    status: str = Field(json_schema_extra={"enum": ["active", "inactive", "pending"]})


class OutputWithNested(BaseModel):
    """Output schema with nested structure."""

    class Inner(BaseModel):
        value: int
        name: str

    outer_name: str
    inner: Inner


class OutputWithArray(BaseModel):
    """Output schema with array field."""

    items: list[str]
    counts: list[int]


class OutputWithConstraints(BaseModel):
    """Output schema with value constraints."""

    score: float = Field(ge=0.0, le=1.0)
    age: int = Field(ge=0, le=120)


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_generate_simple_schema(self):
        """Should generate valid output for simple schema."""
        provider = MockLLMProvider(seed=42)

        result = provider.generate(SimpleOutput)

        assert isinstance(result, SimpleOutput)
        assert isinstance(result.message, str)
        assert isinstance(result.count, int)

    def test_generate_deterministic_with_seed(self):
        """Should generate same output with same seed."""
        provider1 = MockLLMProvider(seed=123)
        provider2 = MockLLMProvider(seed=123)

        result1 = provider1.generate(SimpleOutput)
        result2 = provider2.generate(SimpleOutput)

        assert result1.message == result2.message
        assert result1.count == result2.count

    def test_generate_different_with_different_seed(self):
        """Should generate different output with different seeds."""
        provider1 = MockLLMProvider(seed=1)
        provider2 = MockLLMProvider(seed=2)

        result1 = provider1.generate(SimpleOutput)
        result2 = provider2.generate(SimpleOutput)

        # Very unlikely to be the same
        assert result1.message != result2.message or result1.count != result2.count

    def test_generate_with_optional_field(self):
        """Should handle optional fields."""
        provider = MockLLMProvider(seed=42)

        result = provider.generate(OutputWithOptional)

        assert isinstance(result, OutputWithOptional)
        assert isinstance(result.required_field, str)
        # Optional field may or may not be present

    def test_generate_with_enum_constraint(self):
        """Should generate valid enum values."""
        provider = MockLLMProvider(seed=42)

        result = provider.generate(OutputWithEnum)

        assert isinstance(result, OutputWithEnum)
        assert result.status in ["active", "inactive", "pending"]

    def test_generate_nested_schema(self):
        """Should handle nested schemas."""
        provider = MockLLMProvider(seed=42)

        result = provider.generate(OutputWithNested)

        assert isinstance(result, OutputWithNested)
        assert isinstance(result.outer_name, str)
        assert isinstance(result.inner.value, int)
        assert isinstance(result.inner.name, str)

    def test_generate_array_fields(self):
        """Should handle array fields."""
        provider = MockLLMProvider(seed=42)

        result = provider.generate(OutputWithArray)

        assert isinstance(result, OutputWithArray)
        assert isinstance(result.items, list)
        assert isinstance(result.counts, list)
        assert all(isinstance(i, str) for i in result.items)
        assert all(isinstance(c, int) for c in result.counts)

    def test_generate_respects_max_array_length(self):
        """Should respect max array length setting."""
        provider = MockLLMProvider(seed=42, max_array_length=3)

        result = provider.generate(OutputWithArray)

        assert len(result.items) <= 3
        assert len(result.counts) <= 3

    def test_generate_respects_max_string_length(self):
        """Should respect max string length setting."""
        provider = MockLLMProvider(seed=42, max_string_length=20)

        result = provider.generate(SimpleOutput)

        assert len(result.message) <= 20

    def test_generate_raw_returns_dict(self):
        """Should return raw dict for JSON schema."""
        provider = MockLLMProvider(seed=42)
        schema = SimpleOutput.model_json_schema()

        result = provider.generate_raw(schema)

        assert isinstance(result, dict)
        assert "message" in result or "count" in result

    def test_call_count_increments(self):
        """Should track call count."""
        provider = MockLLMProvider(seed=42)

        assert provider.call_count == 0

        provider.generate(SimpleOutput)
        assert provider.call_count == 1

        provider.generate(SimpleOutput)
        assert provider.call_count == 2

    def test_reset_call_count(self):
        """Should reset call count."""
        provider = MockLLMProvider(seed=42)
        provider.generate(SimpleOutput)
        provider.generate(SimpleOutput)

        provider.reset_call_count()

        assert provider.call_count == 0

    def test_handles_format_email(self):
        """Should generate valid email format."""

        class EmailOutput(BaseModel):
            email: str = Field(json_schema_extra={"format": "email"})

        provider = MockLLMProvider(seed=42)
        result = provider.generate(EmailOutput)

        assert "@" in result.email
        assert ".com" in result.email

    def test_handles_format_uri(self):
        """Should generate valid URI format."""

        class UriOutput(BaseModel):
            url: str = Field(json_schema_extra={"format": "uri"})

        provider = MockLLMProvider(seed=42)
        result = provider.generate(UriOutput)

        assert result.url.startswith("https://")

    def test_handles_const_value(self):
        """Should return const value."""

        class ConstOutput(BaseModel):
            version: str = Field(json_schema_extra={"const": "1.0.0"})

        provider = MockLLMProvider(seed=42)
        result = provider.generate(ConstOutput)

        assert result.version == "1.0.0"


class TestMockChatProvider:
    """Tests for MockChatProvider."""

    def test_complete_returns_schema_instance(self):
        """Should return instance of output schema."""
        provider = MockChatProvider(seed=42)
        messages = [{"role": "user", "content": "Hello"}]

        result = provider.complete(messages, SimpleOutput)

        assert isinstance(result, SimpleOutput)

    def test_complete_ignores_messages(self):
        """Should not fail with various message inputs."""
        provider = MockChatProvider(seed=42)

        result1 = provider.complete([], SimpleOutput)
        result2 = provider.complete(
            [{"role": "user", "content": "Question"}],
            SimpleOutput,
        )
        result3 = provider.complete(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            SimpleOutput,
        )

        assert isinstance(result1, SimpleOutput)
        assert isinstance(result2, SimpleOutput)
        assert isinstance(result3, SimpleOutput)

    def test_complete_with_system_prompt(self):
        """Should accept system prompt parameter."""
        provider = MockChatProvider(seed=42)

        result = provider.complete(
            [{"role": "user", "content": "Hi"}],
            SimpleOutput,
            system_prompt="You are a helpful assistant.",
        )

        assert isinstance(result, SimpleOutput)

    @pytest.mark.asyncio
    async def test_complete_async(self):
        """Should work asynchronously."""
        provider = MockChatProvider(seed=42)
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.complete_async(messages, SimpleOutput)

        assert isinstance(result, SimpleOutput)

    def test_deterministic_completions(self):
        """Should produce deterministic results with seed."""
        provider1 = MockChatProvider(seed=99)
        provider2 = MockChatProvider(seed=99)
        messages = [{"role": "user", "content": "test"}]

        result1 = provider1.complete(messages, SimpleOutput)
        result2 = provider2.complete(messages, SimpleOutput)

        assert result1.message == result2.message
        assert result1.count == result2.count
