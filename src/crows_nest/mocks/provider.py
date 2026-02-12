"""Mock LLM provider for testing and development.

Generates JSON Schema-conformant responses without calling real LLM APIs.
Uses hypothesis-jsonschema for generating valid structured data.
"""

from __future__ import annotations

import random
import string
from typing import Any

from hypothesis import given, settings
from hypothesis_jsonschema import from_schema
from pydantic import BaseModel


class MockLLMProvider:
    """Mock LLM provider that generates schema-conformant responses.

    This provider is useful for:
    - Testing without burning API tokens
    - Development and debugging
    - CI/CD pipelines
    - Deterministic testing with seeds

    Example:
        provider = MockLLMProvider(seed=42)

        class OutputSchema(BaseModel):
            message: str
            confidence: float

        response = provider.generate(OutputSchema)
        # Returns a valid OutputSchema instance with random data
    """

    def __init__(
        self,
        seed: int | None = None,
        max_string_length: int = 50,
        max_array_length: int = 5,
    ) -> None:
        """Initialize the mock provider.

        Args:
            seed: Random seed for reproducible outputs. If None, random.
            max_string_length: Maximum length for generated strings.
            max_array_length: Maximum length for generated arrays.
        """
        self._seed = seed
        self._max_string_length = max_string_length
        self._max_array_length = max_array_length
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311 - mock data
        self._call_count = 0

    def generate(
        self,
        output_schema: type[BaseModel],
        *,
        _context: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Generate a response conforming to the given Pydantic schema.

        Args:
            output_schema: A Pydantic model class defining the expected output.
            _context: Optional context (ignored by mock, but matches real interface).

        Returns:
            An instance of output_schema with generated data.
        """
        self._call_count += 1

        # Get JSON Schema from Pydantic model
        json_schema = output_schema.model_json_schema()

        # Generate conformant data
        data = self._generate_from_schema(json_schema)

        # Validate and return as Pydantic model
        return output_schema.model_validate(data)

    def generate_raw(
        self,
        json_schema: dict[str, Any],
        *,
        _context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate raw JSON data conforming to a JSON Schema.

        Args:
            json_schema: A JSON Schema dictionary.
            _context: Optional context (ignored by mock).

        Returns:
            A dictionary conforming to the schema.
        """
        self._call_count += 1
        result = self._generate_from_schema(json_schema)
        return dict(result) if isinstance(result, dict) else {"value": result}

    def _generate_from_schema(self, schema: dict[str, Any]) -> Any:
        """Generate data conforming to a JSON Schema.

        Uses a combination of hypothesis-jsonschema for complex schemas
        and custom generation for common patterns.
        """
        schema_type = schema.get("type")

        # Handle common simple types directly for better control
        if schema_type == "string":
            return self._generate_string(schema)
        if schema_type == "integer":
            return self._generate_integer(schema)
        if schema_type == "number":
            return self._generate_number(schema)
        if schema_type == "boolean":
            return self._rng.choice([True, False])
        if schema_type == "null":
            return None
        if schema_type == "array":
            return self._generate_array(schema)
        if schema_type == "object":
            return self._generate_object(schema)

        # Handle anyOf/oneOf/allOf
        if "anyOf" in schema:
            chosen = self._rng.choice(schema["anyOf"])
            return self._generate_from_schema(chosen)
        if "oneOf" in schema:
            chosen = self._rng.choice(schema["oneOf"])
            return self._generate_from_schema(chosen)

        # Handle const/enum
        if "const" in schema:
            return schema["const"]
        if "enum" in schema:
            return self._rng.choice(schema["enum"])

        # Handle $ref - resolve reference
        if "$ref" in schema:
            # For now, skip $ref handling - would need full schema context
            return {}

        # Fallback: use hypothesis for complex schemas
        return self._generate_with_hypothesis(schema)

    def _generate_string(self, schema: dict[str, Any]) -> str:
        """Generate a string value."""
        # Handle enum
        if "enum" in schema:
            return str(self._rng.choice(schema["enum"]))

        # Handle const
        if "const" in schema:
            return str(schema["const"])

        # Handle format
        fmt = schema.get("format")
        if fmt == "email":
            return f"user{self._rng.randint(1, 9999)}@example.com"
        if fmt in ("uri", "url"):
            return f"https://example.com/{self._random_word()}"
        if fmt == "uuid":
            return self._generate_uuid()
        if fmt == "date":
            return f"2024-{self._rng.randint(1, 12):02d}-{self._rng.randint(1, 28):02d}"
        if fmt == "date-time":
            date = f"2024-{self._rng.randint(1, 12):02d}-{self._rng.randint(1, 28):02d}"
            time = f"{self._rng.randint(0, 23):02d}:{self._rng.randint(0, 59):02d}"
            return f"{date}T{time}:{self._rng.randint(0, 59):02d}Z"

        # Handle pattern (simplified)
        if "pattern" in schema:
            # Can't easily generate from regex, use placeholder
            return f"pattern_match_{self._rng.randint(1, 999)}"

        # Generate based on min/max length
        min_len = schema.get("minLength", 1)
        max_len = min(schema.get("maxLength", self._max_string_length), self._max_string_length)

        length = self._rng.randint(min_len, max(min_len, max_len))
        return self._random_text(length)

    def _generate_integer(self, schema: dict[str, Any]) -> int:
        """Generate an integer value."""
        if "enum" in schema:
            return int(self._rng.choice(schema["enum"]))
        if "const" in schema:
            return int(schema["const"])

        minimum = schema.get("minimum", -1000)
        maximum = schema.get("maximum", 1000)

        if "exclusiveMinimum" in schema:
            minimum = schema["exclusiveMinimum"] + 1
        if "exclusiveMaximum" in schema:
            maximum = schema["exclusiveMaximum"] - 1

        return self._rng.randint(int(minimum), int(maximum))

    def _generate_number(self, schema: dict[str, Any]) -> float:
        """Generate a number (float) value."""
        if "enum" in schema:
            return float(self._rng.choice(schema["enum"]))
        if "const" in schema:
            return float(schema["const"])

        minimum = schema.get("minimum", -1000.0)
        maximum = schema.get("maximum", 1000.0)

        return self._rng.uniform(float(minimum), float(maximum))

    def _generate_array(self, schema: dict[str, Any]) -> list[Any]:
        """Generate an array value."""
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        max_items = min(schema.get("maxItems", self._max_array_length), self._max_array_length)

        length = self._rng.randint(min_items, max(min_items, max_items))
        return [self._generate_from_schema(items_schema) for _ in range(length)]

    def _generate_object(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate an object value."""
        result: dict[str, Any] = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Handle $defs for referenced schemas
        defs = schema.get("$defs", {})

        for prop_name, prop_schema in properties.items():
            # Resolve $ref if present
            resolved_schema = prop_schema
            if "$ref" in prop_schema:
                ref = prop_schema["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.split("/")[-1]
                    if def_name in defs:
                        resolved_schema = defs[def_name]

            # Generate required properties always, optional with 70% probability
            if prop_name in required or self._rng.random() < 0.7:
                result[prop_name] = self._generate_from_schema(resolved_schema)

        return result

    def _generate_with_hypothesis(self, schema: dict[str, Any]) -> Any:
        """Use hypothesis-jsonschema for complex schemas."""
        # This is a fallback for schemas we can't handle directly
        # We generate a single example using hypothesis
        result: list[Any] = []

        @given(from_schema(schema))
        @settings(max_examples=1, database=None)
        def generate_one(value: Any) -> None:
            result.append(value)

        try:
            generate_one()
            return result[0] if result else {}
        except Exception:
            # If hypothesis fails, return empty object/default
            return {}

    def _generate_uuid(self) -> str:
        """Generate a random UUID-like string."""
        parts = [
            f"{self._rng.randint(0, 0xFFFFFFFF):08x}",
            f"{self._rng.randint(0, 0xFFFF):04x}",
            f"4{self._rng.randint(0, 0xFFF):03x}",
            f"{self._rng.choice(['8', '9', 'a', 'b'])}{self._rng.randint(0, 0xFFF):03x}",
            f"{self._rng.randint(0, 0xFFFFFFFFFFFF):012x}",
        ]
        return "-".join(parts)

    def _random_word(self) -> str:
        """Generate a random word-like string."""
        length = self._rng.randint(4, 10)
        return "".join(self._rng.choices(string.ascii_lowercase, k=length))

    def _random_text(self, length: int) -> str:
        """Generate random text of given length."""
        if length <= 0:
            return ""
        words = []
        remaining = length
        while remaining > 0:
            word_len = min(self._rng.randint(3, 8), remaining)
            words.append("".join(self._rng.choices(string.ascii_lowercase, k=word_len)))
            remaining -= word_len + 1  # +1 for space
        return " ".join(words)[:length]

    @property
    def call_count(self) -> int:
        """Number of generation calls made."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


class MockChatMessage(BaseModel):
    """A mock chat message for testing."""

    role: str
    content: str


class MockCompletionResponse(BaseModel):
    """A mock completion response."""

    message: MockChatMessage
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class MockChatProvider(MockLLMProvider):
    """Mock provider specifically for chat-style completions.

    Wraps MockLLMProvider to provide a chat-like interface matching
    what atomic-agents expects.
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
        _temperature: float = 0.7,
        _max_tokens: int = 1000,
    ) -> BaseModel:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            output_schema: Pydantic model for structured output.
            system_prompt: Optional system prompt.
            _temperature: Temperature (ignored by mock).
            _max_tokens: Max tokens (ignored by mock).

        Returns:
            Instance of output_schema with generated data.
        """
        # Context could include messages for smarter mocking in future
        context = {
            "messages": messages,
            "system_prompt": system_prompt,
        }
        return self.generate(output_schema, _context=context)

    async def complete_async(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> BaseModel:
        """Async version of complete (just calls sync version)."""
        return self.complete(
            messages,
            output_schema,
            system_prompt=system_prompt,
            _temperature=temperature,
            _max_tokens=max_tokens,
        )
