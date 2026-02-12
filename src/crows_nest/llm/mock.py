"""Mock LLM provider for testing and exploration.

Provides several mock behaviors:
- Echo: Reflects input back with optional transformation
- Scripted: Returns predefined responses in sequence
- Pattern: Matches input patterns to responses
- Deterministic: Hash-based reproducible responses
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class MockResponse:
    """A mock response with optional metadata."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"


@dataclass
class MockAgent(Generic[InputT, OutputT]):
    """A mock agent that doesn't call any LLM API.

    Useful for:
    - Testing pipelines without API costs
    - Exploring conversation flows
    - Deterministic reproduction of scenarios
    """

    output_schema: type[OutputT]
    """The Pydantic model for output."""

    mode: str = "echo"
    """Mode: 'echo', 'scripted', 'pattern', 'deterministic'."""

    responses: list[str] = field(default_factory=list)
    """Canned responses for 'scripted' mode."""

    patterns: dict[str, str] = field(default_factory=dict)
    """Pattern -> response mapping for 'pattern' mode."""

    echo_template: str = "I received: {input}\n\nReflecting on this..."
    """Template for 'echo' mode. {input} is replaced with user input."""

    deterministic_seed: str = "crows-nest"
    """Seed for deterministic mode hash."""

    _response_index: int = field(default=0, repr=False)

    def _get_input_text(self, user_input: InputT) -> str:
        """Extract text representation from input."""
        if hasattr(user_input, "chat_message"):
            return str(user_input.chat_message)
        if hasattr(user_input, "message"):
            return str(user_input.message)
        if hasattr(user_input, "content"):
            return str(user_input.content)
        return str(user_input.model_dump())

    def _generate_response(self, user_input: InputT) -> str:
        """Generate response based on mode."""
        input_text = self._get_input_text(user_input)

        if self.mode == "echo":
            return self.echo_template.format(input=input_text)

        elif self.mode == "scripted":
            if not self.responses:
                return f"[No scripted responses configured. Input was: {input_text}]"
            response = self.responses[self._response_index % len(self.responses)]
            self._response_index += 1
            return response

        elif self.mode == "pattern":
            for pattern, response in self.patterns.items():
                if re.search(pattern, input_text, re.IGNORECASE):
                    return response
            return f"[No pattern matched. Input was: {input_text}]"

        elif self.mode == "deterministic":
            # Generate deterministic response based on input hash
            hash_input = f"{self.deterministic_seed}:{input_text}"
            hash_val = hashlib.sha256(hash_input.encode()).hexdigest()

            # Use hash to select from a set of template responses
            templates = [
                "Considering '{input}', I think the key insight is structure and composition.",
                "The concept of '{input}' relates to building abstractions layer by layer.",
                "When we examine '{input}', we see patterns of transformation and flow.",
                "'{input}' suggests a need for both flexibility and constraint.",
                "Reflecting on '{input}': this is about making the implicit explicit.",
            ]
            idx = int(hash_val[:8], 16) % len(templates)
            return templates[idx].format(input=input_text[:50])

        else:
            return f"[Unknown mode: {self.mode}. Input was: {input_text}]"

    def _build_output(self, response: str) -> OutputT:
        """Build output model from response string."""
        # Try to fit response into output schema
        output_fields = self.output_schema.model_fields

        # Common patterns for chat output schemas
        if "response" in output_fields:
            return self.output_schema(response=response)
        if "content" in output_fields:
            return self.output_schema(content=response)
        if "chat_message" in output_fields:
            return self.output_schema(chat_message=response)
        if "message" in output_fields:
            return self.output_schema(message=response)

        # Try to construct with just the response as first field
        first_field = next(iter(output_fields.keys()))
        return self.output_schema(**{first_field: response})

    def run(self, user_input: InputT) -> OutputT:
        """Run synchronously."""
        response = self._generate_response(user_input)
        return self._build_output(response)

    async def run_async(self, user_input: InputT) -> OutputT:
        """Run asynchronously (same as sync for mock)."""
        return self.run(user_input)

    async def run_async_stream(self, user_input: InputT) -> Any:
        """Stream response word by word."""
        response = self._generate_response(user_input)
        words = response.split()

        # Yield partial outputs
        accumulated = ""
        for i, word in enumerate(words):
            accumulated += word + (" " if i < len(words) - 1 else "")
            yield self._build_output(accumulated)


@dataclass
class ConversationRecorder:
    """Records conversations for replay or analysis.

    Can be used to capture real LLM interactions and replay them
    as mock responses for testing.
    """

    turns: list[dict[str, Any]] = field(default_factory=list)

    def record(self, user_input: Any, response: Any) -> None:
        """Record a conversation turn."""
        input_data = (
            user_input.model_dump() if hasattr(user_input, "model_dump") else str(user_input)
        )
        output_data = response.model_dump() if hasattr(response, "model_dump") else str(response)
        self.turns.append({"input": input_data, "output": output_data})

    def to_scripted_responses(self) -> list[str]:
        """Extract responses for scripted mock mode."""
        responses = []
        for turn in self.turns:
            output = turn["output"]
            if isinstance(output, dict):
                # Try common response field names
                for key in ["response", "content", "message", "chat_message"]:
                    if key in output:
                        responses.append(str(output[key]))
                        break
                else:
                    responses.append(json.dumps(output))
            else:
                responses.append(str(output))
        return responses

    def save(self, path: str) -> None:
        """Save conversation to file."""
        from pathlib import Path

        with Path(path).open("w") as f:
            json.dump(self.turns, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> ConversationRecorder:
        """Load conversation from file."""
        from pathlib import Path

        with Path(path).open() as f:
            turns = json.load(f)
        return cls(turns=turns)
