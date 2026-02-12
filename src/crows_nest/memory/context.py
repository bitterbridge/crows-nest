"""Context window management with auto-summarization.

Provides context window tracking with automatic summarization
when the window exceeds capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from crows_nest.memory.models import MemoryType

if TYPE_CHECKING:
    from collections.abc import Callable

    from crows_nest.memory.store import MemoryStore


@dataclass
class ContextMessage:
    """A message in the context window.

    Attributes:
        id: Unique message identifier.
        role: Message role (system, user, assistant, tool).
        content: Message content.
        timestamp: When the message was added.
        token_count: Estimated token count.
        metadata: Additional message metadata.
    """

    role: str
    content: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextMessage:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            role=data["role"],
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(UTC)
            ),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )


def simple_token_counter(text: str) -> int:
    """Simple token estimation based on word count.

    Approximates tokens as words * 1.3 (rough average for English).
    For production, use a proper tokenizer.
    """
    words = len(text.split())
    return int(words * 1.3)


@dataclass
class ContextWindow:
    """Manages a context window with auto-summarization.

    The context window tracks messages and automatically summarizes
    older content when the window exceeds its token capacity.

    Usage:
        window = ContextWindow(max_tokens=4000)
        window.add_message("user", "Hello!")
        window.add_message("assistant", "Hi there!")

        # When window is full, old messages are summarized
        messages = window.get_messages()

    Attributes:
        agent_id: ID of the agent this window belongs to.
        max_tokens: Maximum tokens before summarization triggers.
        summarize_threshold: Fraction of max_tokens that triggers summarization.
        messages: Current messages in the window.
        summaries: List of past summaries.
        total_tokens: Current token count.
    """

    agent_id: UUID
    max_tokens: int = 4000
    summarize_threshold: float = 0.9
    token_counter: Callable[[str], int] = simple_token_counter
    messages: list[ContextMessage] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    total_tokens: int = 0
    _memory_store: MemoryStore | None = None
    _summarizer: Callable[[list[ContextMessage]], str] | None = None

    def set_memory_store(self, store: MemoryStore) -> None:
        """Set the memory store for persisting summaries."""
        self._memory_store = store

    def set_summarizer(self, summarizer: Callable[[list[ContextMessage]], str]) -> None:
        """Set a custom summarization function.

        The summarizer receives a list of messages and should return
        a summary string.
        """
        self._summarizer = summarizer

    @property
    def trigger_threshold(self) -> int:
        """Token count that triggers summarization."""
        return int(self.max_tokens * self.summarize_threshold)

    @property
    def needs_summarization(self) -> bool:
        """Check if the window needs summarization."""
        return self.total_tokens >= self.trigger_threshold

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextMessage:
        """Add a message to the context window.

        Args:
            role: Message role (system, user, assistant, tool).
            content: Message content.
            metadata: Optional message metadata.

        Returns:
            The created ContextMessage.
        """
        token_count = self.token_counter(content)
        message = ContextMessage(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.total_tokens += token_count
        return message

    def get_messages(
        self,
        include_summaries: bool = True,
    ) -> list[ContextMessage]:
        """Get messages for the current context.

        Args:
            include_summaries: Whether to include summary messages.

        Returns:
            List of context messages.
        """
        result = []

        # Include summaries as system messages if present
        if include_summaries and self.summaries:
            combined_summary = "\n\n".join(self.summaries)
            summary_msg = ContextMessage(
                role="system",
                content=f"[Previous conversation summary]\n{combined_summary}",
                metadata={"is_summary": True},
            )
            result.append(summary_msg)

        result.extend(self.messages)
        return result

    async def summarize(self) -> str | None:
        """Summarize older messages and compact the window.

        Returns:
            The summary string if summarization occurred, None otherwise.
        """
        if not self.needs_summarization:
            return None

        if len(self.messages) < 2:
            return None

        # Keep recent messages (approximately 1/3 of capacity)
        keep_tokens = self.max_tokens // 3
        keep_count = 0
        kept_tokens = 0

        # Count from the end to find how many to keep
        for msg in reversed(self.messages):
            if kept_tokens + msg.token_count > keep_tokens:
                break
            kept_tokens += msg.token_count
            keep_count += 1

        # Ensure we keep at least the most recent message
        keep_count = max(1, keep_count)

        # Split messages
        to_summarize = self.messages[:-keep_count] if keep_count else self.messages
        to_keep = self.messages[-keep_count:] if keep_count else []

        if not to_summarize:
            return None

        # Generate summary
        summary = self._generate_summary(to_summarize)
        self.summaries.append(summary)

        # Store summary in memory if available
        if self._memory_store is not None:
            await self._memory_store.remember(
                agent_id=self.agent_id,
                content=summary,
                memory_type=MemoryType.EPISODIC,
                importance=0.7,
                metadata={
                    "type": "context_summary",
                    "message_count": len(to_summarize),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        # Update window
        self.messages = to_keep
        self.total_tokens = sum(m.token_count for m in to_keep)

        return summary

    def _generate_summary(self, messages: list[ContextMessage]) -> str:
        """Generate a summary of messages.

        If a custom summarizer is set, uses that. Otherwise uses
        a simple concatenation approach.
        """
        if self._summarizer is not None:
            return self._summarizer(messages)

        # Default: simple extraction of key points
        lines = []
        for msg in messages:
            if msg.role == "system":
                continue
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"[{msg.role}]: {content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages (keeps summaries)."""
        self.messages = []
        self.total_tokens = 0

    def clear_all(self) -> None:
        """Clear all messages and summaries."""
        self.messages = []
        self.summaries = []
        self.total_tokens = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "max_tokens": self.max_tokens,
            "summarize_threshold": self.summarize_threshold,
            "messages": [m.to_dict() for m in self.messages],
            "summaries": self.summaries,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        token_counter: Callable[[str], int] | None = None,
    ) -> ContextWindow:
        """Create from dictionary."""
        window = cls(
            agent_id=UUID(data["agent_id"]),
            max_tokens=data.get("max_tokens", 4000),
            summarize_threshold=data.get("summarize_threshold", 0.9),
            token_counter=token_counter or simple_token_counter,
        )
        window.messages = [ContextMessage.from_dict(m) for m in data.get("messages", [])]
        window.summaries = data.get("summaries", [])
        window.total_tokens = data.get("total_tokens", 0)
        return window
