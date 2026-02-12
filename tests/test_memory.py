"""Tests for the memory system."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from crows_nest.memory import (
    ContextMessage,
    ContextWindow,
    MemoryEntry,
    MemoryQuery,
    MemoryScope,
    MemoryStats,
    MemoryType,
    SQLiteMemoryStore,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types_exist(self) -> None:
        """All memory types are defined."""
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"

    def test_memory_type_from_string(self) -> None:
        """MemoryType can be created from string."""
        assert MemoryType("episodic") == MemoryType.EPISODIC
        assert MemoryType("semantic") == MemoryType.SEMANTIC
        assert MemoryType("procedural") == MemoryType.PROCEDURAL


class TestMemoryScope:
    """Tests for MemoryScope enum."""

    def test_memory_scopes_exist(self) -> None:
        """All memory scopes are defined."""
        assert MemoryScope.PRIVATE == "private"
        assert MemoryScope.CHILDREN == "children"
        assert MemoryScope.GLOBAL == "global"


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_memory_entry_creation(self) -> None:
        """MemoryEntry can be created with defaults."""
        agent_id = uuid4()
        entry = MemoryEntry(agent_id=agent_id, content="Test memory")

        assert entry.agent_id == agent_id
        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.scope == MemoryScope.PRIVATE
        assert entry.importance == 0.5
        assert entry.access_count == 0
        assert entry.id is not None
        assert entry.created_at is not None
        assert entry.metadata == {}

    def test_memory_entry_touch(self) -> None:
        """Touch updates access time and count."""
        entry = MemoryEntry(agent_id=uuid4(), content="Test")
        original_accessed = entry.accessed_at
        original_count = entry.access_count

        # Small delay to ensure time difference
        import time

        time.sleep(0.01)
        entry.touch()

        assert entry.accessed_at > original_accessed
        assert entry.access_count == original_count + 1

    def test_memory_entry_is_expired(self) -> None:
        """Expiration check works correctly."""
        entry = MemoryEntry(agent_id=uuid4(), content="Test")
        assert not entry.is_expired()

        # Set expiration in past
        entry.expires_at = datetime.now(UTC) - timedelta(hours=1)
        assert entry.is_expired()

        # Set expiration in future
        entry.expires_at = datetime.now(UTC) + timedelta(hours=1)
        assert not entry.is_expired()

    def test_memory_entry_age_seconds(self) -> None:
        """Age calculation works."""
        entry = MemoryEntry(agent_id=uuid4(), content="Test")
        assert entry.age_seconds >= 0
        assert entry.age_seconds < 1  # Should be nearly instant

    def test_memory_entry_staleness_seconds(self) -> None:
        """Staleness calculation works."""
        entry = MemoryEntry(agent_id=uuid4(), content="Test")
        assert entry.staleness_seconds >= 0
        assert entry.staleness_seconds < 1

    def test_memory_entry_decay_score(self) -> None:
        """Decay score calculation works."""
        entry = MemoryEntry(agent_id=uuid4(), content="Test", importance=0.8)
        score = entry.decay_score()

        # Fresh memory with high importance should have decent score
        assert 0 < score <= 1

        # Higher importance should give higher score
        entry_low = MemoryEntry(agent_id=uuid4(), content="Test", importance=0.2)
        assert entry.decay_score() > entry_low.decay_score()

    def test_memory_entry_to_dict(self) -> None:
        """Serialization to dict works."""
        agent_id = uuid4()
        entry = MemoryEntry(
            agent_id=agent_id,
            content="Test",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.GLOBAL,
            metadata={"key": "value"},
        )
        d = entry.to_dict()

        assert d["agent_id"] == str(agent_id)
        assert d["content"] == "Test"
        assert d["memory_type"] == "episodic"
        assert d["scope"] == "global"
        assert d["metadata"] == {"key": "value"}

    def test_memory_entry_from_dict(self) -> None:
        """Deserialization from dict works."""
        agent_id = uuid4()
        entry = MemoryEntry(
            agent_id=agent_id,
            content="Test",
            memory_type=MemoryType.PROCEDURAL,
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)

        assert restored.id == entry.id
        assert restored.agent_id == entry.agent_id
        assert restored.content == entry.content
        assert restored.memory_type == entry.memory_type


class TestMemoryQuery:
    """Tests for MemoryQuery dataclass."""

    def test_memory_query_defaults(self) -> None:
        """MemoryQuery has sensible defaults."""
        query = MemoryQuery()
        assert query.text is None
        assert query.memory_type is None
        assert query.scope is None
        assert query.min_importance == 0.0
        assert query.limit == 10
        assert query.offset == 0
        assert query.include_expired is False

    def test_memory_query_to_dict(self) -> None:
        """MemoryQuery serializes to dict."""
        query = MemoryQuery(
            text="search",
            memory_type=MemoryType.SEMANTIC,
            min_importance=0.5,
        )
        d = query.to_dict()
        assert d["text"] == "search"
        assert d["memory_type"] == "semantic"
        assert d["min_importance"] == 0.5


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_memory_stats_defaults(self) -> None:
        """MemoryStats has sensible defaults."""
        agent_id = uuid4()
        stats = MemoryStats(agent_id=agent_id)
        assert stats.total_count == 0
        assert stats.by_type == {}
        assert stats.by_scope == {}
        assert stats.avg_importance == 0.0
        assert stats.total_size_bytes == 0

    def test_memory_stats_to_dict(self) -> None:
        """MemoryStats serializes to dict."""
        agent_id = uuid4()
        stats = MemoryStats(
            agent_id=agent_id,
            total_count=10,
            by_type={"semantic": 5, "episodic": 5},
            avg_importance=0.7,
        )
        d = stats.to_dict()
        assert d["total_count"] == 10
        assert d["by_type"] == {"semantic": 5, "episodic": 5}


class TestSQLiteMemoryStore:
    """Tests for SQLiteMemoryStore."""

    @pytest.fixture
    async def store(self) -> SQLiteMemoryStore:
        """Create an in-memory SQLite store for testing."""
        store = SQLiteMemoryStore(":memory:")
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_store_initialize(self) -> None:
        """Store initializes correctly."""
        store = SQLiteMemoryStore(":memory:")
        await store.initialize()
        assert store._conn is not None
        await store.close()

    @pytest.mark.asyncio
    async def test_store_close(self) -> None:
        """Store closes correctly."""
        store = SQLiteMemoryStore(":memory:")
        await store.initialize()
        await store.close()
        assert store._conn is None

    @pytest.mark.asyncio
    async def test_remember_creates_entry(self, store: SQLiteMemoryStore) -> None:
        """Remember creates a new memory entry."""
        agent_id = uuid4()
        entry = await store.remember(
            agent_id=agent_id,
            content="Test memory",
            importance=0.8,
        )

        assert entry.agent_id == agent_id
        assert entry.content == "Test memory"
        assert entry.importance == 0.8

    @pytest.mark.asyncio
    async def test_remember_clamps_importance(self, store: SQLiteMemoryStore) -> None:
        """Remember clamps importance to [0, 1]."""
        agent_id = uuid4()

        entry_high = await store.remember(agent_id=agent_id, content="High", importance=1.5)
        assert entry_high.importance == 1.0

        entry_low = await store.remember(agent_id=agent_id, content="Low", importance=-0.5)
        assert entry_low.importance == 0.0

    @pytest.mark.asyncio
    async def test_recall_returns_memories(self, store: SQLiteMemoryStore) -> None:
        """Recall returns matching memories."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="First memory")
        await store.remember(agent_id=agent_id, content="Second memory")

        query = MemoryQuery()
        memories = await store.recall(agent_id=agent_id, query=query)

        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_recall_filters_by_type(self, store: SQLiteMemoryStore) -> None:
        """Recall filters by memory type."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="Fact", memory_type=MemoryType.SEMANTIC)
        await store.remember(agent_id=agent_id, content="Event", memory_type=MemoryType.EPISODIC)

        query = MemoryQuery(memory_type=MemoryType.SEMANTIC)
        memories = await store.recall(agent_id=agent_id, query=query)

        assert len(memories) == 1
        assert memories[0].memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_recall_filters_by_text(self, store: SQLiteMemoryStore) -> None:
        """Recall filters by text content."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="The sky is blue")
        await store.remember(agent_id=agent_id, content="Grass is green")

        query = MemoryQuery(text="sky")
        memories = await store.recall(agent_id=agent_id, query=query)

        assert len(memories) == 1
        assert "sky" in memories[0].content

    @pytest.mark.asyncio
    async def test_recall_filters_by_importance(self, store: SQLiteMemoryStore) -> None:
        """Recall filters by minimum importance."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="Important", importance=0.9)
        await store.remember(agent_id=agent_id, content="Less important", importance=0.3)

        query = MemoryQuery(min_importance=0.5)
        memories = await store.recall(agent_id=agent_id, query=query)

        assert len(memories) == 1
        assert memories[0].importance >= 0.5

    @pytest.mark.asyncio
    async def test_recall_respects_limit(self, store: SQLiteMemoryStore) -> None:
        """Recall respects limit parameter."""
        agent_id = uuid4()

        for i in range(5):
            await store.remember(agent_id=agent_id, content=f"Memory {i}")

        query = MemoryQuery(limit=3)
        memories = await store.recall(agent_id=agent_id, query=query)

        assert len(memories) == 3

    @pytest.mark.asyncio
    async def test_get_returns_memory(self, store: SQLiteMemoryStore) -> None:
        """Get returns a specific memory."""
        agent_id = uuid4()
        entry = await store.remember(agent_id=agent_id, content="Test")

        retrieved = await store.get(agent_id=agent_id, memory_id=entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.content == entry.content

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, store: SQLiteMemoryStore) -> None:
        """Get returns None for non-existent memory."""
        agent_id = uuid4()
        result = await store.get(agent_id=agent_id, memory_id=uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_update_modifies_memory(self, store: SQLiteMemoryStore) -> None:
        """Update modifies memory content."""
        agent_id = uuid4()
        entry = await store.remember(agent_id=agent_id, content="Original")

        updated = await store.update(
            agent_id=agent_id,
            memory_id=entry.id,
            content="Updated",
            importance=0.9,
        )

        assert updated is not None
        assert updated.content == "Updated"
        assert updated.importance == 0.9

    @pytest.mark.asyncio
    async def test_update_returns_none_for_missing(self, store: SQLiteMemoryStore) -> None:
        """Update returns None for non-existent memory."""
        agent_id = uuid4()
        result = await store.update(agent_id=agent_id, memory_id=uuid4(), content="Updated")
        assert result is None

    @pytest.mark.asyncio
    async def test_forget_deletes_memory(self, store: SQLiteMemoryStore) -> None:
        """Forget deletes a memory."""
        agent_id = uuid4()
        entry = await store.remember(agent_id=agent_id, content="To delete")

        deleted = await store.forget(agent_id=agent_id, memory_id=entry.id)
        assert deleted is True

        # Verify it's gone
        result = await store.get(agent_id=agent_id, memory_id=entry.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_forget_returns_false_for_missing(self, store: SQLiteMemoryStore) -> None:
        """Forget returns False for non-existent memory."""
        agent_id = uuid4()
        result = await store.forget(agent_id=agent_id, memory_id=uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_forget_all_deletes_by_type(self, store: SQLiteMemoryStore) -> None:
        """Forget all deletes memories by type."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="Semantic", memory_type=MemoryType.SEMANTIC)
        await store.remember(agent_id=agent_id, content="Episodic", memory_type=MemoryType.EPISODIC)

        count = await store.forget_all(agent_id=agent_id, memory_type=MemoryType.SEMANTIC)
        assert count == 1

        # Verify semantic is gone but episodic remains
        query = MemoryQuery()
        remaining = await store.recall(agent_id=agent_id, query=query)
        assert len(remaining) == 1
        assert remaining[0].memory_type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_decay_removes_stale_memories(self, store: SQLiteMemoryStore) -> None:
        """Decay removes memories below threshold."""
        agent_id = uuid4()

        # Create memory with very low importance (will decay fast)
        await store.remember(agent_id=agent_id, content="Low importance", importance=0.01)
        await store.remember(agent_id=agent_id, content="High importance", importance=0.99)

        # Use very short half-life to simulate decay
        count = await store.decay(agent_id=agent_id, threshold=0.1, half_life_seconds=0.001)

        # Low importance memory should be removed
        assert count >= 1

    @pytest.mark.asyncio
    async def test_get_stats_returns_statistics(self, store: SQLiteMemoryStore) -> None:
        """Get stats returns memory statistics."""
        agent_id = uuid4()

        await store.remember(
            agent_id=agent_id,
            content="Test 1",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )
        await store.remember(
            agent_id=agent_id,
            content="Test 2",
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
        )

        stats = await store.get_stats(agent_id=agent_id)

        assert stats.total_count == 2
        assert stats.by_type.get("semantic") == 1
        assert stats.by_type.get("episodic") == 1
        assert stats.avg_importance == 0.7  # (0.8 + 0.6) / 2

    @pytest.mark.asyncio
    async def test_export_memories(self, store: SQLiteMemoryStore) -> None:
        """Export returns all memories as dicts."""
        agent_id = uuid4()

        await store.remember(agent_id=agent_id, content="First")
        await store.remember(agent_id=agent_id, content="Second")

        exported = await store.export_memories(agent_id=agent_id)

        assert len(exported) == 2
        assert all("content" in m for m in exported)

    @pytest.mark.asyncio
    async def test_import_memories(self, store: SQLiteMemoryStore) -> None:
        """Import creates memories from dicts."""
        agent_id = uuid4()
        other_agent = uuid4()

        # Create and export from one agent
        entry = await store.remember(agent_id=other_agent, content="To import")
        exported = await store.export_memories(agent_id=other_agent)

        # Delete original so ID doesn't conflict
        await store.forget(agent_id=other_agent, memory_id=entry.id)

        # Import to different agent
        count = await store.import_memories(agent_id=agent_id, memories=exported)

        assert count == 1

        # Verify imported to new agent
        query = MemoryQuery()
        memories = await store.recall(agent_id=agent_id, query=query)
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_can_access_private(self, store: SQLiteMemoryStore) -> None:
        """Private memories only accessible by owner."""
        owner_id = uuid4()
        other_id = uuid4()

        entry = MemoryEntry(
            agent_id=owner_id,
            content="Private",
            scope=MemoryScope.PRIVATE,
        )

        assert store.can_access(owner_id, entry) is True
        assert store.can_access(other_id, entry) is False

    @pytest.mark.asyncio
    async def test_can_access_global(self, store: SQLiteMemoryStore) -> None:
        """Global memories accessible by anyone."""
        owner_id = uuid4()
        other_id = uuid4()

        entry = MemoryEntry(
            agent_id=owner_id,
            content="Global",
            scope=MemoryScope.GLOBAL,
        )

        assert store.can_access(owner_id, entry) is True
        assert store.can_access(other_id, entry) is True

    @pytest.mark.asyncio
    async def test_can_access_children(self, store: SQLiteMemoryStore) -> None:
        """Children scope accessible by descendants."""
        parent_id = uuid4()
        child_id = uuid4()
        unrelated_id = uuid4()

        entry = MemoryEntry(
            agent_id=parent_id,
            content="For children",
            scope=MemoryScope.CHILDREN,
        )

        # Child with parent in ancestors can access
        assert store.can_access(child_id, entry, agent_ancestors=[parent_id]) is True

        # Unrelated agent cannot
        assert store.can_access(unrelated_id, entry, agent_ancestors=[]) is False


class TestContextMessage:
    """Tests for ContextMessage."""

    def test_context_message_creation(self) -> None:
        """ContextMessage can be created."""
        msg = ContextMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.id is not None
        assert msg.timestamp is not None

    def test_context_message_to_dict(self) -> None:
        """ContextMessage serializes to dict."""
        msg = ContextMessage(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"

    def test_context_message_from_dict(self) -> None:
        """ContextMessage deserializes from dict."""
        msg = ContextMessage(role="user", content="Test")
        d = msg.to_dict()
        restored = ContextMessage.from_dict(d)
        assert restored.id == msg.id
        assert restored.role == msg.role
        assert restored.content == msg.content


class TestContextWindow:
    """Tests for ContextWindow."""

    def test_context_window_creation(self) -> None:
        """ContextWindow can be created."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id, max_tokens=4000)
        assert window.agent_id == agent_id
        assert window.max_tokens == 4000
        assert window.total_tokens == 0
        assert len(window.messages) == 0

    def test_add_message(self) -> None:
        """Add message increases token count."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id)

        msg = window.add_message("user", "Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert window.total_tokens > 0
        assert len(window.messages) == 1

    def test_get_messages(self) -> None:
        """Get messages returns all messages."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id)

        window.add_message("user", "Hello")
        window.add_message("assistant", "Hi there")

        messages = window.get_messages()
        assert len(messages) == 2

    def test_needs_summarization(self) -> None:
        """Needs summarization when over threshold."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id, max_tokens=100, summarize_threshold=0.5)

        assert not window.needs_summarization

        # Add enough content to exceed threshold
        window.add_message("user", "a " * 100)  # Many tokens
        assert window.needs_summarization

    @pytest.mark.asyncio
    async def test_summarize_compacts_window(self) -> None:
        """Summarize reduces message count."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id, max_tokens=100, summarize_threshold=0.3)

        # Add many messages
        for i in range(10):
            window.add_message("user", f"Message {i} with some content")

        original_count = len(window.messages)
        await window.summarize()

        # Should have fewer messages after summarization
        assert len(window.messages) < original_count
        # Should have created a summary
        assert len(window.summaries) > 0

    def test_clear(self) -> None:
        """Clear removes messages but keeps summaries."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id)

        window.add_message("user", "Hello")
        window.summaries.append("Previous summary")
        window.clear()

        assert len(window.messages) == 0
        assert len(window.summaries) == 1

    def test_clear_all(self) -> None:
        """Clear all removes messages and summaries."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id)

        window.add_message("user", "Hello")
        window.summaries.append("Previous summary")
        window.clear_all()

        assert len(window.messages) == 0
        assert len(window.summaries) == 0

    def test_to_dict(self) -> None:
        """Window serializes to dict."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id, max_tokens=2000)
        window.add_message("user", "Hello")

        d = window.to_dict()
        assert d["agent_id"] == str(agent_id)
        assert d["max_tokens"] == 2000
        assert len(d["messages"]) == 1

    def test_from_dict(self) -> None:
        """Window deserializes from dict."""
        agent_id = uuid4()
        window = ContextWindow(agent_id=agent_id)
        window.add_message("user", "Hello")

        d = window.to_dict()
        restored = ContextWindow.from_dict(d)

        assert restored.agent_id == agent_id
        assert len(restored.messages) == 1
