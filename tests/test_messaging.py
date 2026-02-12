"""Tests for the messaging system (events, queues, bus)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.messaging import (
    Event,
    EventBus,
    Message,
    MessageQueue,
    MessageStatus,
    QueueStats,
    ThunkCompletedEvent,
    ThunkFailedEvent,
    ThunkStartedEvent,
    deserialize_event,
    get_event_bus,
    reset_event_bus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator


@pytest.fixture(autouse=True)
def reset_bus() -> Generator[None, None, None]:
    """Reset the global event bus before and after each test."""
    reset_event_bus()
    yield
    reset_event_bus()


class TestEvent:
    """Tests for the Event base class."""

    def test_event_creation(self) -> None:
        """Event can be created with default values."""
        event = Event(event_type="test.event")
        assert event.event_type == "test.event"
        assert event.id is not None
        assert event.timestamp is not None
        assert event.trace_id is not None
        assert event.payload == {}

    def test_event_to_dict(self) -> None:
        """Event serializes to dict."""
        event = Event(event_type="test.event", payload={"key": "value"})
        d = event.to_dict()
        assert d["event_type"] == "test.event"
        assert d["payload"] == {"key": "value"}
        assert "id" in d
        assert "timestamp" in d

    def test_event_from_dict(self) -> None:
        """Event deserializes from dict."""
        event = Event(event_type="test.event", payload={"key": "value"})
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.id == event.id
        assert restored.payload == event.payload


class TestThunkEvents:
    """Tests for thunk lifecycle events."""

    def test_thunk_started_event(self) -> None:
        """ThunkStartedEvent has correct type."""
        thunk_id = uuid4()
        event = ThunkStartedEvent(
            thunk_id=thunk_id,
            operation="test.op",
        )
        assert event.event_type == "thunk.started"
        assert event.thunk_id == thunk_id
        assert event.operation == "test.op"

    def test_thunk_completed_event(self) -> None:
        """ThunkCompletedEvent has correct type and duration."""
        thunk_id = uuid4()
        event = ThunkCompletedEvent(
            thunk_id=thunk_id,
            operation="test.op",
            duration_ms=150,
        )
        assert event.event_type == "thunk.completed"
        assert event.duration_ms == 150

    def test_thunk_failed_event(self) -> None:
        """ThunkFailedEvent captures error info."""
        thunk_id = uuid4()
        event = ThunkFailedEvent(
            thunk_id=thunk_id,
            operation="test.op",
            error_type="ValueError",
            error_message="Something went wrong",
            duration_ms=100,
        )
        assert event.event_type == "thunk.failed"
        assert event.error_type == "ValueError"
        assert event.error_message == "Something went wrong"

    def test_deserialize_thunk_event(self) -> None:
        """Thunk events deserialize correctly."""
        event = ThunkStartedEvent(
            thunk_id=uuid4(),
            operation="test.op",
        )
        d = event.to_dict()
        d["event_type"] = "thunk.started"  # Ensure type is in dict
        restored = deserialize_event(d)
        assert isinstance(restored, ThunkStartedEvent)


class TestEventBus:
    """Tests for the EventBus pub/sub system."""

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self) -> None:
        """Publishing with no subscribers returns 0."""
        bus = EventBus()
        event = Event(event_type="test.event")
        count = await bus.publish(event)
        assert count == 0

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self) -> None:
        """Subscriber receives published events."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        event = Event(event_type="test.event")
        count = await bus.publish(event)

        assert count == 1
        assert len(received) == 1
        assert received[0] == event

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        """Multiple subscribers all receive events."""
        bus = EventBus()
        received1: list[Event] = []
        received2: list[Event] = []

        async def handler1(event: Event) -> None:
            received1.append(event)

        async def handler2(event: Event) -> None:
            received2.append(event)

        bus.subscribe("test.event", handler1)
        bus.subscribe("test.event", handler2)

        event = Event(event_type="test.event")
        count = await bus.publish(event)

        assert count == 2
        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self) -> None:
        """Wildcard patterns match multiple event types."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.*", handler)

        await bus.publish(Event(event_type="test.one"))
        await bus.publish(Event(event_type="test.two"))
        await bus.publish(Event(event_type="other.event"))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """Unsubscribed handler no longer receives events."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        await bus.publish(Event(event_type="test.event"))
        assert len(received) == 1

        bus.unsubscribe("test.event", handler)
        await bus.publish(Event(event_type="test.event"))
        assert len(received) == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_sync_handler(self) -> None:
        """Synchronous handlers are supported."""
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        await bus.publish(Event(event_type="test.event"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_others(self) -> None:
        """Handler errors don't prevent other handlers from receiving events."""
        bus = EventBus()
        received: list[Event] = []

        async def bad_handler(event: Event) -> None:
            raise ValueError("Handler error")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", bad_handler)
        bus.subscribe("test.event", good_handler)

        # Should not raise
        count = await bus.publish(Event(event_type="test.event"))

        # Good handler still received it
        assert len(received) == 1
        # But both were "called"
        assert count == 1  # Only successful ones counted

    def test_get_handler_count(self) -> None:
        """Handler count is tracked correctly."""
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        assert bus.get_handler_count() == 0
        bus.subscribe("test.event", handler)
        assert bus.get_handler_count() == 1
        assert bus.get_handler_count("test.event") == 1
        assert bus.get_handler_count("other.event") == 0

    def test_clear(self) -> None:
        """Clear removes all subscriptions."""
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("test.event", handler)
        bus.subscribe("other.event", handler)
        assert bus.get_handler_count() == 2

        bus.clear()
        assert bus.get_handler_count() == 0


class TestGlobalEventBus:
    """Tests for the global event bus instance."""

    def test_get_event_bus_singleton(self) -> None:
        """get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus(self) -> None:
        """reset_event_bus creates a new instance."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2


class TestMessage:
    """Tests for the Message class."""

    def test_message_creation(self) -> None:
        """Message can be created with defaults."""
        msg = Message(queue_name="test", payload={"key": "value"})
        assert msg.queue_name == "test"
        assert msg.payload == {"key": "value"}
        assert msg.status == MessageStatus.PENDING
        assert msg.attempts == 0
        assert msg.max_attempts == 3

    def test_message_to_dict(self) -> None:
        """Message serializes to dict."""
        msg = Message(queue_name="test", payload={"key": "value"})
        d = msg.to_dict()
        assert d["queue_name"] == "test"
        assert d["payload"] == {"key": "value"}
        assert d["status"] == "pending"

    def test_message_from_dict(self) -> None:
        """Message deserializes from dict."""
        msg = Message(queue_name="test", payload={"key": "value"})
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.queue_name == msg.queue_name
        assert restored.payload == msg.payload

    def test_message_mark_processing(self) -> None:
        """mark_processing updates status and increments attempts."""
        msg = Message(queue_name="test", payload={})
        msg.mark_processing()
        assert msg.status == MessageStatus.PROCESSING
        assert msg.attempts == 1

    def test_message_mark_completed(self) -> None:
        """mark_completed updates status."""
        msg = Message(queue_name="test", payload={})
        msg.mark_processing()
        msg.mark_completed()
        assert msg.status == MessageStatus.COMPLETED

    def test_message_mark_failed_with_retry(self) -> None:
        """mark_failed sets FAILED status when retries available."""
        msg = Message(queue_name="test", payload={}, max_attempts=3)
        msg.mark_processing()
        msg.mark_failed("Error")
        assert msg.status == MessageStatus.FAILED
        assert msg.can_retry()

    def test_message_mark_failed_to_dead(self) -> None:
        """mark_failed sets DEAD status when no retries left."""
        msg = Message(queue_name="test", payload={}, max_attempts=1)
        msg.mark_processing()
        msg.mark_failed("Error")
        assert msg.status == MessageStatus.DEAD
        assert not msg.can_retry()


class TestQueueStats:
    """Tests for the QueueStats class."""

    def test_queue_stats_to_dict(self) -> None:
        """QueueStats includes depth calculation."""
        stats = QueueStats(
            queue_name="test",
            pending=5,
            processing=2,
            failed=1,
        )
        d = stats.to_dict()
        assert d["queue_name"] == "test"
        assert d["depth"] == 8  # pending + processing + failed


class TestMessageQueue:
    """Tests for the MessageQueue class."""

    @pytest.fixture
    async def queue(self) -> AsyncGenerator[MessageQueue, None]:
        """Create a message queue for testing."""
        q = MessageQueue()
        await q.initialize()
        yield q
        await q.close()

    @pytest.mark.asyncio
    async def test_publish_message(self, queue: MessageQueue) -> None:
        """Messages can be published to a queue."""
        msg = await queue.publish("test", {"action": "do_thing"})
        assert msg.queue_name == "test"
        assert msg.payload == {"action": "do_thing"}
        assert msg.status == MessageStatus.PENDING

    @pytest.mark.asyncio
    async def test_publish_increments_stats(self, queue: MessageQueue) -> None:
        """Publishing increments queue stats."""
        await queue.publish("test", {})
        await queue.publish("test", {})

        stats = queue.get_stats("test")
        assert stats.pending == 2
        assert stats.total_published == 2

    @pytest.mark.asyncio
    async def test_consume_message(self, queue: MessageQueue) -> None:
        """Messages can be consumed from a queue."""
        await queue.publish("test", {"value": 1})
        await queue.publish("test", {"value": 2})

        consumed: list[Message] = []

        async def handler(msg: Message) -> bool:
            consumed.append(msg)
            return True

        await queue.consume("test", handler)
        assert len(consumed) == 1
        assert consumed[0].payload == {"value": 1}  # FIFO

    @pytest.mark.asyncio
    async def test_consume_ack_removes_message(self, queue: MessageQueue) -> None:
        """Acking a message removes it from the queue."""
        await queue.publish("test", {})

        async def handler(msg: Message) -> bool:
            return True  # Ack

        await queue.consume("test", handler)

        stats = queue.get_stats("test")
        assert stats.completed == 1
        assert stats.pending == 0

    @pytest.mark.asyncio
    async def test_consume_nack_requeues_message(self, queue: MessageQueue) -> None:
        """Nacking a message requeues it for retry."""
        await queue.publish("test", {}, max_attempts=3)

        async def handler(msg: Message) -> bool:
            return False  # Nack

        await queue.consume("test", handler)

        stats = queue.get_stats("test")
        assert stats.pending == 1  # Requeued
        assert stats.failed == 1

    @pytest.mark.asyncio
    async def test_message_moved_to_dead_letter(self, queue: MessageQueue) -> None:
        """Messages exceeding max attempts go to dead letter queue."""
        await queue.publish("test", {}, max_attempts=1)

        async def handler(msg: Message) -> bool:
            return False

        await queue.consume("test", handler)

        stats = queue.get_stats("test")
        assert stats.dead == 1
        assert stats.pending == 0

        dead_letters = await queue.get_dead_letters("test")
        assert len(dead_letters) == 1

    @pytest.mark.asyncio
    async def test_retry_dead_letter(self, queue: MessageQueue) -> None:
        """Dead letter messages can be retried."""
        await queue.publish("test", {}, max_attempts=1)

        async def bad_handler(msg: Message) -> bool:
            return False

        await queue.consume("test", bad_handler)

        dead_letters = await queue.get_dead_letters("test")
        msg_id = dead_letters[0].id

        success = await queue.retry_dead_letter(msg_id)
        assert success

        stats = queue.get_stats("test")
        assert stats.dead == 0
        assert stats.pending == 1

    @pytest.mark.asyncio
    async def test_purge_queue(self, queue: MessageQueue) -> None:
        """Purge removes all messages from a queue."""
        await queue.publish("test", {})
        await queue.publish("test", {})

        count = await queue.purge("test")
        assert count == 2

        stats = queue.get_stats("test")
        assert stats.pending == 0

    @pytest.mark.asyncio
    async def test_list_queues(self, queue: MessageQueue) -> None:
        """list_queues returns all queue names."""
        await queue.publish("queue1", {})
        await queue.publish("queue2", {})

        names = queue.list_queues()
        assert "queue1" in names
        assert "queue2" in names

    @pytest.mark.asyncio
    async def test_fifo_ordering(self, queue: MessageQueue) -> None:
        """Messages are consumed in FIFO order."""
        await queue.publish("test", {"order": 1})
        await queue.publish("test", {"order": 2})
        await queue.publish("test", {"order": 3})

        orders: list[int] = []

        async def handler(msg: Message) -> bool:
            orders.append(msg.payload["order"])
            return True

        await queue.consume("test", handler)
        await queue.consume("test", handler)
        await queue.consume("test", handler)

        assert orders == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_handler_exception_nacks(self, queue: MessageQueue) -> None:
        """Handler exceptions cause automatic nack."""
        await queue.publish("test", {}, max_attempts=2)

        async def bad_handler(msg: Message) -> bool:
            raise ValueError("Handler error")

        await queue.consume("test", bad_handler)

        stats = queue.get_stats("test")
        assert stats.pending == 1  # Requeued for retry

    @pytest.mark.asyncio
    async def test_consume_empty_queue(self, queue: MessageQueue) -> None:
        """Consuming from empty queue does nothing."""
        consumed: list[Message] = []

        async def handler(msg: Message) -> bool:
            consumed.append(msg)
            return True

        await queue.consume("empty", handler)
        assert len(consumed) == 0

    @pytest.mark.asyncio
    async def test_get_pending_count(self, queue: MessageQueue) -> None:
        """get_pending_count returns correct count."""
        assert queue.get_pending_count("test") == 0
        await queue.publish("test", {})
        await queue.publish("test", {})
        assert queue.get_pending_count("test") == 2

    @pytest.mark.asyncio
    async def test_get_dead_letter_count(self, queue: MessageQueue) -> None:
        """get_dead_letter_count returns correct count."""
        assert queue.get_dead_letter_count("test") == 0
        await queue.publish("test", {}, max_attempts=1)

        async def bad_handler(msg: Message) -> bool:
            return False

        await queue.consume("test", bad_handler)
        assert queue.get_dead_letter_count("test") == 1

    @pytest.mark.asyncio
    async def test_retry_nonexistent_dead_letter(self, queue: MessageQueue) -> None:
        """Retrying nonexistent dead letter returns False."""
        success = await queue.retry_dead_letter(uuid4())
        assert not success


class TestEventSerialization:
    """Tests for event serialization edge cases."""

    def test_thunk_started_event_to_dict(self) -> None:
        """ThunkStartedEvent serializes all fields."""
        parent_id = uuid4()
        event = ThunkStartedEvent(
            thunk_id=uuid4(),
            operation="test.op",
            parent_id=parent_id,
        )
        d = event.to_dict()
        assert d["parent_id"] == str(parent_id)

    def test_thunk_completed_event_to_dict(self) -> None:
        """ThunkCompletedEvent serializes all fields."""
        event = ThunkCompletedEvent(
            thunk_id=uuid4(),
            operation="test.op",
            duration_ms=100,
            result_summary="Success",
        )
        d = event.to_dict()
        assert d["result_summary"] == "Success"
        assert d["duration_ms"] == 100

    def test_thunk_failed_event_to_dict(self) -> None:
        """ThunkFailedEvent serializes all fields."""
        event = ThunkFailedEvent(
            thunk_id=uuid4(),
            operation="test.op",
            error_type="ValueError",
            error_message="Bad value",
            duration_ms=50,
        )
        d = event.to_dict()
        assert d["error_type"] == "ValueError"
        assert d["error_message"] == "Bad value"


class TestEventBusPatterns:
    """Tests for event bus pattern matching."""

    @pytest.mark.asyncio
    async def test_match_all_pattern(self) -> None:
        """Star pattern matches all events."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("*", handler)

        await bus.publish(Event(event_type="one"))
        await bus.publish(Event(event_type="two.three"))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_nested_wildcard(self) -> None:
        """Nested wildcards work correctly."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("a.*.c", handler)

        await bus.publish(Event(event_type="a.b.c"))
        await bus.publish(Event(event_type="a.x.c"))
        await bus.publish(Event(event_type="a.b.d"))

        assert len(received) == 2

    def test_get_patterns(self) -> None:
        """get_patterns returns all subscribed patterns."""
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("test.one", handler)
        bus.subscribe("test.two", handler)

        patterns = bus.get_patterns()
        assert "test.one" in patterns
        assert "test.two" in patterns

    def test_unsubscribe_nonexistent(self) -> None:
        """Unsubscribing nonexistent handler returns False."""
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        result = bus.unsubscribe("test.event", handler)
        assert not result
