"""Agent memory system.

Provides persistent memory storage for agents with support for
different memory types, scopes, and query patterns.
"""

from crows_nest.memory.context import ContextMessage, ContextWindow
from crows_nest.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryScope,
    MemoryStats,
    MemoryType,
)
from crows_nest.memory.operations import get_memory_store, set_memory_store
from crows_nest.memory.sqlite import SQLiteMemoryStore
from crows_nest.memory.store import MemoryStore

__all__ = [
    # Context
    "ContextMessage",
    "ContextWindow",
    # Models
    "MemoryEntry",
    "MemoryQuery",
    "MemoryScope",
    "MemoryStats",
    # Stores
    "MemoryStore",
    "MemoryType",
    "SQLiteMemoryStore",
    # Functions
    "get_memory_store",
    "set_memory_store",
]
