"""SQLite-based memory store implementation.

Provides persistent memory storage using SQLite with support for
text search, metadata filtering, and optional vector similarity search.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from crows_nest.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryScope,
    MemoryStats,
    MemoryType,
)
from crows_nest.memory.store import MemoryStore

# SQL for creating the memories table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    scope TEXT NOT NULL DEFAULT 'private',
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    embedding TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5,
    expires_at TEXT DEFAULT NULL
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
]


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory storage.

    Provides persistent memory storage using SQLite with support for
    text search and metadata filtering.

    Usage:
        store = SQLiteMemoryStore(path="memories.db")
        await store.initialize()

        entry = await store.remember(
            agent_id=agent.id,
            content="User prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )

        memories = await store.recall(
            agent_id=agent.id,
            query=MemoryQuery(text="preferences"),
        )
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        check_same_thread: bool = False,
    ) -> None:
        """Initialize SQLite memory store.

        Args:
            path: Path to SQLite database file, or ":memory:" for in-memory.
            check_same_thread: SQLite check_same_thread parameter.
        """
        self.path = Path(path) if path != ":memory:" else path
        self.check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the database connection, raising if not initialized."""
        if self._conn is None:
            msg = "Store not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._conn

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.path) if isinstance(self.path, Path) else self.path,
            check_same_thread=self.check_same_thread,
        )
        self._conn.row_factory = sqlite3.Row

        # Create tables and indexes
        self._conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEXES_SQL:
            self._conn.execute(index_sql)
        self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def remember(
        self,
        agent_id: UUID,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        scope: MemoryScope = MemoryScope.PRIVATE,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryEntry:
        """Store a new memory."""
        entry = MemoryEntry(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            scope=scope,
            importance=max(0.0, min(1.0, importance)),  # Clamp to [0, 1]
            metadata=metadata or {},
            embedding=embedding,
        )

        self.conn.execute(
            """
            INSERT INTO memories (
                id, agent_id, memory_type, scope, content, metadata,
                embedding, created_at, accessed_at, access_count,
                importance, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry.id),
                str(entry.agent_id),
                entry.memory_type.value,
                entry.scope.value,
                entry.content,
                json.dumps(entry.metadata),
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.created_at.isoformat(),
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.importance,
                entry.expires_at.isoformat() if entry.expires_at else None,
            ),
        )
        self.conn.commit()
        return entry

    async def recall(
        self,
        agent_id: UUID,
        query: MemoryQuery,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching a query."""
        conditions = ["agent_id = ?"]
        params: list[Any] = [str(agent_id)]

        # Filter by memory type
        if query.memory_type:
            conditions.append("memory_type = ?")
            params.append(query.memory_type.value)

        # Filter by scope
        if query.scope:
            conditions.append("scope = ?")
            params.append(query.scope.value)

        # Filter by minimum importance
        if query.min_importance > 0:
            conditions.append("importance >= ?")
            params.append(query.min_importance)

        # Filter by time range
        if query.time_range:
            conditions.append("created_at >= ? AND created_at <= ?")
            params.extend(
                [
                    query.time_range[0].isoformat(),
                    query.time_range[1].isoformat(),
                ]
            )

        # Filter expired
        if not query.include_expired:
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now(UTC).isoformat())

        # Text search (simple LIKE matching)
        if query.text:
            conditions.append("content LIKE ?")
            params.append(f"%{query.text}%")

        where_clause = " AND ".join(conditions)
        # Query built from hardcoded column names, values use ? placeholders
        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY importance DESC, created_at DESC
            LIMIT ? OFFSET ?
        """  # noqa: S608  # nosec B608 - column names are hardcoded
        params.extend([query.limit, query.offset])

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        # Update access times for recalled memories
        for entry in entries:
            entry.touch()
            self.conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = ? WHERE id = ?",
                (entry.accessed_at.isoformat(), entry.access_count, str(entry.id)),
            )
        self.conn.commit()

        return entries

    async def get(
        self,
        agent_id: UUID,
        memory_id: UUID,
    ) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE id = ? AND agent_id = ?",
            (str(memory_id), str(agent_id)),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        entry = self._row_to_entry(row)
        entry.touch()
        self.conn.execute(
            "UPDATE memories SET accessed_at = ?, access_count = ? WHERE id = ?",
            (entry.accessed_at.isoformat(), entry.access_count, str(entry.id)),
        )
        self.conn.commit()
        return entry

    async def update(
        self,
        agent_id: UUID,
        memory_id: UUID,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        """Update an existing memory."""
        entry = await self.get(agent_id, memory_id)
        if entry is None:
            return None

        updates = []
        params: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
            entry.content = content

        if importance is not None:
            clamped = max(0.0, min(1.0, importance))
            updates.append("importance = ?")
            params.append(clamped)
            entry.importance = clamped

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
            entry.metadata = metadata

        if updates:
            params.append(str(memory_id))
            # Column names are hardcoded, values use ? placeholders
            sql = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"  # noqa: S608  # nosec B608
            self.conn.execute(sql, params)
            self.conn.commit()

        return entry

    async def forget(
        self,
        agent_id: UUID,
        memory_id: UUID,
    ) -> bool:
        """Delete a memory."""
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ? AND agent_id = ?",
            (str(memory_id), str(agent_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    async def forget_all(
        self,
        agent_id: UUID,
        memory_type: MemoryType | None = None,
        older_than: float | None = None,
    ) -> int:
        """Delete multiple memories."""
        conditions = ["agent_id = ?"]
        params: list[Any] = [str(agent_id)]

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        if older_than is not None:
            cutoff = datetime.now(UTC).timestamp() - older_than
            cutoff_dt = datetime.fromtimestamp(cutoff, tz=UTC)
            conditions.append("created_at < ?")
            params.append(cutoff_dt.isoformat())

        where_clause = " AND ".join(conditions)
        # Column names are hardcoded, values use ? placeholders
        cursor = self.conn.execute(
            f"DELETE FROM memories WHERE {where_clause}",  # noqa: S608  # nosec B608
            params,
        )
        self.conn.commit()
        return cursor.rowcount

    async def decay(
        self,
        agent_id: UUID,
        threshold: float = 0.1,
        half_life_seconds: float = 86400.0,
    ) -> int:
        """Remove memories with decay score below threshold."""
        # Need to calculate decay scores in Python since SQLite
        # doesn't have the math functions we need
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ?",
            (str(agent_id),),
        )
        rows = cursor.fetchall()

        to_delete = []
        now = datetime.now(UTC)

        for row in rows:
            accessed_at = datetime.fromisoformat(row["accessed_at"])
            staleness = (now - accessed_at).total_seconds()
            importance = row["importance"]

            # Match the decay_score calculation from MemoryEntry
            effective_half_life = half_life_seconds * (1 + importance)
            decay = math.exp(-0.693 * staleness / effective_half_life)
            score = decay * importance

            if score < threshold:
                to_delete.append(row["id"])

        if to_delete:
            # Placeholders are just "?" repeated, safe for parameterized query
            placeholders = ",".join("?" * len(to_delete))
            self.conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",  # noqa: S608  # nosec B608
                to_delete,
            )
            self.conn.commit()

        return len(to_delete)

    async def get_stats(self, agent_id: UUID) -> MemoryStats:
        """Get statistics about an agent's memories."""
        agent_str = str(agent_id)

        # Total count
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_id = ?",
            (agent_str,),
        )
        total_count = cursor.fetchone()[0]

        # Count by type
        cursor = self.conn.execute(
            """
            SELECT memory_type, COUNT(*) as count
            FROM memories WHERE agent_id = ?
            GROUP BY memory_type
            """,
            (agent_str,),
        )
        by_type = {row["memory_type"]: row["count"] for row in cursor.fetchall()}

        # Count by scope
        cursor = self.conn.execute(
            """
            SELECT scope, COUNT(*) as count
            FROM memories WHERE agent_id = ?
            GROUP BY scope
            """,
            (agent_str,),
        )
        by_scope = {row["scope"]: row["count"] for row in cursor.fetchall()}

        # Average importance
        cursor = self.conn.execute(
            "SELECT AVG(importance) FROM memories WHERE agent_id = ?",
            (agent_str,),
        )
        avg_importance = cursor.fetchone()[0] or 0.0

        # Oldest and newest
        cursor = self.conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM memories WHERE agent_id = ?",
            (agent_str,),
        )
        row = cursor.fetchone()
        oldest = datetime.fromisoformat(row[0]) if row[0] else None
        newest = datetime.fromisoformat(row[1]) if row[1] else None

        # Approximate size (content + metadata length)
        cursor = self.conn.execute(
            """
            SELECT SUM(LENGTH(content) + LENGTH(metadata))
            FROM memories WHERE agent_id = ?
            """,
            (agent_str,),
        )
        total_size = cursor.fetchone()[0] or 0

        return MemoryStats(
            agent_id=agent_id,
            total_count=total_count,
            by_type=by_type,
            by_scope=by_scope,
            avg_importance=avg_importance,
            oldest=oldest,
            newest=newest,
            total_size_bytes=total_size,
        )

    async def export_memories(
        self,
        agent_id: UUID,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Export all memories for an agent."""
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at",
            (str(agent_id),),
        )
        rows = cursor.fetchall()

        result = []
        for row in rows:
            entry = self._row_to_entry(row)
            data = entry.to_dict()
            if not include_embeddings:
                data.pop("embedding", None)
            result.append(data)

        return result

    async def import_memories(
        self,
        agent_id: UUID,
        memories: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> int:
        """Import memories for an agent."""
        imported = 0

        for data in memories:
            # Override agent_id to the importing agent
            data["agent_id"] = str(agent_id)
            entry = MemoryEntry.from_dict(data)

            try:
                if overwrite:
                    # Delete existing if present
                    self.conn.execute(
                        "DELETE FROM memories WHERE id = ?",
                        (str(entry.id),),
                    )

                self.conn.execute(
                    """
                    INSERT INTO memories (
                        id, agent_id, memory_type, scope, content, metadata,
                        embedding, created_at, accessed_at, access_count,
                        importance, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(entry.id),
                        str(entry.agent_id),
                        entry.memory_type.value,
                        entry.scope.value,
                        entry.content,
                        json.dumps(entry.metadata),
                        json.dumps(entry.embedding) if entry.embedding else None,
                        entry.created_at.isoformat(),
                        entry.accessed_at.isoformat(),
                        entry.access_count,
                        entry.importance,
                        entry.expires_at.isoformat() if entry.expires_at else None,
                    ),
                )
                imported += 1
            except sqlite3.IntegrityError:
                # Skip duplicates if not overwriting
                continue

        self.conn.commit()
        return imported

    async def search_semantic(
        self,
        _agent_id: UUID,
        _embedding: list[float],
        _limit: int = 10,
        _threshold: float = 0.7,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories by vector similarity.

        Note: This basic SQLite implementation does not support vector search.
        Override with a specialized backend (e.g., sqlite-vss) for vector support.
        """
        return []

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])

        return MemoryEntry(
            id=UUID(row["id"]),
            agent_id=UUID(row["agent_id"]),
            memory_type=MemoryType(row["memory_type"]),
            scope=MemoryScope(row["scope"]),
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            access_count=row["access_count"],
            importance=row["importance"],
            expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
        )
