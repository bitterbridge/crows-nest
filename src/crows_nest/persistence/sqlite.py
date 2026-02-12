"""SQLite persistence backend.

Implements the PersistenceBackend protocol using SQLite with aiosqlite
for async support.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - needed at runtime
from uuid import UUID  # noqa: TC003 - needed at runtime

import aiosqlite

from crows_nest.core.thunk import Thunk, ThunkResult, ThunkStatus

# Schema version for future migrations
SCHEMA_VERSION = 1

SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Thunks table
CREATE TABLE IF NOT EXISTS thunks (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_thunks_operation ON thunks(operation);
CREATE INDEX IF NOT EXISTS idx_thunks_created_at ON thunks(created_at);
-- Composite index for filtering by operation and ordering by created_at
CREATE INDEX IF NOT EXISTS idx_thunks_operation_created ON thunks(operation, created_at DESC);

-- Results table
CREATE TABLE IF NOT EXISTS results (
    thunk_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    value_json TEXT,
    error_json TEXT,
    forced_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    FOREIGN KEY (thunk_id) REFERENCES thunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
CREATE INDEX IF NOT EXISTS idx_results_forced_at ON results(forced_at);
-- Composite index for filtering by status and ordering by forced_at
CREATE INDEX IF NOT EXISTS idx_results_status_forced ON results(status, forced_at DESC);
"""


class SQLiteBackend:
    """SQLite implementation of PersistenceBackend.

    Uses aiosqlite for async operations. Data is stored as JSON for
    flexibility and schema evolution.

    Example:
        backend = await SQLiteBackend.create("crows_nest.db")
        await backend.save_thunk(thunk)
        loaded = await backend.get_thunk(thunk.id)
        await backend.close()
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize with an open database connection.

        Use SQLiteBackend.create() instead of calling this directly.
        """
        self._db = db

    @classmethod
    async def create(
        cls,
        path: str | Path,
        *,
        create_tables: bool = True,
        enable_wal: bool = True,
    ) -> SQLiteBackend:
        """Create a new SQLite backend.

        Args:
            path: Path to the database file. Use ":memory:" for in-memory.
            create_tables: If True, create tables if they don't exist.
            enable_wal: If True, enable WAL mode for better concurrent access.
                Ignored for in-memory databases.

        Returns:
            Initialized SQLiteBackend.
        """
        db = await aiosqlite.connect(path)

        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")

        # Performance optimizations
        await db.execute("PRAGMA synchronous = NORMAL")  # Faster writes, still durable
        await db.execute("PRAGMA cache_size = -64000")  # 64MB cache
        await db.execute("PRAGMA temp_store = MEMORY")  # Temp tables in memory

        # Enable WAL mode for better concurrent access (file-based only)
        if enable_wal and str(path) != ":memory:":
            await db.execute("PRAGMA journal_mode = WAL")

        if create_tables:
            await db.executescript(SCHEMA)

            # Set or verify schema version
            cursor = await db.execute("SELECT version FROM schema_version LIMIT 1")
            row = await cursor.fetchone()
            if row is None:
                await db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            await db.commit()

        return cls(db)

    async def save_thunk(self, thunk: Thunk) -> None:
        """Save a thunk to the database."""
        await self._db.execute(
            """
            INSERT OR REPLACE INTO thunks (id, operation, inputs_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(thunk.id),
                thunk.operation,
                json.dumps(thunk.to_dict()["inputs"]),
                json.dumps(thunk.metadata.to_dict()),
                thunk.metadata.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def get_thunk(self, thunk_id: UUID) -> Thunk | None:
        """Retrieve a thunk by ID."""
        cursor = await self._db.execute(
            "SELECT id, operation, inputs_json, metadata_json FROM thunks WHERE id = ?",
            (str(thunk_id),),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return Thunk.from_dict(
            {
                "id": row[0],
                "operation": row[1],
                "inputs": json.loads(row[2]),
                "metadata": json.loads(row[3]),
            }
        )

    async def save_result(self, result: ThunkResult) -> None:
        """Save a thunk result to the database."""
        await self._db.execute(
            """
            INSERT OR REPLACE INTO results
            (thunk_id, status, value_json, error_json, forced_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(result.thunk_id),
                result.status.value,
                json.dumps(result.value) if result.value is not None else None,
                json.dumps(result.error.to_dict()) if result.error else None,
                result.forced_at.isoformat(),
                result.duration_ms,
            ),
        )
        await self._db.commit()

    async def get_result(self, thunk_id: UUID) -> ThunkResult | None:
        """Retrieve a thunk result by thunk ID."""
        cursor = await self._db.execute(
            """
            SELECT thunk_id, status, value_json, error_json, forced_at, duration_ms
            FROM results WHERE thunk_id = ?
            """,
            (str(thunk_id),),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return ThunkResult.from_dict(
            {
                "thunk_id": row[0],
                "status": row[1],
                "value": json.loads(row[2]) if row[2] else None,
                "error": json.loads(row[3]) if row[3] else None,
                "forced_at": row[4],
                "duration_ms": row[5],
            }
        )

    async def list_thunks(
        self,
        *,
        status: ThunkStatus | None = None,
        operation: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thunk]:
        """List thunks with optional filtering."""
        query = "SELECT t.id, t.operation, t.inputs_json, t.metadata_json FROM thunks t"
        params: list[str | int] = []
        conditions: list[str] = []

        if status is not None:
            query += " LEFT JOIN results r ON t.id = r.thunk_id"
            conditions.append("r.status = ?")
            params.append(status.value)

        if operation is not None:
            conditions.append("t.operation = ?")
            params.append(operation)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY t.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()

        thunks: list[Thunk] = []
        for row in rows:
            thunks.append(
                Thunk.from_dict(
                    {
                        "id": row[0],
                        "operation": row[1],
                        "inputs": json.loads(row[2]),
                        "metadata": json.loads(row[3]),
                    }
                )
            )

        return thunks

    async def delete_thunk(self, thunk_id: UUID) -> bool:
        """Delete a thunk and its result."""
        # Delete result first (foreign key)
        await self._db.execute("DELETE FROM results WHERE thunk_id = ?", (str(thunk_id),))

        cursor = await self._db.execute(
            "DELETE FROM thunks WHERE id = ?",
            (str(thunk_id),),
        )
        await self._db.commit()

        return bool(cursor.rowcount and cursor.rowcount > 0)

    async def save_thunks_batch(self, thunks: list[Thunk]) -> None:
        """Save multiple thunks in a single transaction.

        More efficient than calling save_thunk() multiple times.
        """
        if not thunks:
            return

        await self._db.executemany(
            """
            INSERT OR REPLACE INTO thunks (id, operation, inputs_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(thunk.id),
                    thunk.operation,
                    json.dumps(thunk.to_dict()["inputs"]),
                    json.dumps(thunk.metadata.to_dict()),
                    thunk.metadata.created_at.isoformat(),
                )
                for thunk in thunks
            ],
        )
        await self._db.commit()

    async def save_results_batch(self, results: list[ThunkResult]) -> None:
        """Save multiple results in a single transaction.

        More efficient than calling save_result() multiple times.
        """
        if not results:
            return

        await self._db.executemany(
            """
            INSERT OR REPLACE INTO results
            (thunk_id, status, value_json, error_json, forced_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(result.thunk_id),
                    result.status.value,
                    json.dumps(result.value) if result.value is not None else None,
                    json.dumps(result.error.to_dict()) if result.error else None,
                    result.forced_at.isoformat(),
                    result.duration_ms,
                )
                for result in results
            ],
        )
        await self._db.commit()

    async def count_thunks(
        self,
        *,
        status: ThunkStatus | None = None,
        operation: str | None = None,
    ) -> int:
        """Count thunks with optional filtering.

        More efficient than list_thunks() when only the count is needed.
        """
        query = "SELECT COUNT(*) FROM thunks t"
        params: list[str] = []
        conditions: list[str] = []

        if status is not None:
            query += " LEFT JOIN results r ON t.id = r.thunk_id"
            conditions.append("r.status = ?")
            params.append(status.value)

        if operation is not None:
            conditions.append("t.operation = ?")
            params.append(operation)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = await self._db.execute(query, params)
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_thunks_by_ids(self, thunk_ids: list[UUID]) -> dict[UUID, Thunk]:
        """Retrieve multiple thunks by their IDs in a single query.

        More efficient than calling get_thunk() multiple times.
        """
        if not thunk_ids:
            return {}

        placeholders = ",".join("?" * len(thunk_ids))
        query = (
            f"SELECT id, operation, inputs_json, metadata_json "  # nosec B608
            f"FROM thunks WHERE id IN ({placeholders})"
        )
        cursor = await self._db.execute(query, [str(tid) for tid in thunk_ids])
        rows = await cursor.fetchall()

        result: dict[UUID, Thunk] = {}
        for row in rows:
            thunk = Thunk.from_dict(
                {
                    "id": row[0],
                    "operation": row[1],
                    "inputs": json.loads(row[2]),
                    "metadata": json.loads(row[3]),
                }
            )
            result[thunk.id] = thunk

        return result

    async def get_results_by_ids(self, thunk_ids: list[UUID]) -> dict[UUID, ThunkResult]:
        """Retrieve multiple results by their thunk IDs in a single query.

        More efficient than calling get_result() multiple times.
        """
        if not thunk_ids:
            return {}

        placeholders = ",".join("?" * len(thunk_ids))
        query = (
            f"SELECT thunk_id, status, value_json, error_json, forced_at, duration_ms "  # nosec B608
            f"FROM results WHERE thunk_id IN ({placeholders})"
        )
        cursor = await self._db.execute(query, [str(tid) for tid in thunk_ids])
        rows = await cursor.fetchall()

        result: dict[UUID, ThunkResult] = {}
        for row in rows:
            tr = ThunkResult.from_dict(
                {
                    "thunk_id": row[0],
                    "status": row[1],
                    "value": json.loads(row[2]) if row[2] else None,
                    "error": json.loads(row[3]) if row[3] else None,
                    "forced_at": row[4],
                    "duration_ms": row[5],
                }
            )
            result[tr.thunk_id] = tr

        return result

    async def close(self) -> None:
        """Close the database connection."""
        await self._db.close()

    # Context manager support

    async def __aenter__(self) -> SQLiteBackend:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context, closing the database."""
        await self.close()
