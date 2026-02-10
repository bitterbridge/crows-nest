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
    ) -> SQLiteBackend:
        """Create a new SQLite backend.

        Args:
            path: Path to the database file. Use ":memory:" for in-memory.
            create_tables: If True, create tables if they don't exist.

        Returns:
            Initialized SQLiteBackend.
        """
        db = await aiosqlite.connect(path)

        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")

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
