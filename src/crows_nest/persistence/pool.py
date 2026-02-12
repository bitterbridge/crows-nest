"""Connection pooling for SQLite.

Provides a simple connection pool for SQLite databases. While SQLite doesn't
benefit from connection pooling as much as client-server databases, pooling
can reduce connection overhead and improve concurrent read performance when
using WAL mode.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@dataclass
class PoolConfig:
    """Configuration for the connection pool.

    Attributes:
        min_connections: Minimum connections to keep open.
        max_connections: Maximum connections allowed.
        idle_timeout: Seconds before idle connections are closed (0 = never).
    """

    min_connections: int = 1
    max_connections: int = 5
    idle_timeout: float = 300.0  # 5 minutes

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_connections < 0:
            msg = "min_connections must be >= 0"
            raise ValueError(msg)
        if self.max_connections < 1:
            msg = "max_connections must be >= 1"
            raise ValueError(msg)
        if self.min_connections > self.max_connections:
            msg = "min_connections must be <= max_connections"
            raise ValueError(msg)


@dataclass
class ConnectionPool:
    """A simple connection pool for SQLite databases.

    The pool maintains a set of idle connections and creates new ones
    on demand up to max_connections. When a connection is released,
    it's returned to the idle pool for reuse.

    Example:
        pool = await ConnectionPool.create("data.db", PoolConfig(max_connections=3))

        # Use a connection
        async with pool.acquire() as conn:
            await conn.execute("SELECT * FROM users")

        # Cleanup
        await pool.close()
    """

    path: str | Path
    config: PoolConfig
    _pragmas: list[str] = field(default_factory=list)
    _idle: asyncio.Queue[aiosqlite.Connection] = field(init=False)
    _in_use: set[aiosqlite.Connection] = field(default_factory=set)
    _total: int = field(default=0)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _closed: bool = field(default=False)

    @classmethod
    async def create(
        cls,
        path: str | Path,
        config: PoolConfig | None = None,
        *,
        enable_wal: bool = True,
    ) -> ConnectionPool:
        """Create a new connection pool.

        Args:
            path: Path to the database file.
            config: Pool configuration (uses defaults if None).
            enable_wal: Enable WAL mode for better concurrent access.

        Returns:
            Initialized connection pool.
        """
        if config is None:
            config = PoolConfig()

        pragmas = [
            "PRAGMA foreign_keys = ON",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA cache_size = -64000",
            "PRAGMA temp_store = MEMORY",
        ]
        if enable_wal and str(path) != ":memory:":
            pragmas.append("PRAGMA journal_mode = WAL")

        pool = cls(path=path, config=config, _pragmas=pragmas)
        pool._idle = asyncio.Queue(maxsize=config.max_connections)

        # Pre-create minimum connections
        for _ in range(config.min_connections):
            conn = await pool._create_connection()
            await pool._idle.put(conn)
            pool._total += 1

        return pool

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection with pragmas applied."""
        conn = await aiosqlite.connect(self.path)
        for pragma in self._pragmas:
            await conn.execute(pragma)
        return conn

    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool.

        Args:
            timeout: Maximum seconds to wait for a connection.

        Yields:
            A database connection.

        Raises:
            asyncio.TimeoutError: If no connection is available within timeout.
            RuntimeError: If the pool is closed.
        """
        if self._closed:
            msg = "Pool is closed"
            raise RuntimeError(msg)

        conn = await self._acquire(timeout)
        try:
            yield conn
        finally:
            await self._release(conn)

    async def _acquire(self, timeout: float) -> aiosqlite.Connection:
        """Internal method to acquire a connection."""
        async with self._lock:
            # Try to get an idle connection
            try:
                conn = self._idle.get_nowait()
                self._in_use.add(conn)
                return conn
            except asyncio.QueueEmpty:
                pass

            # Create a new connection if under limit
            if self._total < self.config.max_connections:
                conn = await self._create_connection()
                self._total += 1
                self._in_use.add(conn)
                return conn

        # Wait for an idle connection (release lock during wait)
        try:
            conn = await asyncio.wait_for(self._idle.get(), timeout=timeout)
            async with self._lock:
                self._in_use.add(conn)
            return conn
        except TimeoutError:
            msg = f"Timeout waiting for connection (max={self.config.max_connections})"
            raise TimeoutError(msg) from None

    async def _release(self, conn: aiosqlite.Connection) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.discard(conn)

                if self._closed:
                    await conn.close()
                    self._total -= 1
                else:
                    try:
                        self._idle.put_nowait(conn)
                    except asyncio.QueueFull:
                        # Pool is full, close the connection
                        await conn.close()
                        self._total -= 1

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            self._closed = True

            # Close idle connections
            while not self._idle.empty():
                try:
                    conn = self._idle.get_nowait()
                    await conn.close()
                    self._total -= 1
                except asyncio.QueueEmpty:
                    break

            # Close in-use connections
            for conn in list(self._in_use):
                await conn.close()
                self._total -= 1
            self._in_use.clear()

    @property
    def size(self) -> int:
        """Total number of connections (idle + in use)."""
        return self._total

    @property
    def idle_count(self) -> int:
        """Number of idle connections."""
        return self._idle.qsize()

    @property
    def in_use_count(self) -> int:
        """Number of connections currently in use."""
        return len(self._in_use)

    @property
    def is_closed(self) -> bool:
        """Whether the pool has been closed."""
        return self._closed

    # Context manager support

    async def __aenter__(self) -> ConnectionPool:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context, closing the pool."""
        await self.close()
