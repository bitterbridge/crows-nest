"""Tests for the connection pool."""

import asyncio

import pytest

from crows_nest.persistence.pool import ConnectionPool, PoolConfig


class TestPoolConfig:
    """Tests for PoolConfig validation."""

    def test_default_config(self) -> None:
        """Default config is valid."""
        config = PoolConfig()
        assert config.min_connections == 1
        assert config.max_connections == 5
        assert config.idle_timeout == 300.0

    def test_custom_config(self) -> None:
        """Custom config values work."""
        config = PoolConfig(min_connections=2, max_connections=10, idle_timeout=60.0)
        assert config.min_connections == 2
        assert config.max_connections == 10
        assert config.idle_timeout == 60.0

    def test_min_zero(self) -> None:
        """Zero min_connections is valid."""
        config = PoolConfig(min_connections=0)
        assert config.min_connections == 0

    def test_min_negative(self) -> None:
        """Negative min_connections raises."""
        with pytest.raises(ValueError, match="min_connections must be >= 0"):
            PoolConfig(min_connections=-1)

    def test_max_zero(self) -> None:
        """Zero max_connections raises."""
        with pytest.raises(ValueError, match="max_connections must be >= 1"):
            PoolConfig(max_connections=0)

    def test_min_greater_than_max(self) -> None:
        """min > max raises."""
        with pytest.raises(ValueError, match="min_connections must be <= max_connections"):
            PoolConfig(min_connections=5, max_connections=3)


class TestConnectionPool:
    """Tests for ConnectionPool."""

    @pytest.fixture
    async def pool(self):
        """Create a test pool."""
        pool = await ConnectionPool.create(
            ":memory:",
            PoolConfig(min_connections=1, max_connections=3),
        )
        yield pool
        await pool.close()

    async def test_create_pool(self) -> None:
        """Pool creates successfully."""
        pool = await ConnectionPool.create(":memory:")
        assert pool.size == 1  # min_connections
        assert pool.idle_count == 1
        assert pool.in_use_count == 0
        await pool.close()

    async def test_acquire_release(self, pool: ConnectionPool) -> None:
        """Acquire and release works."""
        assert pool.idle_count == 1
        assert pool.in_use_count == 0

        async with pool.acquire() as conn:
            assert pool.in_use_count == 1
            assert pool.idle_count == 0
            # Verify connection works
            cursor = await conn.execute("SELECT 1")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

        assert pool.in_use_count == 0
        assert pool.idle_count == 1

    async def test_multiple_concurrent_acquires(self, pool: ConnectionPool) -> None:
        """Multiple concurrent acquires work."""
        assert pool.size == 1

        async with pool.acquire() as conn1:
            async with pool.acquire() as conn2:
                assert pool.size == 2  # Created new connection
                assert pool.in_use_count == 2
                # Both connections work
                await conn1.execute("SELECT 1")
                await conn2.execute("SELECT 2")

        assert pool.in_use_count == 0
        assert pool.idle_count == 2  # Both returned to pool

    async def test_max_connections_limit(self) -> None:
        """Pool respects max_connections."""
        pool = await ConnectionPool.create(
            ":memory:",
            PoolConfig(min_connections=0, max_connections=2),
        )

        try:
            conns = []
            # Acquire max connections
            for _ in range(2):
                # Using acquire directly (not as context manager) for test
                conn = await pool._acquire(timeout=1.0)
                conns.append(conn)

            assert pool.size == 2
            assert pool.in_use_count == 2

            # Third acquire should timeout
            with pytest.raises(asyncio.TimeoutError):
                await pool._acquire(timeout=0.1)

            # Release one
            await pool._release(conns[0])
            assert pool.in_use_count == 1

            # Now we can acquire again
            conn3 = await pool._acquire(timeout=1.0)
            assert pool.in_use_count == 2

            # Cleanup
            await pool._release(conns[1])
            await pool._release(conn3)
        finally:
            await pool.close()

    async def test_connection_reuse(self, pool: ConnectionPool) -> None:
        """Connections are reused from pool."""
        # Get initial connection
        async with pool.acquire() as conn1:
            conn1_id = id(conn1)

        # Should get same connection back
        async with pool.acquire() as conn2:
            assert id(conn2) == conn1_id

    async def test_close_pool(self) -> None:
        """Closing pool closes all connections."""
        pool = await ConnectionPool.create(
            ":memory:",
            PoolConfig(min_connections=2, max_connections=3),
        )
        assert pool.size == 2

        await pool.close()
        assert pool.is_closed
        assert pool.size == 0

    async def test_acquire_closed_pool(self) -> None:
        """Acquiring from closed pool raises."""
        pool = await ConnectionPool.create(":memory:")
        await pool.close()

        with pytest.raises(RuntimeError, match="Pool is closed"):
            async with pool.acquire():
                pass

    async def test_context_manager(self) -> None:
        """Pool can be used as context manager."""
        async with await ConnectionPool.create(":memory:") as pool:
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            assert not pool.is_closed

        assert pool.is_closed

    async def test_concurrent_operations(self) -> None:
        """Concurrent operations don't cause issues."""
        pool = await ConnectionPool.create(
            ":memory:",
            PoolConfig(min_connections=1, max_connections=5),
        )

        async def worker(n: int) -> int:
            async with pool.acquire() as conn:
                # Small delay to simulate work
                await asyncio.sleep(0.01)
                cursor = await conn.execute(f"SELECT {n}")
                row = await cursor.fetchone()
                return row[0] if row else 0

        try:
            # Run 10 concurrent workers
            results = await asyncio.gather(*[worker(i) for i in range(10)])
            assert results == list(range(10))
        finally:
            await pool.close()

    async def test_pool_stats(self, pool: ConnectionPool) -> None:
        """Pool provides accurate statistics."""
        # Initial state
        assert pool.size >= 1
        initial_size = pool.size
        assert pool.idle_count == initial_size
        assert pool.in_use_count == 0

        async with pool.acquire():
            assert pool.in_use_count == 1
            assert pool.idle_count == initial_size - 1

        assert pool.in_use_count == 0
        assert pool.idle_count == initial_size
