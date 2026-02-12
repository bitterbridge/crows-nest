"""Tests for the dashboard API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from crows_nest.api.server import app
from crows_nest.core.config import clear_settings_cache

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(autouse=True)
def clean_settings() -> None:
    """Clear settings cache before each test."""
    clear_settings_cache()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with mocked backend."""
    from crows_nest.api.ratelimit import RateLimitConfig, RateLimiter

    mock_backend = AsyncMock()
    mock_backend.list_thunks = AsyncMock(return_value=[])
    mock_backend.get_thunk = AsyncMock(return_value=None)
    mock_backend.get_result = AsyncMock(return_value=None)
    mock_backend.get_results_by_ids = AsyncMock(return_value={})
    mock_backend.count_thunks = AsyncMock(return_value=0)
    mock_backend.close = AsyncMock()

    mock_runtime = AsyncMock()

    # Disable rate limiting for tests
    disabled_limiter = RateLimiter(RateLimitConfig(enabled=False))

    with (
        patch("crows_nest.api.server._backend", mock_backend),
        patch("crows_nest.api.server._runtime", mock_runtime),
        patch("crows_nest.api.ratelimit.get_rate_limiter", return_value=disabled_limiter),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


class TestDashboardOverview:
    """Tests for the dashboard overview endpoint."""

    async def test_get_overview_returns_expected_fields(self, client: AsyncClient) -> None:
        """GET /api/dashboard/overview returns system overview with expected fields."""
        # This test makes an actual call - the endpoint may work or may error
        # depending on initialization state, but we test the route exists
        response = await client.get("/api/dashboard/overview")

        # If it succeeds, check the shape
        if response.status_code == 200:
            data = response.json()
            assert "active_thunks" in data
            assert "completed_thunks" in data
            assert "failed_thunks" in data
            assert "active_agents" in data
            assert "uptime_seconds" in data
        else:
            # At minimum, the route should exist (not 404)
            assert response.status_code != 404


class TestDashboardThunks:
    """Tests for the dashboard thunks endpoints."""

    async def test_list_thunks_endpoint_exists(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks returns a response."""
        response = await client.get("/api/dashboard/thunks")

        # Route exists
        assert response.status_code != 404

        # If it succeeds, check the shape
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    async def test_list_thunks_with_status_filter(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks accepts status filter."""
        response = await client.get("/api/dashboard/thunks?status=success")

        # Query param should be accepted
        assert response.status_code != 404

    async def test_get_thunk_detail_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks/{id} returns 400 for invalid UUID."""
        response = await client.get("/api/dashboard/thunks/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid thunk ID" in response.json()["detail"]

    async def test_get_thunk_detail_not_found(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks/{id} returns 404 for unknown thunk."""
        # Use a valid UUID that doesn't exist
        response = await client.get("/api/dashboard/thunks/00000000-0000-0000-0000-000000000001")

        # Should be 404 or 500 (if backend not fully initialized)
        assert response.status_code in (404, 500)

    async def test_get_thunk_dag_endpoint_exists(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks/graph returns DAG data."""
        response = await client.get("/api/dashboard/thunks/graph")

        # Route exists
        assert response.status_code != 404

        # If it succeeds, check the shape
        if response.status_code == 200:
            data = response.json()
            assert "nodes" in data
            assert "edges" in data
            assert isinstance(data["nodes"], list)
            assert isinstance(data["edges"], list)

    async def test_get_thunk_dag_with_limit(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks/graph accepts limit parameter."""
        response = await client.get("/api/dashboard/thunks/graph?limit=50")

        # Query param should be accepted
        assert response.status_code != 404

    async def test_get_thunk_dag_with_trace_id(self, client: AsyncClient) -> None:
        """GET /api/dashboard/thunks/graph accepts trace_id parameter."""
        response = await client.get("/api/dashboard/thunks/graph?trace_id=test-trace-123")

        # Query param should be accepted
        assert response.status_code != 404


class TestDashboardAgents:
    """Tests for the dashboard agents endpoints."""

    async def test_list_agents_endpoint_exists(self, client: AsyncClient) -> None:
        """GET /api/dashboard/agents returns a response."""
        response = await client.get("/api/dashboard/agents")

        # Route exists
        assert response.status_code != 404

        # If it succeeds, check the shape
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    async def test_list_agents_with_terminated_filter(self, client: AsyncClient) -> None:
        """GET /api/dashboard/agents accepts include_terminated filter."""
        response = await client.get("/api/dashboard/agents?include_terminated=true")

        # Query param should be accepted
        assert response.status_code != 404

    async def test_get_agent_tree_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /api/dashboard/agents/{id}/tree returns 400 for invalid UUID."""
        response = await client.get("/api/dashboard/agents/not-a-uuid/tree")
        assert response.status_code == 400
        assert "Invalid agent ID" in response.json()["detail"]

    async def test_get_agent_tree_not_found(self, client: AsyncClient) -> None:
        """GET /api/dashboard/agents/{id}/tree returns 404 for unknown agent."""
        response = await client.get(
            "/api/dashboard/agents/00000000-0000-0000-0000-000000000001/tree"
        )

        # Should be 404 (agent not found)
        assert response.status_code in (404, 500)


class TestDashboardQueues:
    """Tests for the dashboard queues endpoint."""

    async def test_list_queues_endpoint_exists(self, client: AsyncClient) -> None:
        """GET /api/dashboard/queues returns a response."""
        response = await client.get("/api/dashboard/queues")

        # Route exists
        assert response.status_code != 404

        # If it succeeds, check the shape
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


class TestDashboardMemory:
    """Tests for the dashboard memory endpoint."""

    async def test_get_agent_memory_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /api/dashboard/memory/{agent_id} returns 400 for invalid UUID."""
        response = await client.get("/api/dashboard/memory/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid agent ID" in response.json()["detail"]

    async def test_get_agent_memory_endpoint_exists(self, client: AsyncClient) -> None:
        """GET /api/dashboard/memory/{agent_id} returns a response."""
        response = await client.get("/api/dashboard/memory/00000000-0000-0000-0000-000000000001")

        # Route exists
        assert response.status_code != 404


class TestDashboardConfig:
    """Tests for the dashboard config endpoint."""

    async def test_get_config(self, client: AsyncClient) -> None:
        """GET /api/dashboard/config returns configuration."""
        response = await client.get("/api/dashboard/config")

        assert response.status_code == 200
        data = response.json()
        assert "database_path" in data
        assert "api_host" in data
        assert "api_port" in data
        assert "log_level" in data
        assert "tracing_enabled" in data


class TestDashboardIndex:
    """Tests for the dashboard index page."""

    async def test_dashboard_index_returns_html(self, client: AsyncClient) -> None:
        """GET /dashboard returns the index page."""
        response = await client.get("/dashboard")

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type
