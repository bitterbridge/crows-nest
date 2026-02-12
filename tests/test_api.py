"""Tests for the FastAPI server."""

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
    # Mock the lifespan context
    mock_backend = AsyncMock()
    mock_backend.list_thunks = AsyncMock(return_value=[])
    mock_backend.close = AsyncMock()

    mock_runtime = AsyncMock()

    with (
        patch("crows_nest.api.server._backend", mock_backend),
        patch("crows_nest.api.server._runtime", mock_runtime),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """GET /health returns 200 with status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    async def test_ready_returns_200_when_connected(self, client: AsyncClient) -> None:
        """GET /ready returns 200 when database is connected."""
        response = await client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["database"] == "connected"


class TestThunkEndpoints:
    """Tests for thunk-related endpoints."""

    async def test_submit_thunk_unknown_operation(self, client: AsyncClient) -> None:
        """POST /thunks rejects unknown operations."""
        response = await client.post(
            "/thunks",
            json={
                "operation": "unknown.operation",
                "inputs": {},
                "capabilities": [],
                "tags": {},
            },
        )

        assert response.status_code == 400
        assert "Unknown operation" in response.json()["detail"]

    async def test_get_thunk_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /thunks/{id} rejects invalid UUIDs."""
        response = await client.get("/thunks/not-a-uuid")

        assert response.status_code == 400
        assert "Invalid thunk ID" in response.json()["detail"]

    async def test_get_thunk_not_found(self, client: AsyncClient) -> None:
        """GET /thunks/{id} returns 404 for unknown thunk."""
        with patch("crows_nest.api.server._backend") as mock_backend:
            mock_backend.get_thunk = AsyncMock(return_value=None)

            response = await client.get("/thunks/00000000-0000-0000-0000-000000000001")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_get_thunk_status_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /thunks/{id}/status rejects invalid UUIDs."""
        response = await client.get("/thunks/not-a-uuid/status")

        assert response.status_code == 400
        assert "Invalid thunk ID" in response.json()["detail"]

    async def test_get_thunk_status_not_found(self, client: AsyncClient) -> None:
        """GET /thunks/{id}/status returns 404 for unknown thunk."""
        with patch("crows_nest.api.server._backend") as mock_backend:
            mock_backend.get_result = AsyncMock(return_value=None)
            mock_backend.get_thunk = AsyncMock(return_value=None)

            response = await client.get("/thunks/00000000-0000-0000-0000-000000000001/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_get_thunk_result_invalid_uuid(self, client: AsyncClient) -> None:
        """GET /thunks/{id}/result rejects invalid UUIDs."""
        response = await client.get("/thunks/not-a-uuid/result")

        assert response.status_code == 400
        assert "Invalid thunk ID" in response.json()["detail"]

    async def test_get_thunk_result_not_found(self, client: AsyncClient) -> None:
        """GET /thunks/{id}/result returns 404 for unknown thunk."""
        with patch("crows_nest.api.server._backend") as mock_backend:
            mock_backend.get_result = AsyncMock(return_value=None)
            mock_backend.get_thunk = AsyncMock(return_value=None)

            response = await client.get("/thunks/00000000-0000-0000-0000-000000000001/result")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestOperationsEndpoint:
    """Tests for the operations endpoint."""

    async def test_list_operations(self, client: AsyncClient) -> None:
        """GET /ops returns list of operations."""
        response = await client.get("/ops")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Operations are registered, should have at least agent operations
        if len(data) > 0:
            op = data[0]
            assert "name" in op
            assert "description" in op
            assert "required_capabilities" in op
            assert "parameters" in op


class TestAskEndpoint:
    """Tests for the /ask endpoint."""

    async def test_ask_unknown_task_returns_error(self, client: AsyncClient) -> None:
        """POST /ask handles unknown tasks gracefully with error message."""
        response = await client.post(
            "/ask",
            json={"query": "do something impossible that no pattern matches"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["error"] is not None
        assert "Cannot create execution plan" in data["error"]
