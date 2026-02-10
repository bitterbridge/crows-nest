"""Shared pytest fixtures and configuration."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"
