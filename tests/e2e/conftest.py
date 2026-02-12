"""Fixtures for Playwright e2e tests."""

from __future__ import annotations

import asyncio
import multiprocessing
import socket
import time
from typing import TYPE_CHECKING

import pytest
import uvicorn

if TYPE_CHECKING:
    from collections.abc import Generator

    from playwright.sync_api import Page

# Mark all tests in this directory as browser tests
pytestmark = pytest.mark.browser


def get_free_port() -> int:
    """Get a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def run_server(port: int) -> None:
    """Run the server in a separate process."""
    import os

    # Use in-memory database for tests
    os.environ["CROWS_NEST_DATABASE__PATH"] = ":memory:"

    # Disable rate limiting for e2e tests
    from crows_nest.api.ratelimit import RateLimitConfig, configure_rate_limiter

    configure_rate_limiter(RateLimitConfig(enabled=False))

    from crows_nest.api.server import app

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Run the server
    asyncio.run(server.serve())


@pytest.fixture(scope="session")
def server_url() -> Generator[str, None, None]:
    """Start the server and return its URL."""
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"

    # Start server in a separate process
    proc = multiprocessing.Process(target=run_server, args=(port,))
    proc.start()

    # Wait for server to be ready
    max_wait = 10.0
    start = time.time()
    while time.time() - start < max_wait:
        try:
            import urllib.request

            with urllib.request.urlopen(f"{url}/health", timeout=1):  # noqa: S310
                break
        except Exception:
            time.sleep(0.2)
    else:
        proc.terminate()
        proc.join()
        pytest.fail("Server did not start in time")

    yield url

    # Shutdown server
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()


@pytest.fixture
def dashboard_page(page: Page, server_url: str) -> Page:
    """Navigate to the dashboard and wait for it to load."""
    page.goto(f"{server_url}/dashboard")
    # Wait for the sidebar to be visible
    page.wait_for_selector(".sidebar", state="visible")
    return page
