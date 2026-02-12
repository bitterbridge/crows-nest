"""Simple in-memory rate limiting for the API."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_second: int = 10
    enabled: bool = True


@dataclass
class ClientState:
    """Tracks rate limit state for a client."""

    requests: list[float] = field(default_factory=list)

    def clean_old_requests(self, window_seconds: float) -> None:
        """Remove requests older than the window."""
        cutoff = time.time() - window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def add_request(self) -> None:
        """Record a new request."""
        self.requests.append(time.time())

    def count_in_window(self, window_seconds: float) -> int:
        """Count requests within the window."""
        cutoff = time.time() - window_seconds
        return sum(1 for t in self.requests if t > cutoff)


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter with config."""
        self.config = config or RateLimitConfig()
        self._clients: dict[str, ClientState] = defaultdict(ClientState)
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # Clean up old entries every minute

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind a proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _maybe_cleanup(self) -> None:
        """Periodically clean up old client entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        # Remove clients with no recent requests
        empty_clients = [
            key for key, state in self._clients.items() if state.count_in_window(60.0) == 0
        ]
        for key in empty_clients:
            del self._clients[key]

    def check(self, request: Request) -> None:
        """Check rate limit and raise HTTPException if exceeded."""
        if not self.config.enabled:
            return

        self._maybe_cleanup()

        client_key = self._get_client_key(request)
        state = self._clients[client_key]

        # Clean old requests
        state.clean_old_requests(60.0)

        # Check per-second limit
        per_second = state.count_in_window(1.0)
        if per_second >= self.config.requests_per_second:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down.",
                headers={"Retry-After": "1"},
            )

        # Check per-minute limit
        per_minute = state.count_in_window(60.0)
        if per_minute >= self.config.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"},
            )

        # Record request
        state.add_request()


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter  # noqa: PLW0603
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def configure_rate_limiter(config: RateLimitConfig) -> None:
    """Configure the global rate limiter."""
    global _rate_limiter  # noqa: PLW0603
    _rate_limiter = RateLimiter(config)


def rate_limit(func: Callable) -> Callable:  # type: ignore[type-arg]
    """Decorator to apply rate limiting to an endpoint."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        # Find Request in args or kwargs
        request = kwargs.get("request")
        if request is None:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

        if request is not None:
            get_rate_limiter().check(request)

        return await func(*args, **kwargs)

    return wrapper
