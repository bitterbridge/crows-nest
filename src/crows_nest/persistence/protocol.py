"""Persistence backend protocol.

Defines the interface for storage backends. Implementations must be async
to support non-blocking I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID  # noqa: TC003 - needed at runtime for Protocol

if TYPE_CHECKING:
    from crows_nest.core.thunk import Thunk, ThunkResult, ThunkStatus


@runtime_checkable
class PersistenceBackend(Protocol):
    """Protocol for persistence backends.

    All methods are async to support non-blocking I/O with databases,
    file systems, or remote storage.
    """

    async def save_thunk(self, thunk: Thunk) -> None:
        """Save a thunk to storage.

        Args:
            thunk: The thunk to save.

        If a thunk with the same ID already exists, it should be updated.
        """
        ...

    async def get_thunk(self, thunk_id: UUID) -> Thunk | None:
        """Retrieve a thunk by ID.

        Args:
            thunk_id: The thunk's unique identifier.

        Returns:
            The thunk if found, None otherwise.
        """
        ...

    async def save_result(self, result: ThunkResult) -> None:
        """Save a thunk result to storage.

        Args:
            result: The result to save.

        If a result for the same thunk ID already exists, it should be updated.
        """
        ...

    async def get_result(self, thunk_id: UUID) -> ThunkResult | None:
        """Retrieve a thunk result by thunk ID.

        Args:
            thunk_id: The thunk's unique identifier.

        Returns:
            The result if found, None otherwise.
        """
        ...

    async def list_thunks(
        self,
        *,
        status: ThunkStatus | None = None,
        operation: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thunk]:
        """List thunks with optional filtering.

        Args:
            status: Filter by result status (requires join with results).
            operation: Filter by operation name.
            limit: Maximum number of thunks to return.
            offset: Number of thunks to skip.

        Returns:
            List of matching thunks.
        """
        ...

    async def delete_thunk(self, thunk_id: UUID) -> bool:
        """Delete a thunk and its result.

        Args:
            thunk_id: The thunk's unique identifier.

        Returns:
            True if the thunk was deleted, False if not found.
        """
        ...

    async def close(self) -> None:
        """Close the backend and release resources."""
        ...
