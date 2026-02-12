"""FastAPI server for Crow's Nest."""

from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crows_nest.api.dashboard import router as dashboard_router
from crows_nest.core.config import get_settings
from crows_nest.core.registry import get_global_registry
from crows_nest.core.runtime import ThunkRuntime
from crows_nest.core.thunk import Thunk
from crows_nest.persistence.sqlite import SQLiteBackend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# Pydantic models for API
class ThunkSubmitRequest(BaseModel):
    """Request body for submitting a thunk."""

    operation: str = Field(
        description="Operation name to execute",
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_.]*$",  # Lowercase, dots, underscores only
    )
    inputs: dict[str, Any] = Field(default_factory=dict, description="Operation inputs")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities to grant",
        max_length=20,  # Max 20 capabilities
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Optional tags")


class ThunkSubmitResponse(BaseModel):
    """Response for thunk submission."""

    thunk_id: str = Field(description="ID of the submitted thunk")
    operation: str = Field(description="Operation name")
    status: str = Field(description="Current status")


class ThunkStatusResponse(BaseModel):
    """Response for thunk status."""

    thunk_id: str = Field(description="Thunk ID")
    status: str = Field(description="Current status")
    operation: str | None = Field(default=None, description="Operation name")
    forced_at: str | None = Field(default=None, description="When the thunk was forced")
    duration_ms: int | None = Field(default=None, description="Execution duration")


class ThunkResultResponse(BaseModel):
    """Response for thunk result."""

    thunk_id: str = Field(description="Thunk ID")
    status: str = Field(description="Status")
    value: Any | None = Field(default=None, description="Result value if successful")
    error: dict[str, Any] | None = Field(default=None, description="Error info if failed")
    forced_at: str | None = Field(default=None, description="When the thunk was forced")
    duration_ms: int | None = Field(default=None, description="Execution duration")


class ThunkDetailsResponse(BaseModel):
    """Response for thunk details."""

    thunk_id: str = Field(description="Thunk ID")
    operation: str = Field(description="Operation name")
    inputs: dict[str, Any] = Field(description="Operation inputs")
    created_at: str = Field(description="Creation timestamp")
    capabilities: list[str] = Field(description="Granted capabilities")
    status: str = Field(description="Current status")


class OperationResponse(BaseModel):
    """Response for operation info."""

    name: str = Field(description="Operation name")
    description: str = Field(description="Operation description")
    required_capabilities: list[str] = Field(description="Required capabilities")
    parameters: dict[str, dict[str, Any]] = Field(description="Parameter info")


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = Field(description="Health status")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Server uptime in seconds")


class ReadyResponse(BaseModel):
    """Response for readiness check."""

    ready: bool = Field(description="Whether the service is ready")
    database: str = Field(description="Database status")
    operations_registered: int = Field(description="Number of registered operations")


class AskRequest(BaseModel):
    """Request for natural language task execution."""

    query: str = Field(
        description="Natural language task description",
        min_length=1,
        max_length=1000,  # Reasonable limit for a task description
    )


class AskResponse(BaseModel):
    """Response for natural language task execution."""

    success: bool = Field(description="Whether the task succeeded")
    query: str = Field(description="Original query")
    output: str = Field(description="Task output")
    steps_completed: int = Field(description="Number of steps completed")
    total_steps: int = Field(description="Total number of steps")
    duration_ms: int = Field(description="Execution duration in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")
    thunk_ids: list[str] = Field(description="IDs of executed thunks")


# Global state
_backend: SQLiteBackend | None = None
_runtime: ThunkRuntime | None = None
_start_time: float = 0.0
_shutdown_event: Any = None  # asyncio.Event for shutdown coordination


async def get_backend() -> SQLiteBackend:
    """Get the database backend."""
    if _backend is None:
        msg = "Backend not initialized"
        raise RuntimeError(msg)
    return _backend


async def get_runtime() -> ThunkRuntime:
    """Get the thunk runtime."""
    if _runtime is None:
        msg = "Runtime not initialized"
        raise RuntimeError(msg)
    return _runtime


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler with graceful shutdown."""
    import asyncio
    import logging
    import time

    global _backend, _runtime, _start_time, _shutdown_event  # noqa: PLW0603

    logger = logging.getLogger("crows_nest.api")

    _start_time = time.time()
    _shutdown_event = asyncio.Event()
    settings = get_settings()

    logger.info("Starting Crow's Nest API server...")

    # Initialize database
    _backend = await SQLiteBackend.create(settings.database.path)
    logger.info("Database initialized: %s", settings.database.path)

    # Initialize runtime
    _runtime = ThunkRuntime(
        registry=get_global_registry(),
        persistence=_backend,
    )

    ops_count = len(get_global_registry().list_operations_info())
    logger.info("Runtime initialized with %d operations", ops_count)

    yield

    # Graceful shutdown
    logger.info("Initiating graceful shutdown...")
    _shutdown_event.set()

    # Give pending tasks a grace period to complete
    shutdown_timeout = 10.0  # seconds
    logger.info("Waiting up to %.1fs for pending tasks to complete...", shutdown_timeout)

    # In a full implementation, we'd wait for pending thunk executions here
    # For now, we just close the database connection cleanly
    await asyncio.sleep(0.1)  # Brief pause for any in-flight requests

    # Cleanup database
    if _backend:
        await _backend.close()
        logger.info("Database connection closed")

    logger.info("Shutdown complete")


# Create app
app = FastAPI(
    title="Crow's Nest",
    description="A thunk-based multi-agent system",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
settings = get_settings()
if settings.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Any, call_next: Any) -> Any:
    """Apply rate limiting to all requests except health checks."""
    from crows_nest.api.ratelimit import get_rate_limiter

    # Skip rate limiting for health check endpoints
    if request.url.path in ("/health", "/ready", "/metrics"):
        return await call_next(request)

    # Apply rate limiting
    get_rate_limiter().check(request)
    return await call_next(request)


# Include dashboard router
app.include_router(dashboard_router)

# Dashboard static files
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

# Mount static files for CSS and JS
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/dashboard", include_in_schema=False)
async def dashboard_index() -> FileResponse:
    """Serve the dashboard index page."""
    index_path = DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found",
        )
    return FileResponse(index_path)


@app.post("/ask", response_model=AskResponse, tags=["Tasks"])
async def ask_task(request: AskRequest) -> AskResponse:
    """Execute a natural language task.

    Takes a natural language description of a task and executes it,
    returning the result.

    Example queries:
    - "list files"
    - "count python files"
    - "show current directory"
    """
    from crows_nest.orchestrator.executor import ask as execute_ask

    backend = await get_backend()
    runtime = await get_runtime()

    result = await execute_ask(
        query=request.query,
        runtime=runtime,
        persistence=backend,
    )

    return AskResponse(
        success=result.success,
        query=result.query,
        output=result.output,
        steps_completed=result.steps_completed,
        total_steps=result.total_steps,
        duration_ms=result.duration_ms,
        error=result.error,
        thunk_ids=result.thunk_ids,
    )


@app.get("/metrics", tags=["Health"], include_in_schema=False)
async def metrics() -> Any:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    import time

    from fastapi.responses import PlainTextResponse

    lines = []

    # Uptime
    uptime = time.time() - _start_time if _start_time > 0 else 0.0
    lines.append("# HELP crows_nest_uptime_seconds Server uptime in seconds")
    lines.append("# TYPE crows_nest_uptime_seconds gauge")
    lines.append(f"crows_nest_uptime_seconds {uptime:.2f}")

    # Operations count
    registry = get_global_registry()
    ops_count = len(registry.list_operations_info())
    lines.append("# HELP crows_nest_operations_total Total registered operations")
    lines.append("# TYPE crows_nest_operations_total gauge")
    lines.append(f"crows_nest_operations_total {ops_count}")

    # Database status
    try:
        backend = await get_backend()
        thunks = await backend.list_thunks(limit=1000)

        # Thunk counts by status
        pending = sum(1 for t in thunks if t.get("status") == "pending")
        success = sum(1 for t in thunks if t.get("status") == "success")
        failure = sum(1 for t in thunks if t.get("status") == "failure")
        total = len(thunks)

        lines.append("# HELP crows_nest_thunks_total Total thunks by status")
        lines.append("# TYPE crows_nest_thunks_total gauge")
        lines.append(f'crows_nest_thunks_total{{status="pending"}} {pending}')
        lines.append(f'crows_nest_thunks_total{{status="success"}} {success}')
        lines.append(f'crows_nest_thunks_total{{status="failure"}} {failure}')
        lines.append(f'crows_nest_thunks_total{{status="all"}} {total}')

        lines.append("# HELP crows_nest_database_up Database connectivity status")
        lines.append("# TYPE crows_nest_database_up gauge")
        lines.append("crows_nest_database_up 1")
    except Exception:
        lines.append("# HELP crows_nest_database_up Database connectivity status")
        lines.append("# TYPE crows_nest_database_up gauge")
        lines.append("crows_nest_database_up 0")

    # Build info
    lines.append("# HELP crows_nest_build_info Build information")
    lines.append("# TYPE crows_nest_build_info gauge")
    lines.append('crows_nest_build_info{version="0.1.0"} 1')

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns basic health status. Use /ready for full readiness check.
    """
    import time

    uptime = time.time() - _start_time if _start_time > 0 else 0.0
    return HealthResponse(status="healthy", version="0.1.0", uptime_seconds=round(uptime, 2))


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def ready() -> ReadyResponse:
    """Readiness check endpoint.

    Verifies database connectivity and returns service status.
    """
    registry = get_global_registry()
    ops_count = len(registry.list_operations_info())

    try:
        backend = await get_backend()
        # Try a simple query to verify database connection
        await backend.list_thunks(limit=1)
        return ReadyResponse(ready=True, database="connected", operations_registered=ops_count)
    except Exception:
        return ReadyResponse(ready=False, database="disconnected", operations_registered=ops_count)


@app.post(
    "/thunks",
    response_model=ThunkSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Thunks"],
)
async def submit_thunk(request: ThunkSubmitRequest) -> ThunkSubmitResponse:
    """Submit a thunk for execution."""
    backend = await get_backend()
    runtime = await get_runtime()

    # Verify operation exists
    registry = get_global_registry()
    if not registry.has(request.operation):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown operation: {request.operation}",
        )

    # Create thunk
    thunk = Thunk.create(
        operation=request.operation,
        inputs=request.inputs,
        capabilities=frozenset(request.capabilities),
        tags=request.tags,
    )

    # Save thunk
    await backend.save_thunk(thunk)

    # Execute thunk (fire and forget for now)
    # In production, this would go to a queue
    # We ignore execution errors here - the result (success or failure)
    # will be stored in the database and can be retrieved via /result endpoint
    with contextlib.suppress(Exception):
        await runtime.force(thunk)

    return ThunkSubmitResponse(
        thunk_id=str(thunk.id),
        operation=thunk.operation,
        status="submitted",
    )


@app.get("/thunks/{thunk_id}", response_model=ThunkDetailsResponse, tags=["Thunks"])
async def get_thunk(thunk_id: str) -> ThunkDetailsResponse:
    """Get thunk details."""
    backend = await get_backend()

    try:
        uuid = UUID(thunk_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid thunk ID: {thunk_id}",
        ) from e

    thunk = await backend.get_thunk(uuid)
    if thunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thunk not found: {thunk_id}",
        )

    # Get result to determine status
    result = await backend.get_result(uuid)
    thunk_status = result.status.value if result else "pending"

    return ThunkDetailsResponse(
        thunk_id=str(thunk.id),
        operation=thunk.operation,
        inputs=thunk.inputs,
        created_at=thunk.metadata.created_at.isoformat(),
        capabilities=sorted(thunk.metadata.capabilities),
        status=thunk_status,
    )


@app.get("/thunks/{thunk_id}/status", response_model=ThunkStatusResponse, tags=["Thunks"])
async def get_thunk_status(thunk_id: str) -> ThunkStatusResponse:
    """Get thunk execution status."""
    backend = await get_backend()

    try:
        uuid = UUID(thunk_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid thunk ID: {thunk_id}",
        ) from e

    # Check for result
    result = await backend.get_result(uuid)
    if result:
        return ThunkStatusResponse(
            thunk_id=thunk_id,
            status=result.status.value,
            forced_at=result.forced_at.isoformat(),
            duration_ms=result.duration_ms,
        )

    # Check if thunk exists
    thunk = await backend.get_thunk(uuid)
    if thunk:
        return ThunkStatusResponse(
            thunk_id=thunk_id,
            status="pending",
            operation=thunk.operation,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Thunk not found: {thunk_id}",
    )


@app.get("/thunks/{thunk_id}/result", response_model=ThunkResultResponse, tags=["Thunks"])
async def get_thunk_result(thunk_id: str) -> ThunkResultResponse:
    """Get thunk result."""
    backend = await get_backend()

    try:
        uuid = UUID(thunk_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid thunk ID: {thunk_id}",
        ) from e

    result = await backend.get_result(uuid)
    if result is None:
        # Check if thunk exists
        thunk = await backend.get_thunk(uuid)
        if thunk is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thunk not found: {thunk_id}",
            )
        return ThunkResultResponse(
            thunk_id=thunk_id,
            status="pending",
        )

    return ThunkResultResponse(
        thunk_id=thunk_id,
        status=result.status.value,
        value=result.value,
        error=result.error.to_dict() if result.error else None,
        forced_at=result.forced_at.isoformat(),
        duration_ms=result.duration_ms,
    )


@app.get("/ops", response_model=list[OperationResponse], tags=["Operations"])
async def list_operations() -> list[OperationResponse]:
    """List all registered operations."""
    registry = get_global_registry()
    operations = registry.list_operations_info()

    return [
        OperationResponse(
            name=op.name,
            description=op.description,
            required_capabilities=sorted(op.required_capabilities),
            parameters={
                name: {
                    "annotation": param.annotation,
                    "required": param.required,
                    "default": param.default,
                }
                for name, param in op.parameters.items()
            },
        )
        for op in operations
    ]
