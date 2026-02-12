"""Dashboard API endpoints for Crow's Nest.

Provides endpoints for the web dashboard including system overview,
thunk inspection, agent hierarchy, queue monitoring, and memory browsing.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field


# Pydantic models for dashboard responses
class SystemOverview(BaseModel):
    """System overview statistics."""

    active_thunks: int = Field(description="Number of active thunks")
    completed_thunks: int = Field(description="Number of completed thunks")
    failed_thunks: int = Field(description="Number of failed thunks")
    active_agents: int = Field(description="Number of active agents")
    queue_depth: int = Field(description="Total messages in queues")
    uptime_seconds: float = Field(description="Server uptime in seconds")


class ThunkSummary(BaseModel):
    """Summary of a thunk for list views."""

    id: str = Field(description="Thunk ID")
    operation: str = Field(description="Operation name")
    status: str = Field(description="Current status")
    created_at: str = Field(description="Creation timestamp")
    duration_ms: int | None = Field(default=None, description="Execution duration")


class ThunkDetail(BaseModel):
    """Detailed thunk information."""

    id: str = Field(description="Thunk ID")
    operation: str = Field(description="Operation name")
    status: str = Field(description="Current status")
    inputs: dict[str, Any] = Field(description="Operation inputs")
    capabilities: list[str] = Field(description="Granted capabilities")
    dependencies: list[str] = Field(description="Dependency thunk IDs")
    created_at: str = Field(description="Creation timestamp")
    forced_at: str | None = Field(default=None, description="Execution timestamp")
    duration_ms: int | None = Field(default=None, description="Execution duration")
    value: Any | None = Field(default=None, description="Result value")
    error: dict[str, Any] | None = Field(default=None, description="Error info")
    trace_id: str = Field(description="Trace ID for correlation")


class AgentSummary(BaseModel):
    """Summary of an agent for list views."""

    id: str = Field(description="Agent ID")
    parent_id: str | None = Field(default=None, description="Parent agent ID")
    depth: int = Field(description="Depth in hierarchy")
    children_count: int = Field(description="Number of children")
    is_active: bool = Field(description="Whether agent is active")
    created_at: str = Field(description="Creation timestamp")


class AgentTree(BaseModel):
    """Agent hierarchy tree node."""

    id: str = Field(description="Agent ID")
    is_active: bool = Field(description="Whether agent is active")
    depth: int = Field(description="Depth in hierarchy")
    capabilities: list[str] = Field(description="Agent capabilities")
    children: list[AgentTree] = Field(default_factory=list, description="Child agents")


class QueueSummary(BaseModel):
    """Summary of a message queue."""

    name: str = Field(description="Queue name")
    pending: int = Field(description="Pending messages")
    processing: int = Field(description="Messages being processed")
    dead: int = Field(description="Dead letter count")


class MemorySummary(BaseModel):
    """Summary of an agent's memory entry."""

    id: str = Field(description="Memory ID")
    memory_type: str = Field(description="Memory type")
    scope: str = Field(description="Memory scope")
    content_preview: str = Field(description="Content preview")
    importance: float = Field(description="Importance score")
    created_at: str = Field(description="Creation timestamp")
    accessed_at: str = Field(description="Last access timestamp")


class EventMessage(BaseModel):
    """Event message for WebSocket streaming."""

    id: str = Field(description="Event ID")
    event_type: str = Field(description="Event type")
    timestamp: str = Field(description="Event timestamp")
    payload: dict[str, Any] = Field(description="Event payload")


class ConfigView(BaseModel):
    """Configuration view (read-only sensitive values)."""

    database_path: str = Field(description="Database path")
    api_host: str = Field(description="API host")
    api_port: int = Field(description="API port")
    log_level: str = Field(description="Log level")
    tracing_enabled: bool = Field(description="Whether tracing is enabled")


class DAGNode(BaseModel):
    """Node in the thunk dependency graph."""

    id: str = Field(description="Thunk ID")
    label: str = Field(description="Display label (operation name)")
    status: str = Field(description="Current status: pending, running, success, failure")
    operation: str = Field(description="Operation name")
    created_at: str = Field(description="Creation timestamp")
    duration_ms: int | None = Field(default=None, description="Execution duration")


class DAGEdge(BaseModel):
    """Edge in the thunk dependency graph."""

    source: str = Field(description="Source thunk ID (dependency)")
    target: str = Field(description="Target thunk ID (dependent)")


class ThunkDAG(BaseModel):
    """Thunk dependency graph for visualization."""

    nodes: list[DAGNode] = Field(description="Graph nodes (thunks)")
    edges: list[DAGEdge] = Field(description="Graph edges (dependencies)")


class PaginatedThunks(BaseModel):
    """Paginated list of thunks with metadata."""

    items: list[ThunkSummary] = Field(description="Thunk summaries")
    total: int = Field(description="Total count matching filters")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Current offset")
    has_more: bool = Field(description="Whether there are more results")


# Track server start time for uptime calculation
_server_start_time = datetime.now(UTC)

# WebSocket connections for event streaming
_websocket_connections: set[WebSocket] = set()


# Create router
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/overview", response_model=SystemOverview)
async def get_overview() -> SystemOverview:
    """Get system overview statistics."""
    # Imports here to avoid circular imports
    from crows_nest.agents import get_agent_registry
    from crows_nest.api.server import get_backend
    from crows_nest.messaging.queue import get_message_queue

    backend = await get_backend()
    uptime = (datetime.now(UTC) - _server_start_time).total_seconds()

    # Count thunks by status
    all_thunks = await backend.list_thunks(limit=10000)
    active_count = 0
    completed_count = 0
    failed_count = 0

    for thunk in all_thunks:
        result = await backend.get_result(thunk.id)
        if result is None:
            active_count += 1
        elif result.is_success:
            completed_count += 1
        else:
            failed_count += 1

    # Count active agents
    try:
        registry = await get_agent_registry()
        agents = registry.list_agents(include_terminated=False)
        active_agents = len(agents)
    except Exception:
        active_agents = 0

    # Get total queue depth
    try:
        queue = await get_message_queue()
        queue_names = queue.list_queues()
        total_depth = sum(queue.get_pending_count(name) for name in queue_names)
    except Exception:
        total_depth = 0

    return SystemOverview(
        active_thunks=active_count,
        completed_thunks=completed_count,
        failed_thunks=failed_count,
        active_agents=active_agents,
        queue_depth=total_depth,
        uptime_seconds=uptime,
    )


@router.get("/thunks", response_model=list[ThunkSummary])
async def list_thunks(
    status: str | None = Query(
        default=None,
        description="Filter by status",
        pattern=r"^(pending|success|failure)$",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
) -> list[ThunkSummary]:
    """List thunks with optional filtering."""
    from crows_nest.api.server import get_backend
    from crows_nest.core.thunk import ThunkStatus as CoreThunkStatus

    backend = await get_backend()

    # Map status filter to ThunkStatus enum for server-side filtering
    status_filter = None
    if status == "success":
        status_filter = CoreThunkStatus.COMPLETED
    elif status == "failure":
        status_filter = CoreThunkStatus.FAILED
    # Note: "pending" means no result, can't filter server-side

    # Get thunks with server-side filtering when possible
    thunks = await backend.list_thunks(
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    # Batch fetch all results
    thunk_ids = [t.id for t in thunks]
    results_map = await backend.get_results_by_ids(thunk_ids)

    results = []
    for thunk in thunks:
        result = results_map.get(thunk.id)

        if result is None:
            thunk_status = "pending"
            duration = None
        else:
            thunk_status = result.status.value
            duration = result.duration_ms

        # Apply pending filter client-side (can't do server-side)
        if status == "pending" and result is not None:
            continue

        results.append(
            ThunkSummary(
                id=str(thunk.id),
                operation=thunk.operation,
                status=thunk_status,
                created_at=thunk.metadata.created_at.isoformat(),
                duration_ms=duration,
            )
        )

    return results


@router.get("/thunks/paginated", response_model=PaginatedThunks)
async def list_thunks_paginated(
    status: str | None = Query(
        default=None,
        description="Filter by status",
        pattern=r"^(pending|success|failure)$",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
) -> PaginatedThunks:
    """List thunks with pagination metadata."""
    from crows_nest.api.server import get_backend
    from crows_nest.core.thunk import ThunkStatus as CoreThunkStatus

    backend = await get_backend()

    # Map status filter to ThunkStatus enum for server-side filtering
    status_filter = None
    if status == "success":
        status_filter = CoreThunkStatus.COMPLETED
    elif status == "failure":
        status_filter = CoreThunkStatus.FAILED

    # Get total count for pagination
    total = await backend.count_thunks(status=status_filter)

    # Get thunks with server-side filtering
    thunks = await backend.list_thunks(
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    # Batch fetch all results
    thunk_ids = [t.id for t in thunks]
    results_map = await backend.get_results_by_ids(thunk_ids)

    items = []
    for thunk in thunks:
        result = results_map.get(thunk.id)

        if result is None:
            thunk_status = "pending"
            duration = None
        else:
            thunk_status = result.status.value
            duration = result.duration_ms

        # Apply pending filter client-side (can't do server-side)
        if status == "pending" and result is not None:
            continue

        items.append(
            ThunkSummary(
                id=str(thunk.id),
                operation=thunk.operation,
                status=thunk_status,
                created_at=thunk.metadata.created_at.isoformat(),
                duration_ms=duration,
            )
        )

    return PaginatedThunks(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(items)) < total,
    )


@router.get("/thunks/{thunk_id}", response_model=ThunkDetail)
async def get_thunk_detail(thunk_id: str) -> ThunkDetail:
    """Get detailed thunk information."""
    from crows_nest.api.server import get_backend

    backend = await get_backend()

    try:
        uuid = UUID(thunk_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid thunk ID: {thunk_id}") from e

    thunk = await backend.get_thunk(uuid)
    if thunk is None:
        raise HTTPException(status_code=404, detail=f"Thunk not found: {thunk_id}")

    result = await backend.get_result(uuid)

    if result is None:
        return ThunkDetail(
            id=str(thunk.id),
            operation=thunk.operation,
            status="pending",
            inputs=thunk.inputs,
            capabilities=sorted(thunk.metadata.capabilities),
            dependencies=[str(dep.thunk_id) for dep in thunk.get_dependencies()],
            created_at=thunk.metadata.created_at.isoformat(),
            trace_id=thunk.metadata.trace_id,
        )

    return ThunkDetail(
        id=str(thunk.id),
        operation=thunk.operation,
        status=result.status.value,
        inputs=thunk.inputs,
        capabilities=sorted(thunk.metadata.capabilities),
        dependencies=[str(dep.thunk_id) for dep in thunk.get_dependencies()],
        created_at=thunk.metadata.created_at.isoformat(),
        forced_at=result.forced_at.isoformat(),
        duration_ms=result.duration_ms,
        value=result.value,
        error=result.error.to_dict() if result.error else None,
        trace_id=thunk.metadata.trace_id,
    )


@router.get("/thunks/graph", response_model=ThunkDAG)
async def get_thunk_dag(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum thunks"),
    trace_id: str | None = Query(default=None, max_length=100, description="Filter by trace ID"),
) -> ThunkDAG:
    """Get thunk dependency graph for visualization.

    Returns nodes (thunks) and edges (dependencies) for D3.js visualization.
    """
    from crows_nest.api.server import get_backend

    backend = await get_backend()
    thunks = await backend.list_thunks(limit=limit)

    # Filter by trace_id if specified
    if trace_id:
        thunks = [t for t in thunks if t.metadata.trace_id == trace_id]

    # Batch fetch all results
    thunk_uuid_ids = [t.id for t in thunks]
    results_map = await backend.get_results_by_ids(thunk_uuid_ids)

    # Build nodes and edges
    nodes: list[DAGNode] = []
    edges: list[DAGEdge] = []
    thunk_ids = {str(t.id) for t in thunks}

    for thunk in thunks:
        result = results_map.get(thunk.id)

        if result is None:
            status = "pending"
            duration = None
        else:
            status = result.status.value
            duration = result.duration_ms

        # Shorten operation name for display
        label = thunk.operation.split(".")[-1] if "." in thunk.operation else thunk.operation

        nodes.append(
            DAGNode(
                id=str(thunk.id),
                label=label,
                status=status,
                operation=thunk.operation,
                created_at=thunk.metadata.created_at.isoformat(),
                duration_ms=duration,
            )
        )

        # Add edges for dependencies (only if dependency is in our result set)
        for dep in thunk.get_dependencies():
            dep_id = str(dep.thunk_id)
            if dep_id in thunk_ids:
                edges.append(
                    DAGEdge(
                        source=dep_id,
                        target=str(thunk.id),
                    )
                )

    return ThunkDAG(nodes=nodes, edges=edges)


@router.get("/agents", response_model=list[AgentSummary])
async def list_agents(
    include_terminated: bool = Query(default=False, description="Include terminated agents"),
) -> list[AgentSummary]:
    """List all agents."""
    from crows_nest.agents import get_agent_registry

    try:
        registry = await get_agent_registry()
        agents = registry.list_agents(include_terminated=include_terminated)
    except Exception:
        return []

    return [
        AgentSummary(
            id=str(agent.agent_id),
            parent_id=str(agent.parent_id) if agent.parent_id else None,
            depth=agent.depth,
            children_count=len(agent.children),
            is_active=agent.is_active,
            created_at=agent.created_at.isoformat(),
        )
        for agent in agents
    ]


@router.get("/agents/{agent_id}/tree", response_model=AgentTree)
async def get_agent_tree(agent_id: str) -> AgentTree:
    """Get agent hierarchy tree rooted at specified agent."""
    from crows_nest.agents import get_agent_registry
    from crows_nest.agents.spawning import get_agent_resources

    try:
        uuid = UUID(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {agent_id}") from e

    try:
        registry = await get_agent_registry()
        # Verify agent exists
        await registry.get_lineage(uuid)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}") from e

    async def build_tree(agent_uuid: UUID) -> AgentTree:
        agent_lineage = await registry.get_lineage(agent_uuid)
        resources = await get_agent_resources(agent_uuid)

        caps: list[str] = []
        if resources:
            caps = sorted(resources.capabilities.available)

        children_trees = []
        for child_id in agent_lineage.children:
            try:
                child_tree = await build_tree(child_id)
                children_trees.append(child_tree)
            except Exception:  # noqa: S112  # nosec B112 - intentionally skip failed children
                # Skip failed children in tree view
                continue

        return AgentTree(
            id=str(agent_uuid),
            is_active=agent_lineage.is_active,
            depth=agent_lineage.depth,
            capabilities=caps,
            children=children_trees,
        )

    return await build_tree(uuid)


@router.get("/queues", response_model=list[QueueSummary])
async def list_queues() -> list[QueueSummary]:
    """List all message queues with stats."""
    from crows_nest.messaging.queue import get_message_queue

    try:
        queue = await get_message_queue()
        queue_names = queue.list_queues()
    except Exception:
        return []

    results = []
    for name in queue_names:
        stats = queue.get_stats(name)
        results.append(
            QueueSummary(
                name=name,
                pending=stats.pending,
                processing=stats.processing,
                dead=stats.dead,
            )
        )

    return results


@router.get("/memory/{agent_id}", response_model=list[MemorySummary])
async def browse_memory(
    agent_id: str,
    query: str | None = Query(default=None, description="Search text"),
    memory_type: str | None = Query(default=None, description="Filter by type"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
) -> list[MemorySummary]:
    """Browse agent memories."""
    from crows_nest.memory import SQLiteMemoryStore
    from crows_nest.memory.models import MemoryQuery, MemoryType

    try:
        uuid = UUID(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {agent_id}") from e

    store = SQLiteMemoryStore()
    await store.initialize()

    try:
        mem_query = MemoryQuery(
            text=query,
            memory_type=MemoryType(memory_type) if memory_type else None,
            limit=limit,
        )
        memories = await store.recall(agent_id=uuid, query=mem_query)
    finally:
        await store.close()

    results = []
    for mem in memories:
        preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
        results.append(
            MemorySummary(
                id=str(mem.id),
                memory_type=mem.memory_type.value,
                scope=mem.scope.value,
                content_preview=preview,
                importance=mem.importance,
                created_at=mem.created_at.isoformat(),
                accessed_at=mem.accessed_at.isoformat(),
            )
        )

    return results


@router.get("/config", response_model=ConfigView)
async def get_config() -> ConfigView:
    """Get current configuration (sanitized)."""
    from crows_nest.core.config import get_settings

    settings = get_settings()

    return ConfigView(
        database_path=settings.database.path,
        api_host=settings.api.host,
        api_port=settings.api.port,
        log_level=settings.general.log_level,
        tracing_enabled=settings.tracing.enabled,
    )


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming."""
    from crows_nest.messaging import get_event_bus

    await websocket.accept()
    _websocket_connections.add(websocket)

    bus = get_event_bus()

    async def forward_event(event: Any) -> None:
        """Forward events to this WebSocket client."""
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "id": str(event.id),
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "payload": event.payload,
                }
            )

    # Subscribe to all events
    bus.subscribe("*", forward_event)

    try:
        while True:
            # Keep connection alive by receiving messages
            # Client can send ping/pong or filter requests
            data = await websocket.receive_text()
            # Could process filter requests here
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe("*", forward_event)
        _websocket_connections.discard(websocket)


async def broadcast_event(event: dict[str, Any]) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    disconnected = set()
    for ws in _websocket_connections:
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.add(ws)

    _websocket_connections.difference_update(disconnected)
