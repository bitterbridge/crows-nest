"""Command-line interface for Crow's Nest."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import click

from crows_nest.core.config import get_settings

if TYPE_CHECKING:
    from typing import Any


def _run_async(coro: Any) -> Any:
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If there's already a running loop (e.g., in tests with pytest-asyncio),
        # create a new thread to run the coroutine
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


@click.group()
@click.version_option(package_name="crows-nest")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file.",
)
@click.pass_context
def main(ctx: click.Context, config: Path | None) -> None:
    """Crow's Nest: A thunk-based multi-agent system."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = str(config) if config else None


@main.command()
@click.argument("thunk_file", type=click.File("r"), default="-")
@click.option("--wait/--no-wait", default=True, help="Wait for completion.")
@click.pass_context
def run(ctx: click.Context, thunk_file: click.utils.LazyFile, wait: bool) -> None:
    """Execute a thunk from a JSON file or stdin.

    Pass '-' as THUNK_FILE to read from stdin.
    """
    from crows_nest.core import ThunkRuntime, get_global_registry
    from crows_nest.core.thunk import Thunk
    from crows_nest.persistence.sqlite import SQLiteBackend

    # Load thunk from JSON
    try:
        thunk_data = json.load(thunk_file)
        thunk = Thunk.from_dict(thunk_data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise click.ClickException(f"Invalid thunk JSON: {e}") from e

    # Get settings
    settings = get_settings(ctx.obj.get("config_path"))

    async def execute() -> None:
        # Create persistence backend
        backend = await SQLiteBackend.create(settings.database.path)

        try:
            # Create runtime
            runtime = ThunkRuntime(
                registry=get_global_registry(),
                persistence=backend,
            )

            # Save and execute thunk
            await backend.save_thunk(thunk)
            result = await runtime.force(thunk)

            # Output result
            if result.is_success:
                click.echo(json.dumps(result.value, indent=2))
            else:
                error_msg = result.error.message if result.error else "Unknown error"
                raise click.ClickException(f"Thunk failed: {error_msg}")
        finally:
            await backend.close()

    if wait:
        _run_async(execute())
    else:
        # For async execution, just save and return ID
        click.echo(json.dumps({"thunk_id": str(thunk.id), "status": "submitted"}))


@main.command()
@click.argument("thunk_id")
@click.pass_context
def status(ctx: click.Context, thunk_id: str) -> None:
    """Get the status of a thunk."""
    from crows_nest.persistence.sqlite import SQLiteBackend

    settings = get_settings(ctx.obj.get("config_path"))

    async def get_status() -> None:
        backend = await SQLiteBackend.create(settings.database.path)
        try:
            try:
                uuid = UUID(thunk_id)
            except ValueError as e:
                raise click.ClickException(f"Invalid thunk ID: {thunk_id}") from e

            # Check for result first
            result = await backend.get_result(uuid)
            if result:
                click.echo(
                    json.dumps(
                        {
                            "thunk_id": thunk_id,
                            "status": result.status.value,
                            "forced_at": result.forced_at.isoformat(),
                            "duration_ms": result.duration_ms,
                        },
                        indent=2,
                    )
                )
                return

            # Check if thunk exists
            thunk = await backend.get_thunk(uuid)
            if thunk:
                click.echo(
                    json.dumps(
                        {
                            "thunk_id": thunk_id,
                            "status": "pending",
                            "operation": thunk.operation,
                        },
                        indent=2,
                    )
                )
            else:
                raise click.ClickException(f"Thunk not found: {thunk_id}")
        finally:
            await backend.close()

    _run_async(get_status())


@main.command()
@click.argument("thunk_id")
@click.pass_context
def result(ctx: click.Context, thunk_id: str) -> None:
    """Get the result of a thunk."""
    from crows_nest.persistence.sqlite import SQLiteBackend

    settings = get_settings(ctx.obj.get("config_path"))

    async def get_result() -> None:
        backend = await SQLiteBackend.create(settings.database.path)
        try:
            try:
                uuid = UUID(thunk_id)
            except ValueError as e:
                raise click.ClickException(f"Invalid thunk ID: {thunk_id}") from e

            thunk_result = await backend.get_result(uuid)
            if thunk_result is None:
                raise click.ClickException(f"No result found for thunk: {thunk_id}")

            click.echo(json.dumps(thunk_result.to_dict(), indent=2))
        finally:
            await backend.close()

    _run_async(get_result())


@main.command()
@click.argument("query")
@click.option("--verbose", "-v", is_flag=True, help="Show execution progress.")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def ask(ctx: click.Context, query: str, verbose: bool, json_output: bool) -> None:
    """Execute a natural language task.

    Example: crows-nest ask "count python files"
    """
    from crows_nest.core import ThunkRuntime, get_global_registry
    from crows_nest.orchestrator.executor import ExecutionProgress
    from crows_nest.orchestrator.executor import ask as execute_ask
    from crows_nest.persistence.sqlite import SQLiteBackend

    settings = get_settings(ctx.obj.get("config_path"))

    def progress_callback(progress: ExecutionProgress) -> None:
        if not verbose:
            return
        status_color = {
            "pending": "yellow",
            "running": "blue",
            "success": "green",
            "failure": "red",
        }.get(progress.status, "white")

        step_info = f"[{progress.step_index + 1}/{progress.total_steps}]"
        click.echo(
            f"{step_info} {click.style(progress.status, fg=status_color)} "
            f"{progress.step_description}"
        )

        if progress.output and progress.status == "success":
            # Show truncated output in verbose mode
            output = progress.output.strip()
            if len(output) > 200:
                output = output[:200] + "..."
            if output:
                click.echo(f"    → {output}")

        if progress.error:
            click.echo(f"    ✗ {click.style(progress.error, fg='red')}")

    async def execute() -> None:
        backend = await SQLiteBackend.create(settings.database.path)
        try:
            runtime = ThunkRuntime(
                registry=get_global_registry(),
                persistence=backend,
            )

            result = await execute_ask(
                query=query,
                runtime=runtime,
                persistence=backend,
                on_progress=progress_callback if verbose else None,
            )

            if json_output:
                import json as json_module

                click.echo(
                    json_module.dumps(
                        {
                            "success": result.success,
                            "query": result.query,
                            "output": result.output,
                            "steps_completed": result.steps_completed,
                            "total_steps": result.total_steps,
                            "duration_ms": result.duration_ms,
                            "error": result.error,
                            "thunk_ids": result.thunk_ids,
                        },
                        indent=2,
                    )
                )
            elif result.success:
                click.echo(result.output)
            else:
                raise click.ClickException(result.error or "Task failed")
        finally:
            await backend.close()

    _run_async(execute())


@main.group()
def ops() -> None:
    """Manage operations."""


@ops.command("list")
def ops_list() -> None:
    """List all registered operations."""
    from crows_nest.core import get_global_registry

    registry = get_global_registry()
    operations = registry.list_operations_info()

    if not operations:
        click.echo("No operations registered.")
        return

    for op in operations:
        click.echo(f"\n{click.style(op.name, fg='green', bold=True)}")
        click.echo(f"  {op.description}")
        if op.required_capabilities:
            caps = ", ".join(sorted(op.required_capabilities))
            click.echo(f"  Requires: {caps}")
        if op.parameters:
            click.echo("  Parameters:")
            for name, param in op.parameters.items():
                req = "(required)" if param.required else f"(default: {param.default})"
                type_str = param.annotation or "Any"
                click.echo(f"    - {name}: {type_str} {req}")


@main.command()
@click.option("--host", "-h", help="Host to bind to.")
@click.option("--port", "-p", type=int, help="Port to bind to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    reload: bool,
) -> None:
    """Start the API server."""
    import uvicorn

    settings = get_settings(ctx.obj.get("config_path"))

    uvicorn.run(
        "crows_nest.api.server:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )


@main.group()
def config() -> None:
    """Configuration management."""


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show the resolved configuration."""
    settings = get_settings(ctx.obj.get("config_path"))
    click.echo(json.dumps(settings.to_dict(), indent=2))


@config.command("path")
def config_path() -> None:
    """Show the config file search paths."""
    from crows_nest.core.config import _find_config_file

    click.echo("Config file search paths:")
    click.echo("  1. ./config.toml")
    click.echo("  2. ./crows_nest.toml")
    click.echo("  3. ~/.config/crows-nest/config.toml")
    click.echo("  4. /etc/crows-nest/config.toml")
    click.echo()

    found = _find_config_file()
    if found:
        click.echo(f"Found: {click.style(str(found), fg='green')}")
    else:
        click.echo("No config file found, using defaults.")


@main.group()
def plugins() -> None:
    """Plugin management."""


@plugins.command("list")
def plugins_list() -> None:
    """List all loaded plugins."""
    from crows_nest.plugins import PluginLoader

    loader = PluginLoader()
    loaded_plugins = loader.load_all()

    if not loaded_plugins:
        click.echo("No plugins loaded.")
        return

    # Count successful vs failed
    successful = [p for p in loaded_plugins if p.loaded]
    failed = [p for p in loaded_plugins if not p.loaded]

    click.echo(f"Loaded {len(successful)} plugin(s):")
    for plugin in successful:
        source_badge = f"[{plugin.source.value}]"
        click.echo(
            f"  {click.style(plugin.name, fg='green', bold=True)} "
            f"v{plugin.version} {click.style(source_badge, fg='cyan')}"
        )
        if plugin.operations:
            ops = ", ".join(plugin.operations)
            click.echo(f"    Operations: {ops}")
        if plugin.description:
            click.echo(f"    {plugin.description}")

    if failed:
        click.echo(f"\n{click.style('Failed to load', fg='red')} {len(failed)} plugin(s):")
        for plugin in failed:
            click.echo(f"  {click.style(plugin.name, fg='red')}: {plugin.load_error}")


@plugins.command("info")
@click.argument("name")
def plugins_info(name: str) -> None:
    """Show detailed information about a plugin."""
    from crows_nest.plugins import PluginLoader

    loader = PluginLoader()
    loader.load_all()

    plugin = loader.get_plugin(name)
    if plugin is None:
        raise click.ClickException(f"Plugin not found: {name}")

    click.echo(f"{click.style('Name:', bold=True)} {plugin.name}")
    click.echo(f"{click.style('Version:', bold=True)} {plugin.version}")
    click.echo(f"{click.style('Source:', bold=True)} {plugin.source.value}")

    if plugin.author:
        click.echo(f"{click.style('Author:', bold=True)} {plugin.author}")
    if plugin.description:
        click.echo(f"{click.style('Description:', bold=True)} {plugin.description}")
    if plugin.module_name:
        click.echo(f"{click.style('Module:', bold=True)} {plugin.module_name}")

    if plugin.loaded:
        click.echo(f"{click.style('Status:', bold=True)} {click.style('Loaded', fg='green')}")
        if plugin.operations:
            click.echo(f"{click.style('Operations:', bold=True)}")
            for op in plugin.operations:
                click.echo(f"  - {op}")
        else:
            click.echo(f"{click.style('Operations:', bold=True)} (none)")
    else:
        click.echo(f"{click.style('Status:', bold=True)} {click.style('Failed', fg='red')}")
        click.echo(f"{click.style('Error:', bold=True)} {plugin.load_error}")


@main.group()
def queues() -> None:
    """Message queue management."""


@queues.command("list")
def queues_list() -> None:
    """List all message queues."""
    from crows_nest.messaging import get_message_queue

    async def list_queues() -> None:
        queue = await get_message_queue()
        names = queue.list_queues()

        if not names:
            click.echo("No queues.")
            return

        click.echo(f"Found {len(names)} queue(s):")
        for name in names:
            stats = queue.get_stats(name)
            status_parts = []
            if stats.pending > 0:
                status_parts.append(f"{stats.pending} pending")
            if stats.processing > 0:
                status_parts.append(f"{stats.processing} processing")
            if stats.dead > 0:
                status_parts.append(click.style(f"{stats.dead} dead", fg="red"))

            status = ", ".join(status_parts) if status_parts else "empty"
            click.echo(f"  {click.style(name, fg='green', bold=True)}: {status}")

    _run_async(list_queues())


@queues.command("stats")
@click.argument("name")
def queues_stats(name: str) -> None:
    """Show detailed statistics for a queue."""
    from crows_nest.messaging import get_message_queue

    async def show_stats() -> None:
        queue = await get_message_queue()
        stats = queue.get_stats(name)

        click.echo(f"{click.style('Queue:', bold=True)} {name}")
        click.echo(f"{click.style('Pending:', bold=True)} {stats.pending}")
        click.echo(f"{click.style('Processing:', bold=True)} {stats.processing}")
        click.echo(f"{click.style('Completed:', bold=True)} {stats.completed}")
        click.echo(f"{click.style('Failed:', bold=True)} {stats.failed}")
        click.echo(f"{click.style('Dead:', bold=True)} {stats.dead}")
        click.echo(f"{click.style('Total Published:', bold=True)} {stats.total_published}")
        click.echo(f"{click.style('Total Consumed:', bold=True)} {stats.total_consumed}")

    _run_async(show_stats())


@queues.command("dead-letters")
@click.argument("name")
def queues_dead_letters(name: str) -> None:
    """Show dead letter messages for a queue."""
    from crows_nest.messaging import get_message_queue

    async def show_dead_letters() -> None:
        queue = await get_message_queue()
        messages = await queue.get_dead_letters(name)

        if not messages:
            click.echo(f"No dead letters in queue '{name}'.")
            return

        click.echo(f"Dead letters in '{name}' ({len(messages)}):")
        for msg in messages:
            click.echo(f"\n  {click.style('ID:', bold=True)} {msg.id}")
            click.echo(f"  {click.style('Attempts:', bold=True)} {msg.attempts}/{msg.max_attempts}")
            click.echo(f"  {click.style('Error:', bold=True)} {msg.error}")
            click.echo(f"  {click.style('Created:', bold=True)} {msg.created_at.isoformat()}")

    _run_async(show_dead_letters())


@main.group()
def agents() -> None:
    """Agent hierarchy management."""


@agents.command("list")
@click.option("--all", "-a", "show_all", is_flag=True, help="Include terminated agents.")
def agents_list(show_all: bool) -> None:
    """List all agents."""
    from crows_nest.agents import get_agent_registry

    async def list_agents() -> None:
        registry = await get_agent_registry()
        agent_list = registry.list_agents(include_terminated=show_all)

        if not agent_list:
            click.echo("No agents registered.")
            return

        active = [a for a in agent_list if a.is_active]
        terminated = [a for a in agent_list if not a.is_active]

        click.echo(f"Found {len(agent_list)} agent(s):")

        for agent in active:
            depth_indent = "  " * agent.depth
            status = click.style("active", fg="green")
            if agent.parent_id:
                parent_info = f" (parent: {str(agent.parent_id)[:8]}...)"
            else:
                parent_info = " (root)"
            children_count = len(agent.children)
            children_info = f", {children_count} children" if children_count > 0 else ""
            click.echo(
                f"  {depth_indent}{click.style(str(agent.agent_id)[:8], bold=True)}... "
                f"[{status}]{parent_info}{children_info}"
            )

        if show_all and terminated:
            click.echo(f"\nTerminated ({len(terminated)}):")
            for agent in terminated:
                term_time = agent.terminated_at.isoformat() if agent.terminated_at else "unknown"
                click.echo(
                    f"  {click.style(str(agent.agent_id)[:8], fg='red')}... "
                    f"terminated at {term_time}"
                )

    _run_async(list_agents())


@agents.command("tree")
@click.argument("agent_id", required=False)
def agents_tree(agent_id: str | None) -> None:
    """Show agent hierarchy as a tree.

    If AGENT_ID is provided, show tree rooted at that agent.
    Otherwise, show all root agents and their descendants.
    """
    from crows_nest.agents import get_agent_registry

    async def show_tree() -> None:
        registry = await get_agent_registry()

        def print_tree(lineage: Any, prefix: str = "", is_last: bool = True) -> None:
            """Recursively print the agent tree."""
            connector = "└── " if is_last else "├── "
            if lineage.is_active:
                status = click.style("●", fg="green")
            else:
                status = click.style("○", fg="red")
            agent_id_styled = click.style(str(lineage.agent_id)[:8], bold=True)
            click.echo(f"{prefix}{connector}{status} {agent_id_styled}... (depth={lineage.depth})")

            # Print children
            new_prefix = prefix + ("    " if is_last else "│   ")
            children = [registry.get_lineage_sync(c) for c in lineage.children]
            children = [c for c in children if c is not None]

            for i, child in enumerate(children):
                print_tree(child, new_prefix, i == len(children) - 1)

        if agent_id:
            # Show tree from specific agent
            try:
                uuid = UUID(agent_id)
            except ValueError as e:
                raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

            try:
                lineage = await registry.get_lineage(uuid)
            except Exception as e:
                raise click.ClickException(f"Agent not found: {agent_id}") from e

            click.echo(f"Agent hierarchy for {agent_id[:8]}...:")
            print_tree(lineage, "", True)
        else:
            # Show all root agents
            roots = registry.list_root_agents()
            if not roots:
                click.echo("No agents registered.")
                return

            click.echo(f"Agent hierarchy ({len(roots)} root(s)):")
            for i, root in enumerate(roots):
                print_tree(root, "", i == len(roots) - 1)

    _run_async(show_tree())


def _print_lineage_info(lineage: Any) -> None:
    """Print basic lineage information."""
    click.echo(f"{click.style('Agent ID:', bold=True)} {lineage.agent_id}")
    status_text = (
        click.style("active", fg="green")
        if lineage.is_active
        else click.style("terminated", fg="red")
    )
    click.echo(f"{click.style('Status:', bold=True)} {status_text}")
    click.echo(f"{click.style('Depth:', bold=True)} {lineage.depth}")
    click.echo(f"{click.style('Created:', bold=True)} {lineage.created_at.isoformat()}")

    if lineage.terminated_at:
        click.echo(f"{click.style('Terminated:', bold=True)} {lineage.terminated_at.isoformat()}")

    parent_text = str(lineage.parent_id) if lineage.parent_id else "(root agent)"
    click.echo(f"{click.style('Parent:', bold=True)} {parent_text}")

    children_text = f"{len(lineage.children)}" if lineage.children else "(none)"
    click.echo(f"{click.style('Children:', bold=True)} {children_text}")
    for child_id in lineage.children:
        click.echo(f"  - {child_id}")

    if lineage.metadata:
        click.echo(f"{click.style('Metadata:', bold=True)}")
        click.echo(json.dumps(lineage.metadata, indent=4))


def _print_resources_info(resources: Any) -> None:
    """Print resource information for an agent."""
    click.echo(f"\n{click.style('Resources:', bold=True)}")
    click.echo(f"  {click.style('Capabilities:', bold=True)}")
    click.echo(f"    Total: {sorted(resources.capabilities.total)}")
    click.echo(f"    Available: {sorted(resources.capabilities.available)}")
    if resources.capabilities.allocated:
        click.echo("    Allocated:")
        for child, caps in resources.capabilities.allocated.items():
            click.echo(f"      {str(child)[:8]}...: {sorted(caps)}")

    click.echo(f"  {click.style('Quota:', bold=True)}")
    click.echo(f"    Max children: {resources.quota.max_children}")
    click.echo(f"    Children spawned: {resources.quota.children_spawned}")
    click.echo(f"    Max depth: {resources.quota.max_depth}")
    click.echo(f"    Token limit: {resources.quota.max_total_tokens}")
    click.echo(f"    Tokens used: {resources.quota.total_tokens_used}")


@agents.command("info")
@click.argument("agent_id")
def agents_info(agent_id: str) -> None:
    """Show detailed information about an agent."""
    from crows_nest.agents import get_agent_registry
    from crows_nest.agents.spawning import get_agent_resources

    async def show_info() -> None:
        registry = await get_agent_registry()

        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        try:
            lineage = await registry.get_lineage(uuid)
        except Exception as e:
            raise click.ClickException(f"Agent not found: {agent_id}") from e

        resources = await get_agent_resources(uuid)

        _print_lineage_info(lineage)
        if resources:
            _print_resources_info(resources)

    _run_async(show_info())


@agents.command("terminate")
@click.argument("agent_id")
@click.option("--cascade/--no-cascade", default=True, help="Also terminate child agents.")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt.")
def agents_terminate(agent_id: str, cascade: bool, force: bool) -> None:
    """Terminate an agent."""
    from crows_nest.agents import get_agent_registry
    from crows_nest.agents.spawning import remove_agent_resources

    async def terminate() -> None:
        registry = await get_agent_registry()

        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        try:
            lineage = await registry.get_lineage(uuid)
        except Exception as e:
            raise click.ClickException(f"Agent not found: {agent_id}") from e

        if not lineage.is_active:
            raise click.ClickException(f"Agent {agent_id} is already terminated.")

        # Count descendants
        descendants = await registry.get_descendants(uuid)
        active_descendants = [d for d in descendants if d.is_active]

        if not force:
            msg = f"Terminate agent {agent_id[:8]}...?"
            if cascade and active_descendants:
                count = len(active_descendants)
                msg = f"Terminate agent {agent_id[:8]}... and {count} descendant(s)?"
            if not click.confirm(msg):
                click.echo("Aborted.")
                return

        terminated_ids = await registry.terminate(uuid, cascade=cascade)

        # Clean up resources
        for tid in terminated_ids:
            await remove_agent_resources(tid)

        click.echo(f"Terminated {len(terminated_ids)} agent(s):")
        for tid in terminated_ids:
            click.echo(f"  - {tid}")

    _run_async(terminate())


@agents.command("capabilities")
@click.argument("agent_id")
def agents_capabilities(agent_id: str) -> None:
    """Show capabilities for an agent."""
    from crows_nest.agents.spawning import get_agent_resources

    async def show_capabilities() -> None:
        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        resources = await get_agent_resources(uuid)
        if resources is None:
            raise click.ClickException(f"No resources found for agent: {agent_id}")

        caps = resources.capabilities
        click.echo(f"{click.style('Capabilities for', bold=True)} {agent_id[:8]}...:\n")

        click.echo(f"{click.style('Total:', bold=True)} ({len(caps.total)})")
        for cap in sorted(caps.total):
            click.echo(f"  - {cap}")

        click.echo(f"\n{click.style('Available:', bold=True)} ({len(caps.available)})")
        for cap in sorted(caps.available):
            click.echo(f"  - {cap}")

        if caps.reserved:
            click.echo(f"\n{click.style('Reserved:', bold=True)} ({len(caps.reserved)})")
            for cap in sorted(caps.reserved):
                click.echo(f"  - {cap}")

        if caps.allocated:
            click.echo(f"\n{click.style('Allocated to children:', bold=True)}")
            for child_id, child_caps in caps.allocated.items():
                click.echo(f"  {str(child_id)[:8]}...:")
                for cap in sorted(child_caps):
                    click.echo(f"    - {cap}")

    _run_async(show_capabilities())


@main.group()
def memory() -> None:
    """Agent memory management."""


@memory.command("list")
@click.argument("agent_id")
@click.option("--type", "-t", "memory_type", help="Filter by memory type.")
@click.option("--scope", "-s", help="Filter by scope.")
@click.option("--limit", "-n", default=10, help="Maximum results.")
@click.option("--min-importance", default=0.0, help="Minimum importance threshold.")
def memory_list(
    agent_id: str,
    memory_type: str | None,
    scope: str | None,
    limit: int,
    min_importance: float,
) -> None:
    """List memories for an agent."""
    from crows_nest.memory import SQLiteMemoryStore
    from crows_nest.memory.models import MemoryQuery, MemoryScope, MemoryType

    async def list_memories() -> None:
        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        store = SQLiteMemoryStore()
        await store.initialize()

        query = MemoryQuery(
            memory_type=MemoryType(memory_type) if memory_type else None,
            scope=MemoryScope(scope) if scope else None,
            min_importance=min_importance,
            limit=limit,
        )

        try:
            memories = await store.recall(agent_id=uuid, query=query)
        finally:
            await store.close()

        if not memories:
            click.echo("No memories found.")
            return

        click.echo(f"Found {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:")
        for mem in memories:
            type_color = {
                "episodic": "cyan",
                "semantic": "blue",
                "procedural": "magenta",
            }.get(mem.memory_type.value, "white")
            content_preview = mem.content[:60] + "..." if len(mem.content) > 60 else mem.content
            click.echo(
                f"  {click.style(str(mem.id)[:8], bold=True)}... "
                f"[{click.style(mem.memory_type.value, fg=type_color)}] "
                f"imp={mem.importance:.2f} "
            )
            click.echo(f"    {content_preview}")

    _run_async(list_memories())


@memory.command("stats")
@click.argument("agent_id")
def memory_stats(agent_id: str) -> None:
    """Show memory statistics for an agent."""
    from crows_nest.memory import SQLiteMemoryStore

    async def show_stats() -> None:
        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        store = SQLiteMemoryStore()
        await store.initialize()

        try:
            stats = await store.get_stats(agent_id=uuid)
        finally:
            await store.close()

        click.echo(f"Memory stats for agent {agent_id[:8]}...:")
        click.echo(f"  Total: {stats.total_count}")

        if stats.by_type:
            click.echo(f"\n  {click.style('By type:', bold=True)}")
            for mem_type, count in stats.by_type.items():
                click.echo(f"    {mem_type}: {count}")

        if stats.by_scope:
            click.echo(f"\n  {click.style('By scope:', bold=True)}")
            for scope, count in stats.by_scope.items():
                click.echo(f"    {scope}: {count}")

        click.echo(f"\n  Avg importance: {stats.avg_importance:.2f}")
        click.echo(f"  Storage: {stats.total_size_bytes:,} bytes")

        if stats.oldest:
            click.echo(f"  Oldest: {stats.oldest.isoformat()}")
        if stats.newest:
            click.echo(f"  Newest: {stats.newest.isoformat()}")

    _run_async(show_stats())


@memory.command("get")
@click.argument("agent_id")
@click.argument("memory_id")
def memory_get(agent_id: str, memory_id: str) -> None:
    """Get a specific memory by ID."""
    from crows_nest.memory import SQLiteMemoryStore

    async def show_memory() -> None:
        try:
            agent_uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        try:
            memory_uuid = UUID(memory_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid memory ID: {memory_id}") from e

        store = SQLiteMemoryStore()
        await store.initialize()

        try:
            mem = await store.get(agent_id=agent_uuid, memory_id=memory_uuid)
        finally:
            await store.close()

        if not mem:
            raise click.ClickException(f"Memory not found: {memory_id}")

        click.echo(f"Memory {click.style(str(mem.id), bold=True)}:")
        click.echo(f"  Agent: {mem.agent_id}")
        click.echo(f"  Type: {mem.memory_type.value}")
        click.echo(f"  Scope: {mem.scope.value}")
        click.echo(f"  Importance: {mem.importance:.2f}")
        click.echo(f"  Created: {mem.created_at.isoformat()}")
        click.echo(f"  Accessed: {mem.accessed_at.isoformat()}")
        click.echo(f"  Access count: {mem.access_count}")
        if mem.expires_at:
            click.echo(f"  Expires: {mem.expires_at.isoformat()}")
        click.echo(f"\n  {click.style('Content:', bold=True)}")
        click.echo(f"    {mem.content}")
        if mem.metadata:
            click.echo(f"\n  {click.style('Metadata:', bold=True)}")
            click.echo(f"    {json.dumps(mem.metadata, indent=4)}")

    _run_async(show_memory())


@memory.command("forget")
@click.argument("agent_id")
@click.argument("memory_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation.")
def memory_forget(agent_id: str, memory_id: str, force: bool) -> None:
    """Delete a specific memory."""
    from crows_nest.memory import SQLiteMemoryStore

    async def delete_memory() -> None:
        try:
            agent_uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        try:
            memory_uuid = UUID(memory_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid memory ID: {memory_id}") from e

        store = SQLiteMemoryStore()
        await store.initialize()

        try:
            # Check if memory exists
            mem = await store.get(agent_id=agent_uuid, memory_id=memory_uuid)
            if not mem:
                raise click.ClickException(f"Memory not found: {memory_id}")

            if not force:
                content_preview = mem.content[:40] + "..." if len(mem.content) > 40 else mem.content
                if not click.confirm(f"Delete memory '{content_preview}'?"):
                    click.echo("Aborted.")
                    return

            deleted = await store.forget(agent_id=agent_uuid, memory_id=memory_uuid)
        finally:
            await store.close()

        if deleted:
            click.echo(f"Deleted memory {memory_id[:8]}...")
        else:
            raise click.ClickException("Failed to delete memory.")

    _run_async(delete_memory())


@memory.command("decay")
@click.argument("agent_id")
@click.option("--threshold", default=0.1, help="Decay threshold (default: 0.1).")
@click.option("--half-life", default=86400.0, help="Half-life in seconds (default: 1 day).")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted.")
def memory_decay(agent_id: str, threshold: float, half_life: float, dry_run: bool) -> None:
    """Remove low-importance memories based on decay score."""
    from crows_nest.memory import SQLiteMemoryStore

    async def run_decay() -> None:
        try:
            uuid = UUID(agent_id)
        except ValueError as e:
            raise click.ClickException(f"Invalid agent ID: {agent_id}") from e

        store = SQLiteMemoryStore()
        await store.initialize()

        try:
            if dry_run:
                # Preview what would be deleted
                from crows_nest.memory.models import MemoryQuery

                query = MemoryQuery(limit=1000)
                all_memories = await store.recall(agent_id=uuid, query=query)
                would_delete = [m for m in all_memories if m.decay_score(half_life) < threshold]
                count = len(would_delete)
                suffix = "y" if count == 1 else "ies"
                click.echo(f"Would delete {count} memor{suffix}:")
                for mem in would_delete[:10]:
                    score = mem.decay_score(half_life)
                    preview = mem.content[:40] + "..." if len(mem.content) > 40 else mem.content
                    click.echo(f"  {str(mem.id)[:8]}... (score={score:.3f}): {preview}")
                if len(would_delete) > 10:
                    click.echo(f"  ... and {len(would_delete) - 10} more")
            else:
                count = await store.decay(
                    agent_id=uuid, threshold=threshold, half_life_seconds=half_life
                )
                click.echo(f"Decayed {count} memor{'y' if count == 1 else 'ies'}.")
        finally:
            await store.close()

    _run_async(run_decay())
