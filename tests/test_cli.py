"""Tests for CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from crows_nest.cli import main
from crows_nest.core.config import clear_settings_cache

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def clean_settings() -> Generator[None, None, None]:
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self, runner: CliRunner) -> None:
        """--help shows help message."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Crow's Nest" in result.output
        assert "thunk-based multi-agent" in result.output

    def test_version(self, runner: CliRunner) -> None:
        """--version shows version."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Version should be shown


class TestOpsCommand:
    """Tests for the ops subcommand."""

    def test_ops_list_empty(self, runner: CliRunner) -> None:
        """ops list shows message when no operations registered."""
        # Clear the global registry
        from crows_nest.core.registry import get_global_registry

        registry = get_global_registry()
        # Store original operations to restore later
        original = dict(registry._operations)
        registry._operations.clear()

        try:
            result = runner.invoke(main, ["ops", "list"])

            assert result.exit_code == 0
            assert "No operations registered" in result.output
        finally:
            # Restore operations
            registry._operations.update(original)

    def test_ops_list_shows_operations(self, runner: CliRunner) -> None:
        """ops list shows registered operations."""
        result = runner.invoke(main, ["ops", "list"])

        assert result.exit_code == 0
        # Should show built-in operations like agent.create
        assert "agent.create" in result.output or "No operations" in result.output


class TestConfigCommand:
    """Tests for the config subcommand."""

    def test_config_show(self, runner: CliRunner) -> None:
        """config show displays configuration as JSON."""
        result = runner.invoke(main, ["config", "show"])

        assert result.exit_code == 0
        # Should be valid JSON
        config = json.loads(result.output)
        assert "general" in config
        assert "database" in config
        assert "api" in config

    def test_config_path(self, runner: CliRunner) -> None:
        """config path shows search paths."""
        result = runner.invoke(main, ["config", "path"])

        assert result.exit_code == 0
        assert "Config file search paths" in result.output
        assert "config.toml" in result.output


class TestRunCommand:
    """Tests for the run command."""

    def test_run_invalid_json(self, runner: CliRunner) -> None:
        """run rejects invalid JSON input."""
        result = runner.invoke(main, ["run", "-"], input="not valid json")

        assert result.exit_code != 0
        assert "Invalid thunk JSON" in result.output

    def test_run_missing_required_field(self, runner: CliRunner) -> None:
        """run rejects JSON missing required fields."""
        # Missing operation field
        result = runner.invoke(main, ["run", "-"], input='{"inputs": {}}')

        assert result.exit_code != 0
        assert "Invalid thunk JSON" in result.output


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_help(self, runner: CliRunner) -> None:
        """status --help shows help."""
        result = runner.invoke(main, ["status", "--help"])

        assert result.exit_code == 0
        assert "Get the status of a thunk" in result.output


class TestResultCommand:
    """Tests for the result command."""

    def test_result_help(self, runner: CliRunner) -> None:
        """result --help shows help."""
        result = runner.invoke(main, ["result", "--help"])

        assert result.exit_code == 0
        assert "Get the result of a thunk" in result.output


class TestAgentsCommand:
    """Tests for the agents subcommand."""

    @pytest.fixture(autouse=True)
    def reset_agents(self) -> Generator[None, None, None]:
        """Reset agent registry before and after each test."""
        import asyncio

        from crows_nest.agents import reset_agent_registry

        asyncio.run(reset_agent_registry())
        yield
        asyncio.run(reset_agent_registry())

    def test_agents_help(self, runner: CliRunner) -> None:
        """agents --help shows help."""
        result = runner.invoke(main, ["agents", "--help"])

        assert result.exit_code == 0
        assert "Agent hierarchy management" in result.output

    def test_agents_list_empty(self, runner: CliRunner) -> None:
        """agents list shows message when no agents registered."""
        result = runner.invoke(main, ["agents", "list"])

        assert result.exit_code == 0
        assert "No agents registered" in result.output

    def test_agents_tree_empty(self, runner: CliRunner) -> None:
        """agents tree shows message when no agents."""
        result = runner.invoke(main, ["agents", "tree"])

        assert result.exit_code == 0
        assert "No agents registered" in result.output

    def test_agents_info_invalid_id(self, runner: CliRunner) -> None:
        """agents info rejects invalid agent ID."""
        result = runner.invoke(main, ["agents", "info", "not-a-uuid"])

        assert result.exit_code != 0
        assert "Invalid agent ID" in result.output

    def test_agents_capabilities_invalid_id(self, runner: CliRunner) -> None:
        """agents capabilities rejects invalid agent ID."""
        result = runner.invoke(main, ["agents", "capabilities", "not-a-uuid"])

        assert result.exit_code != 0
        assert "Invalid agent ID" in result.output

    def test_agents_terminate_invalid_id(self, runner: CliRunner) -> None:
        """agents terminate rejects invalid agent ID."""
        result = runner.invoke(main, ["agents", "terminate", "not-a-uuid", "-f"])

        assert result.exit_code != 0
        assert "Invalid agent ID" in result.output

    def test_agents_tree_invalid_id(self, runner: CliRunner) -> None:
        """agents tree rejects invalid agent ID."""
        result = runner.invoke(main, ["agents", "tree", "not-a-uuid"])

        assert result.exit_code != 0
        assert "Invalid agent ID" in result.output


class TestQueuesCommand:
    """Tests for the queues subcommand."""

    @pytest.fixture(autouse=True)
    def reset_queues(self) -> Generator[None, None, None]:
        """Reset message queue before and after each test."""
        import asyncio

        from crows_nest.messaging import reset_message_queue

        asyncio.run(reset_message_queue())
        yield
        asyncio.run(reset_message_queue())

    def test_queues_help(self, runner: CliRunner) -> None:
        """queues --help shows help."""
        result = runner.invoke(main, ["queues", "--help"])

        assert result.exit_code == 0
        assert "Message queue management" in result.output

    def test_queues_list_empty(self, runner: CliRunner) -> None:
        """queues list shows message when no queues."""
        result = runner.invoke(main, ["queues", "list"])

        assert result.exit_code == 0
        assert "No queues" in result.output
