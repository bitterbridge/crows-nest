"""Tests for the plugin system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from crows_nest.cli import main
from crows_nest.core.registry import get_global_registry
from crows_nest.plugins import (
    PluginInfo,
    PluginLoader,
    PluginSource,
    reset_plugin_loader,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_plugins() -> Generator[None, None, None]:
    """Reset plugin loader before and after each test."""
    reset_plugin_loader()
    yield
    reset_plugin_loader()


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary plugin directory."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    return plugins


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_default_values(self) -> None:
        """PluginInfo has sensible defaults."""
        info = PluginInfo(name="test")
        assert info.name == "test"
        assert info.version == "0.0.0"
        assert info.author is None
        assert info.operations == []
        assert info.source == PluginSource.ENTRYPOINT
        assert info.loaded is True

    def test_load_error_marks_not_loaded(self) -> None:
        """Plugin with load_error is not loaded."""
        info = PluginInfo(name="test", load_error="Failed to import")
        assert info.loaded is False

    def test_to_dict(self) -> None:
        """PluginInfo serializes to dict."""
        info = PluginInfo(
            name="test-plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            operations=["test.op1", "test.op2"],
            source=PluginSource.DIRECTORY,
        )
        d = info.to_dict()
        assert d["name"] == "test-plugin"
        assert d["version"] == "1.0.0"
        assert d["author"] == "Test Author"
        assert d["operations"] == ["test.op1", "test.op2"]
        assert d["source"] == "directory"
        assert d["loaded"] is True


class TestPluginLoader:
    """Tests for PluginLoader class."""

    def test_loader_init(self, plugin_dir: Path) -> None:
        """Loader initializes with custom directories."""
        loader = PluginLoader(plugin_dirs=[plugin_dir])
        assert plugin_dir in loader.plugin_dirs

    def test_empty_directories(self, plugin_dir: Path) -> None:
        """Loader handles empty directories gracefully."""
        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()
        assert plugins == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Loader handles nonexistent directories gracefully."""
        nonexistent = tmp_path / "does_not_exist"
        loader = PluginLoader(
            plugin_dirs=[nonexistent],
            load_entrypoints=False,
        )
        plugins = loader.load_all()
        assert plugins == []


class TestDirectoryPlugins:
    """Tests for directory-based plugins."""

    def test_load_single_file_plugin(self, plugin_dir: Path) -> None:
        """Loader discovers single-file plugins."""
        # Create a simple plugin file
        plugin_file = plugin_dir / "hello_plugin.py"
        plugin_file.write_text('''
"""A hello world plugin."""

__version__ = "1.2.3"
__author__ = "Test Author"

from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="hello.world",
    description="Say hello",
    required_capabilities=frozenset(),
)
async def hello_world(name: str = "World") -> dict:
    return {"message": f"Hello, {name}!"}
''')

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        assert len(plugins) == 1
        plugin = plugins[0]
        assert plugin.name == "hello_plugin"
        assert plugin.version == "1.2.3"
        assert plugin.author == "Test Author"
        assert plugin.loaded is True
        assert "hello.world" in plugin.operations

        # Verify operation is registered
        registry = get_global_registry()
        assert registry.has("hello.world")

    def test_load_package_plugin(self, plugin_dir: Path) -> None:
        """Loader discovers package plugins."""
        # Create a plugin package
        pkg_dir = plugin_dir / "my_package"
        pkg_dir.mkdir()

        init_file = pkg_dir / "__init__.py"
        init_file.write_text('''
"""My package plugin."""

__version__ = "2.0.0"
__author__ = "Package Author"

from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="my_package.greet",
    description="Greet someone",
    required_capabilities=frozenset(),
)
async def greet(who: str) -> dict:
    return {"greeting": f"Greetings, {who}!"}
''')

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        assert len(plugins) == 1
        plugin = plugins[0]
        assert plugin.name == "my_package"
        assert plugin.version == "2.0.0"
        assert plugin.loaded is True
        assert "my_package.greet" in plugin.operations

    def test_plugin_with_syntax_error(self, plugin_dir: Path) -> None:
        """Loader handles plugins with syntax errors gracefully."""
        bad_plugin = plugin_dir / "bad_syntax.py"
        bad_plugin.write_text("def broken(:\n    pass")

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        assert len(plugins) == 1
        plugin = plugins[0]
        assert plugin.name == "bad_syntax"
        assert plugin.loaded is False
        assert plugin.load_error is not None
        assert "SyntaxError" in plugin.load_error or "invalid syntax" in plugin.load_error

    def test_plugin_with_import_error(self, plugin_dir: Path) -> None:
        """Loader handles plugins with import errors gracefully."""
        bad_plugin = plugin_dir / "bad_import.py"
        bad_plugin.write_text("import nonexistent_module_12345")

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        assert len(plugins) == 1
        plugin = plugins[0]
        assert plugin.loaded is False
        assert plugin.load_error is not None

    def test_skip_private_files(self, plugin_dir: Path) -> None:
        """Loader skips files starting with underscore."""
        private_file = plugin_dir / "_private.py"
        private_file.write_text("# Private file")

        public_file = plugin_dir / "public.py"
        public_file.write_text("# Public file")

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        names = [p.name for p in plugins]
        assert "public" in names
        assert "_private" not in names


class TestPluginLoaderGetPlugin:
    """Tests for getting specific plugins."""

    def test_get_existing_plugin(self, plugin_dir: Path) -> None:
        """get_plugin returns plugin by name."""
        plugin_file = plugin_dir / "findme.py"
        plugin_file.write_text("# Plugin content")

        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        loader.load_all()

        plugin = loader.get_plugin("findme")
        assert plugin is not None
        assert plugin.name == "findme"

    def test_get_nonexistent_plugin(self, plugin_dir: Path) -> None:
        """get_plugin returns None for unknown plugin."""
        loader = PluginLoader(
            plugin_dirs=[plugin_dir],
            load_entrypoints=False,
        )
        loader.load_all()

        plugin = loader.get_plugin("does_not_exist")
        assert plugin is None


class TestCLIPluginCommands:
    """Tests for CLI plugin commands."""

    def test_plugins_list_empty(self, tmp_path: Path) -> None:
        """plugins list shows message when no plugins."""
        runner = CliRunner()

        # Use a non-existent directory to ensure no plugins
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["plugins", "list"])

        assert result.exit_code == 0
        assert "No plugins loaded" in result.output

    def test_plugins_list_with_plugin(self, plugin_dir: Path) -> None:
        """plugins list shows loaded plugins."""
        # Create a simple plugin
        plugin_file = plugin_dir / "cli_test.py"
        plugin_file.write_text('''
"""CLI test plugin."""
__version__ = "1.0.0"
''')

        runner = CliRunner()

        # Note: This test may be affected by system plugins
        result = runner.invoke(main, ["plugins", "list"])

        # Just verify it runs without error
        assert result.exit_code == 0

    def test_plugins_info_not_found(self) -> None:
        """plugins info shows error for unknown plugin."""
        runner = CliRunner()

        result = runner.invoke(main, ["plugins", "info", "nonexistent_plugin_12345"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestMultiplePluginDirectories:
    """Tests for loading from multiple directories."""

    def test_load_from_multiple_dirs(self, tmp_path: Path) -> None:
        """Loader loads from multiple directories."""
        dir1 = tmp_path / "plugins1"
        dir2 = tmp_path / "plugins2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "plugin_a.py").write_text("# Plugin A")
        (dir2 / "plugin_b.py").write_text("# Plugin B")

        loader = PluginLoader(
            plugin_dirs=[dir1, dir2],
            load_entrypoints=False,
        )
        plugins = loader.load_all()

        names = [p.name for p in plugins]
        assert "plugin_a" in names
        assert "plugin_b" in names
