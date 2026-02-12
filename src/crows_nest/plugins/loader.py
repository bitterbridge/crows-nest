"""Plugin discovery and loading system.

Supports two plugin sources:
1. Entry points: Installed packages with `crows_nest.operations` entry point
2. Directory: Python files in `~/.crows-nest/plugins/`

Plugins register operations with the global registry on load.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from crows_nest.core.registry import get_global_registry

# Entry point group name for plugin discovery
ENTRY_POINT_GROUP = "crows_nest.operations"

# Default plugin directory
DEFAULT_PLUGIN_DIR = Path.home() / ".crows-nest" / "plugins"


class PluginSource(StrEnum):
    """Source of a discovered plugin."""

    ENTRYPOINT = "entrypoint"
    DIRECTORY = "directory"


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    def __init__(self, plugin_name: str, message: str) -> None:
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}': {message}")


class PluginLoadError(PluginError):
    """Error loading a plugin."""


class PluginNotFoundError(PluginError):
    """Plugin not found."""


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    Attributes:
        name: Plugin name (package name or file name).
        version: Plugin version string.
        author: Optional author name.
        description: Optional plugin description.
        operations: List of operation names registered by this plugin.
        source: Where the plugin was discovered.
        module_name: The Python module name.
        load_error: Error message if plugin failed to load.
    """

    name: str
    version: str = "0.0.0"
    author: str | None = None
    description: str | None = None
    operations: list[str] = field(default_factory=list)
    source: PluginSource = PluginSource.ENTRYPOINT
    module_name: str | None = None
    load_error: str | None = None

    @property
    def loaded(self) -> bool:
        """Whether the plugin loaded successfully."""
        return self.load_error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "operations": self.operations,
            "source": self.source.value,
            "module_name": self.module_name,
            "loaded": self.loaded,
            "load_error": self.load_error,
        }


class PluginLoader:
    """Discovers and loads plugins from entry points and directories.

    Usage:
        loader = PluginLoader()
        plugins = loader.load_all()
        for plugin in plugins:
            print(f"Loaded: {plugin.name} with {len(plugin.operations)} operations")
    """

    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        load_entrypoints: bool = True,
        load_directories: bool = True,
    ) -> None:
        """Initialize the plugin loader.

        Args:
            plugin_dirs: Custom plugin directories to scan.
                Defaults to ~/.crows-nest/plugins/
            load_entrypoints: Whether to load from entry points.
            load_directories: Whether to load from directories.
        """
        self.plugin_dirs = plugin_dirs or [DEFAULT_PLUGIN_DIR]
        self.load_entrypoints = load_entrypoints
        self.load_directories = load_directories
        self._plugins: dict[str, PluginInfo] = {}

    @property
    def plugins(self) -> list[PluginInfo]:
        """Get all discovered plugins."""
        return list(self._plugins.values())

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def load_all(self) -> list[PluginInfo]:
        """Discover and load all plugins.

        Returns:
            List of PluginInfo for all discovered plugins.
        """
        self._plugins.clear()

        if self.load_entrypoints:
            self._discover_entrypoints()

        if self.load_directories:
            for plugin_dir in self.plugin_dirs:
                self._discover_directory(plugin_dir)

        return self.plugins

    def _discover_entrypoints(self) -> None:
        """Discover plugins via entry points."""
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)

        for ep in entry_points:
            plugin_info = self._load_entrypoint(ep)
            self._plugins[plugin_info.name] = plugin_info

    def _load_entrypoint(self, ep: importlib.metadata.EntryPoint) -> PluginInfo:
        """Load a single entry point plugin.

        Args:
            ep: The entry point to load.

        Returns:
            PluginInfo for the plugin (may have load_error set).
        """
        name = ep.name

        # Get package metadata
        version = "0.0.0"
        author = None
        description = None

        try:
            # Try to get distribution metadata
            dist_name = ep.value.split(":")[0].split(".")[0]
            try:
                dist = importlib.metadata.distribution(dist_name)
                metadata = dist.metadata
                version = metadata.get("Version", "0.0.0")
                author = metadata.get("Author")
                description = metadata.get("Summary")
            except importlib.metadata.PackageNotFoundError:
                pass

            # Load the register function
            register_func = ep.load()

            # Track operations before and after
            registry = get_global_registry()
            ops_before = set(registry.list_operations())

            # Call register function
            register_func()

            # Find new operations
            ops_after = set(registry.list_operations())
            new_ops = list(ops_after - ops_before)

            return PluginInfo(
                name=name,
                version=version,
                author=author,
                description=description,
                operations=new_ops,
                source=PluginSource.ENTRYPOINT,
                module_name=ep.value,
            )

        except Exception as e:
            return PluginInfo(
                name=name,
                version=version,
                author=author,
                description=description,
                source=PluginSource.ENTRYPOINT,
                module_name=ep.value if hasattr(ep, "value") else None,
                load_error=str(e),
            )

    def _discover_directory(self, plugin_dir: Path) -> None:
        """Discover plugins in a directory.

        Args:
            plugin_dir: Directory to scan for plugins.
        """
        if not plugin_dir.exists():
            return

        # Look for Python files and packages
        for item in plugin_dir.iterdir():
            if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                plugin_info = self._load_file_plugin(item)
                self._plugins[plugin_info.name] = plugin_info
            elif item.is_dir() and (item / "__init__.py").exists():
                plugin_info = self._load_package_plugin(item)
                self._plugins[plugin_info.name] = plugin_info

    def _load_file_plugin(self, plugin_file: Path) -> PluginInfo:
        """Load a single-file plugin.

        Args:
            plugin_file: Path to the plugin Python file.

        Returns:
            PluginInfo for the plugin.
        """
        name = plugin_file.stem
        module_name = f"crows_nest_plugin_{name}"

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                return PluginInfo(
                    name=name,
                    source=PluginSource.DIRECTORY,
                    load_error="Could not create module spec",
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            # Track operations
            registry = get_global_registry()
            ops_before = set(registry.list_operations())

            # Execute module (this registers operations)
            spec.loader.exec_module(module)

            # Find new operations
            ops_after = set(registry.list_operations())
            new_ops = list(ops_after - ops_before)

            # Extract metadata from module
            version = getattr(module, "__version__", "0.0.0")
            author = getattr(module, "__author__", None)
            description = getattr(module, "__doc__", None)
            if description:
                description = description.strip().split("\n")[0]

            return PluginInfo(
                name=name,
                version=version,
                author=author,
                description=description,
                operations=new_ops,
                source=PluginSource.DIRECTORY,
                module_name=module_name,
            )

        except Exception as e:
            return PluginInfo(
                name=name,
                source=PluginSource.DIRECTORY,
                module_name=module_name,
                load_error=str(e),
            )

    def _load_package_plugin(self, plugin_dir: Path) -> PluginInfo:
        """Load a package plugin.

        Args:
            plugin_dir: Path to the plugin package directory.

        Returns:
            PluginInfo for the plugin.
        """
        name = plugin_dir.name
        module_name = f"crows_nest_plugin_{name}"
        init_file = plugin_dir / "__init__.py"

        try:
            # Load the package
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_file,
                submodule_search_locations=[str(plugin_dir)],
            )
            if spec is None or spec.loader is None:
                return PluginInfo(
                    name=name,
                    source=PluginSource.DIRECTORY,
                    load_error="Could not create module spec",
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            # Track operations
            registry = get_global_registry()
            ops_before = set(registry.list_operations())

            # Execute module
            spec.loader.exec_module(module)

            # Find new operations
            ops_after = set(registry.list_operations())
            new_ops = list(ops_after - ops_before)

            # Extract metadata
            version = getattr(module, "__version__", "0.0.0")
            author = getattr(module, "__author__", None)
            description = getattr(module, "__doc__", None)
            if description:
                description = description.strip().split("\n")[0]

            return PluginInfo(
                name=name,
                version=version,
                author=author,
                description=description,
                operations=new_ops,
                source=PluginSource.DIRECTORY,
                module_name=module_name,
            )

        except Exception as e:
            return PluginInfo(
                name=name,
                source=PluginSource.DIRECTORY,
                module_name=module_name,
                load_error=str(e),
            )


# Global plugin loader instance
_loader: PluginLoader | None = None


def get_plugin_loader() -> PluginLoader:
    """Get or create the global plugin loader."""
    global _loader  # noqa: PLW0603
    if _loader is None:
        _loader = PluginLoader()
    return _loader


def load_plugins() -> list[PluginInfo]:
    """Load all plugins using the global loader.

    Returns:
        List of loaded plugins.
    """
    return get_plugin_loader().load_all()


def get_loaded_plugins() -> list[PluginInfo]:
    """Get all loaded plugins.

    Returns:
        List of PluginInfo for loaded plugins.
    """
    return get_plugin_loader().plugins


def reset_plugin_loader() -> None:
    """Reset the plugin loader. Useful for testing."""
    global _loader  # noqa: PLW0603
    _loader = None
