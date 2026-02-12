"""Configuration management for Crow's Nest.

Loads configuration from TOML files with environment variable overrides.
Uses pydantic-settings for validation and type safety.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneralSettings(BaseSettings):
    """General application settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_")

    log_level: str = Field(default="INFO", description="Log level")
    json_logs: bool = Field(default=False, description="Enable JSON structured logging")


class DatabaseSettings(BaseSettings):
    """Database settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_DATABASE_")

    path: str = Field(default="crows_nest.db", description="SQLite database path")


class APISettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_API_")

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_LLM_")

    provider: str = Field(default="mock", description="LLM provider: mock, anthropic, openai")
    model: str = Field(default="claude-sonnet-4-20250514", description="Model name")
    api_key: str | None = Field(default=None, description="API key for provider")
    max_tokens: int = Field(default=1000, description="Max tokens for responses")
    temperature: float = Field(default=0.7, description="Temperature for generation")


class TracingSettings(BaseSettings):
    """OpenTelemetry tracing settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_TRACING_")

    enabled: bool = Field(default=False, description="Enable tracing")
    otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP endpoint")
    service_name: str = Field(default="crows-nest", description="Service name")


class CapabilitiesSettings(BaseSettings):
    """Capability settings."""

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_CAPABILITIES_")

    root: list[str] = Field(
        default=[
            "shell.run",
            "shell.pipe",
            "file.read",
            "file.write",
            "agent.create",
            "agent.run",
            "agent.spawn",
        ],
        description="Root orchestrator capabilities",
    )


class Settings(BaseSettings):
    """Main application settings.

    Configuration is loaded in the following order (later overrides earlier):
    1. Default values in this class
    2. Values from config.toml (if exists)
    3. Environment variables (CROWS_NEST_* prefix)
    """

    model_config = SettingsConfigDict(env_prefix="CROWS_NEST_")

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    capabilities: CapabilitiesSettings = Field(default_factory=CapabilitiesSettings)

    @classmethod
    def from_toml(cls, path: Path | str) -> Settings:
        """Load settings from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            Settings instance with values from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Settings:
        """Create settings from a dictionary."""
        return cls(
            general=GeneralSettings(**data.get("general", {})),
            database=DatabaseSettings(**data.get("database", {})),
            api=APISettings(**data.get("api", {})),
            llm=LLMSettings(**data.get("llm", {})),
            tracing=TracingSettings(**data.get("tracing", {})),
            capabilities=CapabilitiesSettings(**data.get("capabilities", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary."""
        return {
            "general": self.general.model_dump(),
            "database": self.database.model_dump(),
            "api": self.api.model_dump(),
            "llm": {
                **self.llm.model_dump(),
                "api_key": "***" if self.llm.api_key else None,  # Mask API key
            },
            "tracing": self.tracing.model_dump(),
            "capabilities": self.capabilities.model_dump(),
        }


def _find_config_file() -> Path | None:
    """Find the config file in standard locations."""
    candidates = [
        Path("config.toml"),
        Path("crows_nest.toml"),
        Path.home() / ".config" / "crows-nest" / "config.toml",
        Path("/etc/crows-nest/config.toml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    """Get the application settings.

    Settings are loaded from:
    1. Default values
    2. Config file (if found or specified)
    3. Environment variables

    The result is cached after first call.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        The resolved Settings instance.
    """
    # Start with defaults
    settings = Settings()

    # Try to load from config file
    path = Path(config_path) if config_path else _find_config_file()
    if path and path.exists():
        settings = Settings.from_toml(path)

    # Environment variables are automatically loaded by pydantic-settings
    # and will override the values from the TOML file
    return settings


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()
