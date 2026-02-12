"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crows_nest.core.config import (
    APISettings,
    DatabaseSettings,
    GeneralSettings,
    LLMSettings,
    Settings,
    TracingSettings,
    clear_settings_cache,
    get_settings,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def clean_settings() -> Generator[None, None, None]:
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestDefaultSettings:
    """Tests for default configuration values."""

    def test_general_defaults(self) -> None:
        """General settings have correct defaults."""
        settings = GeneralSettings()
        assert settings.log_level == "INFO"
        assert settings.json_logs is False

    def test_database_defaults(self) -> None:
        """Database settings have correct defaults."""
        settings = DatabaseSettings()
        assert settings.path == "crows_nest.db"

    def test_api_defaults(self) -> None:
        """API settings have correct defaults."""
        settings = APISettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.cors_enabled is True
        assert settings.cors_origins == ["*"]

    def test_llm_defaults(self) -> None:
        """LLM settings have correct defaults."""
        settings = LLMSettings()
        assert settings.provider == "mock"
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.api_key is None
        assert settings.max_tokens == 1000
        assert settings.temperature == 0.7

    def test_tracing_defaults(self) -> None:
        """Tracing settings have correct defaults."""
        settings = TracingSettings()
        assert settings.enabled is False
        assert settings.otlp_endpoint == "http://localhost:4317"
        assert settings.service_name == "crows-nest"


class TestSettingsFromToml:
    """Tests for loading settings from TOML files."""

    def test_load_from_toml(self, tmp_path: Path) -> None:
        """Settings can be loaded from a TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[general]
log_level = "DEBUG"
json_logs = true

[database]
path = "/tmp/test.db"

[api]
host = "0.0.0.0"
port = 9000
cors_enabled = false

[llm]
provider = "anthropic"
model = "claude-3-opus"
max_tokens = 2000
temperature = 0.5

[tracing]
enabled = true
service_name = "test-service"
""")

        settings = Settings.from_toml(config_file)

        assert settings.general.log_level == "DEBUG"
        assert settings.general.json_logs is True
        assert settings.database.path == "/tmp/test.db"
        assert settings.api.host == "0.0.0.0"
        assert settings.api.port == 9000
        assert settings.api.cors_enabled is False
        assert settings.llm.provider == "anthropic"
        assert settings.llm.model == "claude-3-opus"
        assert settings.llm.max_tokens == 2000
        assert settings.llm.temperature == 0.5
        assert settings.tracing.enabled is True
        assert settings.tracing.service_name == "test-service"

    def test_load_partial_toml(self, tmp_path: Path) -> None:
        """Missing sections in TOML use defaults."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[general]
log_level = "WARNING"
""")

        settings = Settings.from_toml(config_file)

        # Specified value
        assert settings.general.log_level == "WARNING"
        # Defaults for unspecified
        assert settings.database.path == "crows_nest.db"
        assert settings.api.port == 8000

    def test_missing_toml_raises(self, tmp_path: Path) -> None:
        """Loading non-existent TOML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Settings.from_toml(tmp_path / "nonexistent.toml")


class TestSettingsEnvOverride:
    """Tests for environment variable overrides."""

    def test_env_overrides_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override general settings."""
        monkeypatch.setenv("CROWS_NEST_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("CROWS_NEST_JSON_LOGS", "true")

        settings = GeneralSettings()

        assert settings.log_level == "ERROR"
        assert settings.json_logs is True

    def test_env_overrides_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override database settings."""
        monkeypatch.setenv("CROWS_NEST_DATABASE_PATH", "/custom/path.db")

        settings = DatabaseSettings()

        assert settings.path == "/custom/path.db"

    def test_env_overrides_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override API settings."""
        monkeypatch.setenv("CROWS_NEST_API_HOST", "192.168.1.1")
        monkeypatch.setenv("CROWS_NEST_API_PORT", "3000")
        monkeypatch.setenv("CROWS_NEST_API_CORS_ENABLED", "false")

        settings = APISettings()

        assert settings.host == "192.168.1.1"
        assert settings.port == 3000
        assert settings.cors_enabled is False

    def test_env_overrides_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override LLM settings."""
        monkeypatch.setenv("CROWS_NEST_LLM_PROVIDER", "openai")
        monkeypatch.setenv("CROWS_NEST_LLM_API_KEY", "sk-test-key")
        monkeypatch.setenv("CROWS_NEST_LLM_MAX_TOKENS", "4000")

        settings = LLMSettings()

        assert settings.provider == "openai"
        assert settings.api_key == "sk-test-key"
        assert settings.max_tokens == 4000


class TestSettingsToDict:
    """Tests for settings serialization."""

    def test_to_dict(self) -> None:
        """Settings can be converted to a dictionary."""
        settings = Settings()
        result = settings.to_dict()

        assert "general" in result
        assert "database" in result
        assert "api" in result
        assert "llm" in result
        assert "tracing" in result
        assert "capabilities" in result

    def test_to_dict_masks_api_key(self) -> None:
        """API key is masked in dictionary output."""
        settings = Settings(llm=LLMSettings(api_key="secret-key"))
        result = settings.to_dict()

        assert result["llm"]["api_key"] == "***"

    def test_to_dict_shows_none_api_key(self) -> None:
        """None API key is shown as None."""
        settings = Settings()
        result = settings.to_dict()

        assert result["llm"]["api_key"] is None


class TestGetSettings:
    """Tests for the get_settings function."""

    def test_get_settings_default(self) -> None:
        """get_settings returns default settings when no config file."""
        settings = get_settings()

        assert settings.general.log_level == "INFO"
        assert settings.database.path == "crows_nest.db"

    def test_get_settings_with_path(self, tmp_path: Path) -> None:
        """get_settings loads from specified path."""
        config_file = tmp_path / "custom.toml"
        config_file.write_text("""
[general]
log_level = "CRITICAL"
""")

        settings = get_settings(str(config_file))

        assert settings.general.log_level == "CRITICAL"

    def test_get_settings_cached(self) -> None:
        """get_settings returns cached settings."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
