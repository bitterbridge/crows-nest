"""Tests for plugin sandbox execution and verification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crows_nest.plugins.artifacts import (
    PluginArtifact,
    RiskLevel,
    Severity,
    ViolationType,
)
from crows_nest.plugins.sandbox import (
    PluginVerifier,
    Sandbox,
    SandboxConfig,
    StaticAnalyzer,
)
from crows_nest.plugins.sandbox import TestCase as SandboxTestCase
from crows_nest.plugins.storage import reset_artifact_storage

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_storage() -> Generator[None, None, None]:
    """Reset artifact storage before and after each test."""
    reset_artifact_storage()
    yield
    reset_artifact_storage()


def make_artifact(source_code: str, name: str = "test_plugin") -> PluginArtifact:
    """Helper to create a plugin artifact."""
    return PluginArtifact(
        id=uuid4(),
        name=name,
        description="Test plugin",
        source_code=source_code,
        operations=(),
        requested_capabilities=frozenset(),
        generated_by=uuid4(),
        generated_at=datetime.now(tz=UTC),
        generation_context={},
    )


class TestStaticAnalyzer:
    """Tests for static code analysis."""

    def test_detect_eval(self) -> None:
        """Analyzer detects eval() usage."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("result = eval('1 + 1')")

        assert len(violations) >= 1
        assert any(v.type == ViolationType.CODE for v in violations)
        assert any("eval" in v.description.lower() for v in violations)

    def test_detect_exec(self) -> None:
        """Analyzer detects exec() usage."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("exec('print(1)')")

        assert len(violations) >= 1
        assert any("exec" in v.description.lower() for v in violations)

    def test_detect_network_imports(self) -> None:
        """Analyzer detects network library imports."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("import socket")

        assert len(violations) >= 1
        assert any(v.type == ViolationType.NETWORK for v in violations)

    def test_detect_disallowed_import(self) -> None:
        """Analyzer detects disallowed module imports."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("import os")

        assert len(violations) >= 1
        assert any("os" in v.description for v in violations)

    def test_allow_safe_imports(self) -> None:
        """Analyzer allows safe standard library imports."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("""
import json
import re
from datetime import datetime
from collections import defaultdict
""")

        # Should have no import violations
        import_violations = [v for v in violations if "import" in v.description.lower()]
        assert len(import_violations) == 0

    def test_detect_syntax_error(self) -> None:
        """Analyzer detects syntax errors."""
        analyzer = StaticAnalyzer(SandboxConfig())
        violations = analyzer.analyze("def broken(:\n    pass")

        assert len(violations) >= 1
        assert any(v.severity == Severity.CRITICAL for v in violations)
        assert any("syntax" in v.description.lower() for v in violations)

    def test_allow_file_io_when_configured(self) -> None:
        """Analyzer allows file I/O when configured."""
        config = SandboxConfig(allow_file_io=True)
        analyzer = StaticAnalyzer(config)
        violations = analyzer.analyze("f = open('test.txt')")

        # open() should not be flagged when file I/O is allowed
        file_violations = [v for v in violations if v.type == ViolationType.FILESYSTEM]
        assert len(file_violations) == 0


class TestSandbox:
    """Tests for sandbox execution."""

    def test_execute_simple_code(self) -> None:
        """Sandbox executes simple code successfully."""
        sandbox = Sandbox()
        result = sandbox.execute("x = 1 + 1")

        assert result.success is True
        assert result.error is None
        assert result.namespace.get("x") == 2

    def test_execute_with_print(self) -> None:
        """Sandbox captures stdout."""
        sandbox = Sandbox()
        result = sandbox.execute("print('hello')")

        assert result.success is True
        assert "hello" in result.stdout

    def test_execute_code_with_import(self) -> None:
        """Sandbox allows safe imports."""
        sandbox = Sandbox()
        result = sandbox.execute("""
import json
data = json.loads('{"key": "value"}')
""")

        assert result.success is True
        assert result.namespace.get("data") == {"key": "value"}

    def test_execute_dangerous_code_fails_static_analysis(self) -> None:
        """Sandbox rejects code with critical violations."""
        sandbox = Sandbox()
        result = sandbox.execute("def broken(:\n    pass")

        assert result.success is False
        assert any(v.severity == Severity.CRITICAL for v in result.violations)

    def test_execute_tracks_duration(self) -> None:
        """Sandbox tracks execution duration."""
        sandbox = Sandbox()
        result = sandbox.execute("x = sum(range(1000))")

        assert result.success is True
        assert result.duration_ms >= 0

    def test_execute_handles_runtime_error(self) -> None:
        """Sandbox handles runtime errors gracefully."""
        sandbox = Sandbox()
        result = sandbox.execute("x = 1 / 0")

        assert result.success is False
        assert "ZeroDivisionError" in (result.error or "")


class TestPluginVerifier:
    """Tests for plugin verification."""

    def test_verify_valid_plugin(self) -> None:
        """Verifier passes valid plugin code."""
        source = '''
"""Simple test plugin."""
def add(a, b):
    return a + b

result = add(1, 2)
'''
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert report.passed is True
        assert report.loaded_successfully is True
        assert report.risk_level == RiskLevel.LOW

    def test_verify_plugin_with_operations(self) -> None:
        """Verifier detects registered operations."""
        source = '''
"""Plugin with operations."""
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="test.greet",
    description="Say hello",
    required_capabilities=frozenset(),
)
async def greet(name: str = "World") -> dict:
    return {"message": f"Hello, {name}!"}
'''
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert report.loaded_successfully is True
        assert "test.greet" in report.operations_registered

    def test_verify_plugin_with_syntax_error(self) -> None:
        """Verifier fails plugin with syntax error."""
        source = "def broken(:\n    pass"
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert report.passed is False
        assert report.loaded_successfully is False
        assert report.risk_level == RiskLevel.CRITICAL

    def test_verify_plugin_with_dangerous_code(self) -> None:
        """Verifier flags dangerous code patterns."""
        source = '''
"""Dangerous plugin."""
import socket  # Network access
'''
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert len(report.violations) > 0
        assert any(v.type == ViolationType.NETWORK for v in report.violations)

    def test_verify_with_test_cases(self) -> None:
        """Verifier runs test cases."""
        source = '''
"""Plugin with testable function."""
def multiply(a, b):
    return a * b
'''
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        test_cases = [
            SandboxTestCase(
                name="test_multiply_basic",
                operation="multiply",
                input_data={"a": 3, "b": 4},
                expected_output=12,
            ),
            SandboxTestCase(
                name="test_multiply_zero",
                operation="multiply",
                input_data={"a": 5, "b": 0},
                expected_output=0,
            ),
        ]

        report = verifier.verify(artifact, test_cases)

        assert len(report.test_results) == 2
        assert all(t.passed for t in report.test_results)

    def test_verify_with_failing_test(self) -> None:
        """Verifier reports failing tests."""
        source = '''
"""Plugin with buggy function."""
def add(a, b):
    return a - b  # Bug: should be +
'''
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        test_cases = [
            SandboxTestCase(
                name="test_add",
                operation="add",
                input_data={"a": 2, "b": 3},
                expected_output=5,
            ),
        ]

        report = verifier.verify(artifact, test_cases)

        assert report.passed is False
        assert len(report.test_results) == 1
        assert report.test_results[0].passed is False
        assert report.test_results[0].actual_output == -1  # 2 - 3

    def test_verify_records_artifact_hash(self) -> None:
        """Verifier records the artifact hash."""
        source = "x = 1"
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert report.artifact_hash == artifact.content_hash()
        assert report.artifact_id == artifact.id

    def test_verify_generates_summary(self) -> None:
        """Verifier generates human-readable summary."""
        source = "x = 1"
        artifact = make_artifact(source)
        verifier = PluginVerifier()

        report = verifier.verify(artifact)

        assert report.summary is not None
        assert len(report.summary) > 0


class TestSandboxConfig:
    """Tests for sandbox configuration."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = SandboxConfig()

        assert config.timeout_seconds == 30.0
        assert config.allow_file_io is False
        assert config.allow_network is False
        assert len(config.allowed_imports) > 0

    def test_custom_timeout(self) -> None:
        """Config accepts custom timeout."""
        config = SandboxConfig(timeout_seconds=5.0)
        assert config.timeout_seconds == 5.0

    def test_allow_file_io(self) -> None:
        """Config can enable file I/O."""
        config = SandboxConfig(allow_file_io=True)
        assert config.allow_file_io is True
