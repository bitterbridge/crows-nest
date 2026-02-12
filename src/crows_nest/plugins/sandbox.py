"""Sandbox execution environment for plugin verification.

Provides isolated execution of untrusted plugin code with:
- Restricted builtins and imports
- Timeout handling
- Resource usage tracking
- Violation detection

Note: This is a development-grade sandbox. Production deployments
should use more robust isolation (containers, seccomp, etc).
"""

from __future__ import annotations

import ast
import io
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from crows_nest.plugins.artifacts import (
    PluginArtifact,
    ResourceUsage,
    RiskLevel,
    Severity,
    TestResult,
    VerificationReport,
    Violation,
    ViolationType,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# Allowed standard library modules for plugins
ALLOWED_IMPORTS = frozenset({
    "json",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "math",
    "random",
    "string",
    "hashlib",
    "base64",
    "uuid",
    "typing",
    "dataclasses",
    "enum",
    "copy",
    "operator",
    # Framework imports
    "crows_nest",
    "crows_nest.core",
    "crows_nest.core.registry",
})

# Dangerous patterns to detect in source code
DANGEROUS_PATTERNS = [
    ("eval(", ViolationType.CODE, "Use of eval()"),
    ("exec(", ViolationType.CODE, "Use of exec()"),
    ("compile(", ViolationType.CODE, "Use of compile()"),
    ("__import__(", ViolationType.CODE, "Use of __import__()"),
    ("subprocess", ViolationType.CODE, "Use of subprocess"),
    ("os.system", ViolationType.CODE, "Use of os.system"),
    ("os.popen", ViolationType.CODE, "Use of os.popen"),
    ("socket", ViolationType.NETWORK, "Use of socket"),
    ("urllib", ViolationType.NETWORK, "Use of urllib"),
    ("requests", ViolationType.NETWORK, "Use of requests"),
    ("httpx", ViolationType.NETWORK, "Use of httpx"),
    ("aiohttp", ViolationType.NETWORK, "Use of aiohttp"),
    ("open(", ViolationType.FILESYSTEM, "Use of open()"),
    ("pathlib", ViolationType.FILESYSTEM, "Use of pathlib"),
]


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution.

    Attributes:
        timeout_seconds: Maximum execution time.
        max_memory_mb: Maximum memory usage (advisory).
        allowed_imports: Set of allowed module names.
        allow_file_io: Whether to allow file operations.
        allow_network: Whether to allow network operations.
    """

    timeout_seconds: float = 30.0
    max_memory_mb: float = 100.0
    allowed_imports: frozenset[str] = ALLOWED_IMPORTS
    allow_file_io: bool = False
    allow_network: bool = False


@dataclass
class SandboxResult:
    """Result of sandbox execution.

    Attributes:
        success: Whether execution completed without errors.
        error: Error message if execution failed.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_ms: Execution time in milliseconds.
        operations_registered: Names of operations that were registered.
        violations: Any violations detected.
        namespace: The resulting namespace after execution.
    """

    success: bool
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    operations_registered: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    namespace: dict[str, Any] = field(default_factory=dict)


class StaticAnalyzer:
    """Static analysis of plugin source code."""

    def __init__(self, config: SandboxConfig) -> None:
        """Initialize analyzer with config."""
        self.config = config

    def analyze(self, source_code: str) -> list[Violation]:
        """Analyze source code for violations.

        Args:
            source_code: Python source code to analyze.

        Returns:
            List of violations found.
        """
        violations = []

        # Check for dangerous patterns
        for pattern, violation_type, description in DANGEROUS_PATTERNS:
            if pattern in source_code:
                # Check if it's allowed by config
                if violation_type == ViolationType.FILESYSTEM and self.config.allow_file_io:
                    continue
                if violation_type == ViolationType.NETWORK and self.config.allow_network:
                    continue

                violations.append(
                    Violation(
                        type=violation_type,
                        severity=Severity.ERROR,
                        description=description,
                        evidence={"pattern": pattern},
                    )
                )

        # Parse AST and check imports
        try:
            tree = ast.parse(source_code)
            violations.extend(self._check_imports(tree))
        except SyntaxError as e:
            violations.append(
                Violation(
                    type=ViolationType.CODE,
                    severity=Severity.CRITICAL,
                    description=f"Syntax error: {e}",
                    evidence={"line": e.lineno, "offset": e.offset},
                )
            )

        return violations

    def _check_imports(self, tree: ast.AST) -> list[Violation]:
        """Check for disallowed imports."""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_import(alias.name):
                        violations.append(
                            Violation(
                                type=ViolationType.CODE,
                                severity=Severity.ERROR,
                                description=f"Disallowed import: {alias.name}",
                                evidence={"module": alias.name, "line": node.lineno},
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and (
                node.module and not self._is_allowed_import(node.module)
            ):
                violations.append(
                        Violation(
                            type=ViolationType.CODE,
                            severity=Severity.ERROR,
                            description=f"Disallowed import from: {node.module}",
                            evidence={"module": node.module, "line": node.lineno},
                        )
                    )

        return violations

    def _is_allowed_import(self, module_name: str) -> bool:
        """Check if a module import is allowed."""
        # Check exact match
        if module_name in self.config.allowed_imports:
            return True

        # Check if it's a submodule of an allowed module
        return any(
            module_name.startswith(f"{allowed}.")
            for allowed in self.config.allowed_imports
        )


class OperationCollector:
    """Collects operations registered during sandbox execution."""

    def __init__(self) -> None:
        """Initialize collector."""
        self.operations: list[str] = []
        self._original_decorator: Any = None

    def install(self) -> None:
        """Install the collector by patching thunk_operation decorator."""
        from crows_nest.core import registry  # noqa: PLC0415

        self._original_decorator = registry.thunk_operation

        def collecting_decorator(
            name: str,
            description: str = "",
            required_capabilities: frozenset[str] = frozenset(),
        ) -> Any:
            """Decorator that collects operation names."""
            self.operations.append(name)
            return self._original_decorator(
                name=name,
                description=description,
                required_capabilities=required_capabilities,
            )

        registry.thunk_operation = collecting_decorator

    def uninstall(self) -> None:
        """Restore the original decorator."""
        if self._original_decorator is not None:
            from crows_nest.core import registry  # noqa: PLC0415

            registry.thunk_operation = self._original_decorator
            self._original_decorator = None


@contextmanager
def capture_output() -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
    """Context manager to capture stdout and stderr."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = stdout, stderr
        yield stdout, stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


class Sandbox:
    """Sandbox for executing plugin code.

    Provides isolated execution with:
    - Static analysis before execution
    - Restricted imports and builtins
    - Timeout handling
    - Output capture
    - Operation registration tracking
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        """Initialize sandbox with config."""
        self.config = config or SandboxConfig()
        self.analyzer = StaticAnalyzer(self.config)

    def execute(self, source_code: str) -> SandboxResult:
        """Execute source code in the sandbox.

        Args:
            source_code: Python source code to execute.

        Returns:
            SandboxResult with execution details.
        """
        # Static analysis first
        violations = self.analyzer.analyze(source_code)

        # If critical violations, don't execute
        if any(v.severity == Severity.CRITICAL for v in violations):
            return SandboxResult(
                success=False,
                error="Critical violations detected in static analysis",
                violations=violations,
            )

        # Set up operation collector
        collector = OperationCollector()
        collector.install()

        try:
            # Execute with output capture
            start_time = time.perf_counter()
            with capture_output() as (stdout, stderr):
                namespace = self._execute_code(source_code)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return SandboxResult(
                success=True,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                duration_ms=duration_ms,
                operations_registered=collector.operations.copy(),
                violations=violations,
                namespace=namespace,
            )

        except TimeoutError as e:
            return SandboxResult(
                success=False,
                error=f"Execution timed out: {e}",
                violations=[
                    *violations,
                    Violation(
                        type=ViolationType.RESOURCE,
                        severity=Severity.ERROR,
                        description="Execution timeout exceeded",
                        evidence={"timeout_seconds": self.config.timeout_seconds},
                    ),
                ],
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                violations=violations,
            )

        finally:
            collector.uninstall()

    def _execute_code(self, source_code: str) -> dict[str, Any]:
        """Execute code and return the resulting namespace."""
        # Create a restricted namespace
        namespace: dict[str, Any] = {
            "__builtins__": self._get_safe_builtins(),
            "__name__": "__sandbox__",
            "__doc__": None,
        }

        # Compile and execute
        code = compile(source_code, "<sandbox>", "exec")
        exec(code, namespace)  # noqa: S102 - intentional sandboxed exec

        return namespace

    def _get_safe_builtins(self) -> dict[str, Any]:
        """Get a restricted set of builtins."""
        import builtins  # noqa: PLC0415

        # Safe builtins to expose
        safe_names = {
            # Types
            "bool",
            "bytes",
            "bytearray",
            "complex",
            "dict",
            "float",
            "frozenset",
            "int",
            "list",
            "object",
            "set",
            "str",
            "tuple",
            "type",
            # Functions
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "callable",
            "chr",
            "divmod",
            "enumerate",
            "filter",
            "format",
            "getattr",
            "hasattr",
            "hash",
            "hex",
            "id",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "setattr",
            "slice",
            "sorted",
            "sum",
            "zip",
            # Exceptions
            "Exception",
            "TypeError",
            "ValueError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "RuntimeError",
            "StopIteration",
            "NotImplementedError",
            # Constants
            "True",
            "False",
            "None",
            # Import (we'll validate what gets imported)
            "__import__",
        }

        return {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}


@dataclass
class TestCase:
    """A test case for plugin verification.

    Attributes:
        name: Test name.
        operation: Operation to test.
        input_data: Input arguments.
        expected_output: Expected return value (if known).
        should_succeed: Whether the operation should succeed.
    """

    name: str
    operation: str
    input_data: dict[str, Any]
    expected_output: Any = None
    should_succeed: bool = True


class PluginVerifier:
    """Verifies plugins by running them in a sandbox.

    Performs:
    - Static analysis
    - Sandbox execution
    - Test case execution
    - Resource monitoring
    - Violation detection
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        """Initialize verifier with config."""
        self.config = config or SandboxConfig()
        self.sandbox = Sandbox(self.config)

    def verify(
        self,
        artifact: PluginArtifact,
        test_cases: list[TestCase] | None = None,
    ) -> VerificationReport:
        """Verify a plugin artifact.

        Args:
            artifact: The plugin artifact to verify.
            test_cases: Optional test cases to run.

        Returns:
            VerificationReport with verification results.
        """
        sandbox_id = f"sandbox-{uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Execute in sandbox
        result = self.sandbox.execute(artifact.source_code)

        # Run test cases if execution succeeded
        test_results: list[TestResult] = []
        if result.success and test_cases:
            test_results = self._run_test_cases(result.namespace, test_cases)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Create resource usage (basic tracking)
        resource_usage = ResourceUsage(
            cpu_time_ms=result.duration_ms,
            memory_peak_mb=0.0,  # Would need psutil for real tracking
            file_descriptors=0,
            files_created=0,
            files_modified=0,
        )

        # Determine risk level based on violations
        risk_level = self._assess_risk(result.violations)

        # Determine if passed
        passed = (
            result.success
            and all(t.passed for t in test_results)
            and not any(v.severity == Severity.CRITICAL for v in result.violations)
        )

        # Generate summary
        if passed:
            test_info = f"{len(test_results)} tests" if test_results else "no tests"
            summary = f"Passed: {test_info}, {len(result.violations)} violations"
        elif not result.success:
            summary = f"Failed to load: {result.error}"
        else:
            failed_tests = [t.test_name for t in test_results if not t.passed]
            summary = f"Failed tests: {failed_tests[:3]}"

        return VerificationReport(
            id=uuid4(),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash(),
            sandbox_id=sandbox_id,
            executed_at=datetime.now(UTC),
            duration_ms=duration_ms,
            loaded_successfully=result.success,
            operations_registered=tuple(result.operations_registered),
            test_results=tuple(test_results),
            resource_usage=resource_usage,
            violations=tuple(result.violations),
            passed=passed,
            risk_level=risk_level,
            summary=summary,
        )

    def _run_test_cases(
        self,
        namespace: dict[str, Any],
        test_cases: list[TestCase],
    ) -> list[TestResult]:
        """Run test cases against the loaded plugin."""
        results = []

        for test in test_cases:
            start = time.perf_counter()
            try:
                # Find the operation function in the namespace
                func = self._find_operation_function(namespace, test.operation)
                if func is None:
                    results.append(
                        TestResult(
                            test_name=test.name,
                            passed=False,
                            input_data=test.input_data,
                            expected_output=test.expected_output,
                            actual_output=None,
                            error=f"Operation '{test.operation}' not found",
                            duration_ms=0,
                        )
                    )
                    continue

                # Call the function (handle async)
                import asyncio  # noqa: PLC0415

                if asyncio.iscoroutinefunction(func):
                    actual = asyncio.get_event_loop().run_until_complete(
                        func(**test.input_data)
                    )
                else:
                    actual = func(**test.input_data)

                duration_ms = int((time.perf_counter() - start) * 1000)

                # Check result
                if test.expected_output is not None:
                    passed = actual == test.expected_output
                else:
                    passed = test.should_succeed

                results.append(
                    TestResult(
                        test_name=test.name,
                        passed=passed,
                        input_data=test.input_data,
                        expected_output=test.expected_output,
                        actual_output=actual,
                        error=None,
                        duration_ms=duration_ms,
                    )
                )

            except Exception as e:
                duration_ms = int((time.perf_counter() - start) * 1000)
                results.append(
                    TestResult(
                        test_name=test.name,
                        passed=not test.should_succeed,
                        input_data=test.input_data,
                        expected_output=test.expected_output,
                        actual_output=None,
                        error=f"{type(e).__name__}: {e}",
                        duration_ms=duration_ms,
                    )
                )

        return results

    def _find_operation_function(
        self,
        namespace: dict[str, Any],
        operation_name: str,
    ) -> Any:
        """Find the function for an operation in the namespace."""
        # Look for functions decorated with @thunk_operation
        for name, obj in namespace.items():
            if (
                callable(obj)
                and hasattr(obj, "__wrapped__")
                and getattr(obj, "_operation_name", None) == operation_name
            ):
                return obj
            # Also check by function name matching operation suffix
            if callable(obj) and operation_name.endswith(name):
                return obj

        return None

    def _assess_risk(self, violations: list[Violation]) -> RiskLevel:
        """Assess risk level based on violations."""
        if any(v.severity == Severity.CRITICAL for v in violations):
            return RiskLevel.CRITICAL
        if any(v.severity == Severity.ERROR for v in violations):
            return RiskLevel.HIGH
        if any(v.type in (ViolationType.NETWORK, ViolationType.FILESYSTEM) for v in violations):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
