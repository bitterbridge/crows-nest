"""Security tests for Crow's Nest.

Tests for shell injection prevention, path traversal, rate limiting,
and input validation.
"""

from __future__ import annotations

import pytest

from crows_nest.shell.operations import (
    ALLOWED_COMMANDS,
    DANGEROUS_PATTERNS,
    validate_command,
    validate_working_dir,
)


class TestShellCommandSecurity:
    """Tests for shell command security."""

    def test_command_not_in_allowlist_rejected(self) -> None:
        """Commands not in allowlist are rejected."""
        error = validate_command("rm", ["-rf", "/"])
        assert error is not None
        assert error.type == "CommandNotAllowed"
        assert "rm" in error.message

    def test_allowed_command_accepted(self) -> None:
        """Commands in allowlist are accepted."""
        error = validate_command("ls", ["-la"])
        assert error is None

    def test_restricted_argument_rejected(self) -> None:
        """Restricted arguments for allowed commands are rejected."""
        # tar -c (create) is restricted
        error = validate_command("tar", ["-c", "archive.tar", "files/"])
        assert error is not None
        assert error.type == "ArgumentNotAllowed"

    def test_sed_inplace_rejected(self) -> None:
        """sed -i (in-place edit) is rejected."""
        error = validate_command("sed", ["-i", "s/foo/bar/", "file.txt"])
        assert error is not None
        assert error.type == "ArgumentNotAllowed"

    @pytest.mark.parametrize("pattern", list(DANGEROUS_PATTERNS))
    def test_dangerous_pattern_rejected(self, pattern: str) -> None:
        """Dangerous shell patterns in arguments are rejected."""
        error = validate_command("echo", [f"test{pattern}injection"])
        assert error is not None
        assert error.type == "DangerousPattern"

    def test_semicolon_injection_rejected(self) -> None:
        """Semicolon command chaining is rejected."""
        error = validate_command("echo", ["hello; rm -rf /"])
        assert error is not None
        assert ";" in error.message

    def test_pipe_injection_rejected(self) -> None:
        """Pipe injection is rejected."""
        error = validate_command("cat", ["file.txt | rm -rf /"])
        assert error is not None
        assert "|" in error.message

    def test_command_substitution_rejected(self) -> None:
        """Command substitution with $() is rejected."""
        error = validate_command("echo", ["$(rm -rf /)"])
        assert error is not None
        assert "$(" in error.message

    def test_backtick_substitution_rejected(self) -> None:
        """Command substitution with backticks is rejected."""
        error = validate_command("echo", ["`rm -rf /`"])
        assert error is not None
        assert "`" in error.message

    def test_redirect_injection_rejected(self) -> None:
        """Redirect operators are rejected."""
        error = validate_command("echo", ["data > /etc/passwd"])
        assert error is not None
        assert ">" in error.message

    def test_allowlist_readonly_commands(self) -> None:
        """Verify allowlist contains only read-only/safe commands."""
        dangerous_commands = {"rm", "mv", "cp", "dd", "mkfs", "chmod", "chown", "kill"}
        for cmd in dangerous_commands:
            assert cmd not in ALLOWED_COMMANDS, f"Dangerous command {cmd} in allowlist"


class TestPathTraversalSecurity:
    """Tests for path traversal prevention."""

    def test_path_traversal_rejected(self) -> None:
        """Path traversal with .. is rejected."""
        error = validate_working_dir("../../../etc/passwd")
        assert error is not None
        assert error.type == "PathTraversal"

    def test_double_dot_in_path_rejected(self) -> None:
        """Any .. in path is rejected."""
        error = validate_working_dir("/home/user/../root")
        assert error is not None
        assert error.type == "PathTraversal"

    def test_etc_path_rejected(self) -> None:
        """/etc paths are rejected."""
        error = validate_working_dir("/etc/shadow")
        assert error is not None
        assert error.type == "RestrictedPath"

    def test_root_home_rejected(self) -> None:
        """/root path is rejected."""
        error = validate_working_dir("/root/.ssh")
        assert error is not None
        assert error.type == "RestrictedPath"

    def test_proc_rejected(self) -> None:
        """/proc path is rejected."""
        error = validate_working_dir("/proc/self/environ")
        assert error is not None
        assert error.type == "RestrictedPath"

    def test_normal_path_accepted(self) -> None:
        """Normal paths are accepted."""
        assert validate_working_dir("/home/user/project") is None
        assert validate_working_dir("/tmp/workdir") is None
        assert validate_working_dir("./relative/path") is None

    def test_none_working_dir_accepted(self) -> None:
        """None working directory is accepted."""
        assert validate_working_dir(None) is None


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiter_imports(self) -> None:
        """Rate limiter module imports correctly."""
        from crows_nest.api.ratelimit import (
            get_rate_limiter,
        )

        limiter = get_rate_limiter()
        assert limiter is not None
        assert limiter.config.requests_per_minute > 0
        assert limiter.config.requests_per_second > 0

    def test_rate_limiter_can_be_disabled(self) -> None:
        """Rate limiter can be disabled via config."""
        from crows_nest.api.ratelimit import RateLimitConfig, RateLimiter

        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        assert not limiter.config.enabled


class TestInputValidation:
    """Tests for API input validation."""

    def test_thunk_request_operation_validation(self) -> None:
        """ThunkSubmitRequest validates operation format."""
        from pydantic import ValidationError

        from crows_nest.api.server import ThunkSubmitRequest

        # Valid operations
        ThunkSubmitRequest(operation="shell.run")
        ThunkSubmitRequest(operation="test_op")
        ThunkSubmitRequest(operation="a.b.c")

        # Invalid operations
        with pytest.raises(ValidationError):
            ThunkSubmitRequest(operation="")  # Too short

        with pytest.raises(ValidationError):
            ThunkSubmitRequest(operation="A" * 101)  # Too long

        with pytest.raises(ValidationError):
            ThunkSubmitRequest(operation="UPPERCASE")  # Must be lowercase

        with pytest.raises(ValidationError):
            ThunkSubmitRequest(operation="has spaces")  # No spaces

    def test_ask_request_query_validation(self) -> None:
        """AskRequest validates query length."""
        from pydantic import ValidationError

        from crows_nest.api.server import AskRequest

        # Valid queries
        AskRequest(query="list files")
        AskRequest(query="a")  # Minimum length

        # Invalid queries
        with pytest.raises(ValidationError):
            AskRequest(query="")  # Too short

        with pytest.raises(ValidationError):
            AskRequest(query="x" * 1001)  # Too long
