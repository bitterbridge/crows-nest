"""Tests for shell execution operations."""

from __future__ import annotations

import pytest

from crows_nest.shell.operations import (
    ALLOWED_COMMANDS,
    ShellCommand,
    ShellResult,
    check_capabilities,
    execute_command,
    execute_pipeline,
    shell_pipe,
    shell_run,
    validate_command,
)


class TestAllowedCommands:
    """Tests for the command allowlist."""

    def test_common_commands_allowed(self) -> None:
        """Common safe commands are in the allowlist."""
        safe_commands = ["ls", "cat", "grep", "wc", "find", "echo", "date", "pwd"]
        for cmd in safe_commands:
            assert cmd in ALLOWED_COMMANDS, f"{cmd} should be allowed"

    def test_dangerous_commands_not_allowed(self) -> None:
        """Dangerous commands are not in the allowlist."""
        dangerous = ["rm", "mv", "cp", "chmod", "chown", "sudo", "su", "dd", "mkfs"]
        for cmd in dangerous:
            assert cmd not in ALLOWED_COMMANDS, f"{cmd} should not be allowed"


class TestValidateCommand:
    """Tests for command validation."""

    def test_allowed_command_passes(self) -> None:
        """Allowed commands pass validation."""
        error = validate_command("ls", ["-la"])
        assert error is None

    def test_disallowed_command_fails(self) -> None:
        """Disallowed commands fail validation."""
        error = validate_command("rm", ["-rf", "/"])
        assert error is not None
        assert error.type == "CommandNotAllowed"
        assert "rm" in error.message

    def test_restricted_args_blocked(self) -> None:
        """Restricted arguments for allowed commands are blocked."""
        # tar create is blocked
        error = validate_command("tar", ["-c", "archive.tar"])
        assert error is not None
        assert error.type == "ArgumentNotAllowed"


class TestCheckCapabilities:
    """Tests for capability checking."""

    def test_capability_present_passes(self) -> None:
        """Having required capability passes check."""
        caps = frozenset({"shell.run", "shell.pipe"})
        error = check_capabilities(caps, "shell.run")
        assert error is None

    def test_capability_missing_fails(self) -> None:
        """Missing required capability fails check."""
        caps = frozenset({"shell.pipe"})
        error = check_capabilities(caps, "shell.run")
        assert error is not None
        assert error.type == "CapabilityMissing"


class TestShellCommand:
    """Tests for ShellCommand dataclass."""

    def test_to_string(self) -> None:
        """Commands serialize to shell string correctly."""
        cmd = ShellCommand(command="ls", args=["-la", "/tmp"])
        assert cmd.to_string() == "ls -la /tmp"

    def test_to_string_quotes_special(self) -> None:
        """Special characters are quoted."""
        cmd = ShellCommand(command="echo", args=["hello world"])
        result = cmd.to_string()
        assert "hello world" in result or "'hello world'" in result

    def test_from_dict(self) -> None:
        """Commands deserialize from dict."""
        data = {"command": "grep", "args": ["-r", "pattern"]}
        cmd = ShellCommand.from_dict(data)
        assert cmd.command == "grep"
        assert cmd.args == ["-r", "pattern"]


class TestShellResult:
    """Tests for ShellResult dataclass."""

    def test_success_property(self) -> None:
        """success property reflects exit code and timeout."""
        result = ShellResult(stdout="ok", stderr="", exit_code=0, command="ls")
        assert result.success is True

        result = ShellResult(stdout="", stderr="err", exit_code=1, command="ls")
        assert result.success is False

        result = ShellResult(stdout="", stderr="", exit_code=0, command="ls", timed_out=True)
        assert result.success is False

    def test_to_dict(self) -> None:
        """Results serialize to dict."""
        result = ShellResult(stdout="out", stderr="err", exit_code=0, command="ls")
        d = result.to_dict()
        assert d["stdout"] == "out"
        assert d["stderr"] == "err"
        assert d["exit_code"] == 0
        assert d["success"] is True


class TestExecuteCommand:
    """Tests for execute_command function."""

    async def test_execute_simple_command(self) -> None:
        """Simple commands execute successfully."""
        result = await execute_command("echo", ["hello"])
        assert result.success
        assert "hello" in result.stdout

    async def test_execute_command_with_args(self) -> None:
        """Commands with arguments work correctly."""
        result = await execute_command("echo", ["-n", "test"])
        assert result.success
        assert "test" in result.stdout

    async def test_execute_pwd(self) -> None:
        """pwd command returns working directory."""
        result = await execute_command("pwd", [])
        assert result.success
        assert "/" in result.stdout

    async def test_command_not_found(self) -> None:
        """Non-existent commands return error."""
        result = await execute_command("nonexistentcmd12345", [])
        assert not result.success
        assert result.exit_code == 127

    async def test_command_timeout(self) -> None:
        """Commands can timeout."""
        # This should timeout quickly
        result = await execute_command("sleep", ["10"], timeout_seconds=1)
        assert result.timed_out
        assert not result.success

    async def test_stdin_input(self) -> None:
        """Commands can receive stdin input."""
        result = await execute_command("cat", [], stdin="hello from stdin")
        assert result.success
        assert "hello from stdin" in result.stdout


class TestExecutePipeline:
    """Tests for execute_pipeline function."""

    async def test_simple_pipe(self) -> None:
        """Simple two-command pipeline works."""
        commands = [
            ShellCommand(command="echo", args=["hello world"]),
            ShellCommand(command="wc", args=["-c"]),
        ]
        result = await execute_pipeline(commands)
        assert result.success
        # "hello world\n" is 12 characters
        assert "12" in result.stdout

    async def test_three_command_pipe(self) -> None:
        """Three-command pipeline works."""
        commands = [
            ShellCommand(command="echo", args=["line1\nline2\nline3"]),
            ShellCommand(command="grep", args=["line"]),
            ShellCommand(command="wc", args=["-l"]),
        ]
        result = await execute_pipeline(commands)
        assert result.success
        # Should find 3 lines with "line"
        assert "3" in result.stdout

    async def test_empty_pipeline_fails(self) -> None:
        """Empty pipeline returns error."""
        result = await execute_pipeline([])
        assert not result.success
        assert "Empty pipeline" in result.stderr

    async def test_invalid_command_in_pipeline(self) -> None:
        """Invalid command in pipeline fails."""
        commands = [
            ShellCommand(command="echo", args=["test"]),
            ShellCommand(command="rm", args=["-rf"]),  # Not allowed
        ]
        result = await execute_pipeline(commands)
        assert not result.success
        assert "not in the allowlist" in result.stderr


class TestShellRunOperation:
    """Tests for the shell.run thunk operation."""

    async def test_run_allowed_command(self) -> None:
        """shell.run executes allowed commands."""
        caps = frozenset({"shell.run"})
        result = await shell_run(
            command="echo",
            args=["test"],
            _capabilities=caps,
        )
        assert result["success"]
        assert "test" in result["stdout"]

    async def test_run_without_capability(self) -> None:
        """shell.run fails without required capability."""
        caps = frozenset()  # No capabilities
        with pytest.raises(ValueError, match="Missing required capability"):
            await shell_run(command="echo", args=["test"], _capabilities=caps)

    async def test_run_disallowed_command(self) -> None:
        """shell.run rejects disallowed commands."""
        caps = frozenset({"shell.run"})
        with pytest.raises(ValueError, match="not in the allowlist"):
            await shell_run(command="rm", args=["-rf", "/"], _capabilities=caps)


class TestShellPipeOperation:
    """Tests for the shell.pipe thunk operation."""

    async def test_pipe_allowed_commands(self) -> None:
        """shell.pipe executes allowed command pipelines."""
        caps = frozenset({"shell.pipe"})
        result = await shell_pipe(
            commands=[
                {"command": "echo", "args": ["hello"]},
                {"command": "cat", "args": []},
            ],
            _capabilities=caps,
        )
        assert result["success"]
        assert "hello" in result["stdout"]

    async def test_pipe_without_capability(self) -> None:
        """shell.pipe fails without required capability."""
        caps = frozenset()
        with pytest.raises(ValueError, match="Missing required capability"):
            await shell_pipe(
                commands=[{"command": "echo", "args": ["test"]}],
                _capabilities=caps,
            )

    async def test_pipe_with_disallowed_command(self) -> None:
        """shell.pipe rejects pipelines with disallowed commands."""
        caps = frozenset({"shell.pipe"})
        with pytest.raises(ValueError, match="not in the allowlist"):
            await shell_pipe(
                commands=[
                    {"command": "echo", "args": ["test"]},
                    {"command": "rm", "args": ["-rf"]},
                ],
                _capabilities=caps,
            )
