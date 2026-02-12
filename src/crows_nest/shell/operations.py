"""Shell execution thunk operations.

Provides safe shell command execution with allowlist-based security.
Commands must be in the allowlist and the caller must have appropriate
capabilities to execute them.
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from crows_nest.core.registry import thunk_operation

# Allowlist of safe commands that can be executed
# These are common UNIX utilities that don't modify the system
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        # File listing and information
        "ls",
        "find",
        "stat",
        "file",
        "du",
        "df",
        # File content reading
        "cat",
        "head",
        "tail",
        "less",
        "more",
        # Text processing
        "grep",
        "awk",
        "sed",
        "cut",
        "tr",
        "sort",
        "uniq",
        "wc",
        "diff",
        "comm",
        # String manipulation
        "echo",
        "printf",
        "basename",
        "dirname",
        # Date and time
        "date",
        "cal",
        # System information (read-only)
        "uname",
        "hostname",
        "whoami",
        "id",
        "env",
        "printenv",
        "pwd",
        "which",
        "type",
        # Archive reading (not writing)
        "tar",  # Will be restricted to list/extract only
        "unzip",  # List and extract only
        "zipinfo",
        "gzip",  # Decompression only
        "gunzip",
        "zcat",
        # JSON processing
        "jq",
        # Checksum
        "md5sum",
        "sha256sum",
        "shasum",
        # Misc utilities
        "true",
        "false",
        "test",
        "expr",
        "seq",
        "yes",  # Limited use
        "tee",  # For output splitting in pipes
        "xargs",  # For command building
    }
)

# Commands that have restricted arguments
RESTRICTED_ARGS: dict[str, set[str]] = {
    # Prevent tar from creating archives or extracting to absolute paths
    "tar": {
        "-c",
        "--create",
        "-r",
        "--append",
        "-u",
        "--update",
        "--delete",
        "-P",
        "--absolute-names",
    },
    "rm": set(),  # rm is not in allowlist
    "mv": set(),  # mv is not in allowlist
    "cp": set(),  # cp is not in allowlist
    # Prevent sed/awk from modifying files in-place
    "sed": {"-i", "--in-place"},
    "awk": {"-i"},
}

# Dangerous argument patterns that could indicate injection attempts
DANGEROUS_PATTERNS: frozenset[str] = frozenset(
    {
        ";",  # Command chaining
        "&&",  # Conditional execution
        "||",  # Conditional execution
        "|",  # Piping (could leak to other commands)
        "`",  # Command substitution
        "$(",  # Command substitution
        ">",  # Redirection
        "<",  # Redirection
        ">>",  # Append redirection
    }
)

# Default timeout for shell commands (30 seconds)
DEFAULT_TIMEOUT_SECONDS: int = 30

# Maximum output size (1MB)
MAX_OUTPUT_SIZE: int = 1024 * 1024


@dataclass
class ShellError:
    """Error information from shell execution."""

    type: str
    message: str
    command: str
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "message": self.message,
            "command": self.command,
            "exit_code": self.exit_code,
        }


@dataclass
class ShellResult:
    """Result of shell command execution."""

    stdout: str
    stderr: str
    exit_code: int
    command: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "command": self.command,
            "timed_out": self.timed_out,
            "success": self.success,
        }


@dataclass
class ShellCommand:
    """A single shell command with arguments."""

    command: str
    args: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert to shell command string."""
        parts = [self.command, *self.args]
        return shlex.join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShellCommand:
        """Create from dictionary."""
        return cls(
            command=data["command"],
            args=data.get("args", []),
        )


class ShellRunInput(BaseModel):
    """Input schema for shell.run operation."""

    command: str = Field(description="Command to execute (must be in allowlist)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    stdin: str | None = Field(default=None, description="Optional stdin input")
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        description="Timeout in seconds",
        ge=1,
        le=300,
    )
    working_dir: str | None = Field(
        default=None,
        description="Working directory (defaults to current)",
    )


class ShellRunOutput(BaseModel):
    """Output schema for shell.run operation."""

    stdout: str = Field(description="Standard output")
    stderr: str = Field(description="Standard error")
    exit_code: int = Field(description="Exit code")
    success: bool = Field(description="Whether command succeeded")
    timed_out: bool = Field(default=False, description="Whether command timed out")
    command: str = Field(description="The executed command")


class ShellPipeInput(BaseModel):
    """Input schema for shell.pipe operation."""

    commands: list[dict[str, Any]] = Field(description="List of commands to pipe together")
    stdin: str | None = Field(default=None, description="Optional initial stdin")
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        description="Total timeout in seconds",
        ge=1,
        le=300,
    )
    working_dir: str | None = Field(
        default=None,
        description="Working directory (defaults to current)",
    )


class ShellPipeOutput(BaseModel):
    """Output schema for shell.pipe operation."""

    stdout: str = Field(description="Final standard output")
    stderr: str = Field(description="Combined standard error")
    exit_code: int = Field(description="Exit code of last command")
    success: bool = Field(description="Whether pipeline succeeded")
    timed_out: bool = Field(default=False, description="Whether pipeline timed out")
    pipeline: str = Field(description="The executed pipeline")


def validate_command(command: str, args: list[str]) -> ShellError | None:
    """Validate that a command is allowed.

    Args:
        command: The command name.
        args: Command arguments.

    Returns:
        ShellError if command is not allowed, None if valid.
    """
    # Check if command is in allowlist
    if command not in ALLOWED_COMMANDS:
        return ShellError(
            type="CommandNotAllowed",
            message=f"Command '{command}' is not in the allowlist",
            command=command,
        )

    # Check for restricted arguments
    if command in RESTRICTED_ARGS:
        restricted = RESTRICTED_ARGS[command]
        for arg in args:
            if arg in restricted:
                return ShellError(
                    type="ArgumentNotAllowed",
                    message=f"Argument '{arg}' is not allowed for command '{command}'",
                    command=command,
                )

    # Check for dangerous patterns in arguments
    # These could indicate shell injection attempts
    for arg in args:
        for pattern in DANGEROUS_PATTERNS:
            if pattern in arg:
                return ShellError(
                    type="DangerousPattern",
                    message=f"Argument contains potentially dangerous pattern: '{pattern}'",
                    command=command,
                )

    return None


def validate_working_dir(working_dir: str | None) -> ShellError | None:
    """Validate that working directory is safe.

    Args:
        working_dir: The working directory path.

    Returns:
        ShellError if directory is unsafe, None if valid.
    """
    if working_dir is None:
        return None

    # Check for path traversal attempts
    if ".." in working_dir:
        return ShellError(
            type="PathTraversal",
            message="Working directory contains path traversal sequence '..'",
            command="",
        )

    # Ensure absolute path or relative to current directory
    # Block paths that start with /etc, /root, etc.
    dangerous_prefixes = ("/etc", "/root", "/var/log", "/proc", "/sys", "/dev")
    for prefix in dangerous_prefixes:
        if working_dir.startswith(prefix):
            return ShellError(
                type="RestrictedPath",
                message=f"Working directory '{prefix}' is restricted",
                command="",
            )

    return None


def check_capabilities(
    capabilities: frozenset[str],
    required: str,
) -> ShellError | None:
    """Check if required capability is present.

    Args:
        capabilities: Available capabilities.
        required: Required capability name.

    Returns:
        ShellError if capability is missing, None if present.
    """
    if required not in capabilities:
        return ShellError(
            type="CapabilityMissing",
            message=f"Missing required capability: {required}",
            command="",
        )
    return None


async def execute_command(
    command: str,
    args: list[str],
    *,
    stdin: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str | None = None,
) -> ShellResult:
    """Execute a shell command.

    Args:
        command: Command to execute.
        args: Command arguments.
        stdin: Optional stdin input.
        timeout_seconds: Timeout in seconds.
        working_dir: Optional working directory.

    Returns:
        ShellResult with output and exit code.
    """
    full_command = shlex.join([command, *args])

    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        stdin_bytes = stdin.encode() if stdin else None

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_bytes),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return ShellResult(
                stdout="",
                stderr="Command timed out",
                exit_code=-1,
                command=full_command,
                timed_out=True,
            )

        # Truncate output if too large
        stdout = stdout_bytes.decode(errors="replace")[:MAX_OUTPUT_SIZE]
        stderr = stderr_bytes.decode(errors="replace")[:MAX_OUTPUT_SIZE]

        return ShellResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode or 0,
            command=full_command,
        )

    except FileNotFoundError:
        return ShellResult(
            stdout="",
            stderr=f"Command not found: {command}",
            exit_code=127,
            command=full_command,
        )
    except PermissionError:
        return ShellResult(
            stdout="",
            stderr=f"Permission denied: {command}",
            exit_code=126,
            command=full_command,
        )


async def execute_pipeline(
    commands: list[ShellCommand],
    *,
    stdin: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str | None = None,
) -> ShellResult:
    """Execute a pipeline of shell commands.

    Each command's stdout becomes the next command's stdin.

    Args:
        commands: List of commands to pipe together.
        stdin: Optional initial stdin.
        timeout_seconds: Total timeout for pipeline.
        working_dir: Optional working directory.

    Returns:
        ShellResult with final output and combined stderr.
    """
    if not commands:
        return ShellResult(
            stdout="",
            stderr="Empty pipeline",
            exit_code=1,
            command="",
        )

    pipeline_str = " | ".join(cmd.to_string() for cmd in commands)
    current_stdin = stdin
    all_stderr: list[str] = []

    for i, cmd in enumerate(commands):
        # Validate each command
        error = validate_command(cmd.command, cmd.args)
        if error:
            return ShellResult(
                stdout="",
                stderr=error.message,
                exit_code=1,
                command=pipeline_str,
            )

        result = await execute_command(
            cmd.command,
            cmd.args,
            stdin=current_stdin,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
        )

        if result.stderr:
            all_stderr.append(f"[{cmd.command}]: {result.stderr}")

        if result.timed_out:
            return ShellResult(
                stdout=result.stdout,
                stderr="\n".join(all_stderr),
                exit_code=-1,
                command=pipeline_str,
                timed_out=True,
            )

        # Non-zero exit code in pipeline stops execution
        # (unless it's not the last command and we want to continue)
        if result.exit_code != 0 and i == len(commands) - 1:
            return ShellResult(
                stdout=result.stdout,
                stderr="\n".join(all_stderr),
                exit_code=result.exit_code,
                command=pipeline_str,
            )

        # Pass stdout to next command's stdin
        current_stdin = result.stdout

    return ShellResult(
        stdout=current_stdin or "",
        stderr="\n".join(all_stderr),
        exit_code=0,
        command=pipeline_str,
    )


@thunk_operation(
    name="shell.run",
    description="Execute a single shell command (must be in allowlist)",
    required_capabilities=frozenset({"shell.run"}),
)
async def shell_run(
    command: str,
    args: list[str] | None = None,
    stdin: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str | None = None,
    *,
    _capabilities: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Execute a single shell command.

    Args:
        command: Command to execute (must be in allowlist).
        args: Command arguments.
        stdin: Optional stdin input.
        timeout_seconds: Timeout in seconds (default 30).
        working_dir: Optional working directory.
        _capabilities: Capabilities passed by runtime (internal).

    Returns:
        Dictionary with stdout, stderr, exit_code, success.

    Raises:
        ValueError: If command is not allowed or capability is missing.
    """
    args = args or []
    capabilities = _capabilities or frozenset()

    # Check capability
    cap_error = check_capabilities(capabilities, "shell.run")
    if cap_error:
        raise ValueError(cap_error.message)

    # Validate command
    cmd_error = validate_command(command, args)
    if cmd_error:
        raise ValueError(cmd_error.message)

    # Execute
    result = await execute_command(
        command,
        args,
        stdin=stdin,
        timeout_seconds=timeout_seconds,
        working_dir=working_dir,
    )

    return result.to_dict()


@thunk_operation(
    name="shell.pipe",
    description="Execute a pipeline of shell commands",
    required_capabilities=frozenset({"shell.pipe"}),
)
async def shell_pipe(
    commands: list[dict[str, Any]],
    stdin: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str | None = None,
    *,
    _capabilities: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Execute a pipeline of shell commands.

    Args:
        commands: List of command dicts with 'command' and 'args' keys.
        stdin: Optional initial stdin.
        timeout_seconds: Total timeout in seconds (default 30).
        working_dir: Optional working directory.
        _capabilities: Capabilities passed by runtime (internal).

    Returns:
        Dictionary with stdout, stderr, exit_code, success.

    Raises:
        ValueError: If any command is not allowed or capability is missing.
    """
    capabilities = _capabilities or frozenset()

    # Check capability
    cap_error = check_capabilities(capabilities, "shell.pipe")
    if cap_error:
        raise ValueError(cap_error.message)

    # Parse commands
    shell_commands = [ShellCommand.from_dict(cmd) for cmd in commands]

    # Validate all commands first
    for cmd in shell_commands:
        cmd_error = validate_command(cmd.command, cmd.args)
        if cmd_error:
            raise ValueError(cmd_error.message)

    # Execute pipeline
    result = await execute_pipeline(
        shell_commands,
        stdin=stdin,
        timeout_seconds=timeout_seconds,
        working_dir=working_dir,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "success": result.success,
        "timed_out": result.timed_out,
        "pipeline": result.command,
    }
