"""Shell execution operations for Crow's Nest.

Provides safe, sandboxed shell command execution with an allowlist
of permitted commands and capability-based access control.
"""

from crows_nest.shell.operations import (
    ALLOWED_COMMANDS,
    ShellCommand,
    ShellError,
    ShellPipeInput,
    ShellPipeOutput,
    ShellResult,
    ShellRunInput,
    ShellRunOutput,
)
from crows_nest.shell.verification import (
    VerificationResult,
    VerificationSchema,
    VerificationType,
    VerifyOutputInput,
    VerifyOutputOutput,
)

__all__ = [
    "ALLOWED_COMMANDS",
    "ShellCommand",
    "ShellError",
    "ShellPipeInput",
    "ShellPipeOutput",
    "ShellResult",
    "ShellRunInput",
    "ShellRunOutput",
    "VerificationResult",
    "VerificationSchema",
    "VerificationType",
    "VerifyOutputInput",
    "VerifyOutputOutput",
]
