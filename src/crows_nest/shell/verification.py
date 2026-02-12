"""Output verification thunk operations.

Provides schema-based validation of command outputs to ensure
they meet expected formats and constraints.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from crows_nest.core.registry import thunk_operation


class VerificationType(StrEnum):
    """Types of verification checks."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    REGEX = "regex"
    CONTAINS = "contains"
    NOT_EMPTY = "not_empty"
    JSON = "json"


@dataclass
class VerificationResult:
    """Result of output verification."""

    passed: bool
    message: str
    actual_value: Any
    expected_type: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "message": self.message,
            "actual_value": self.actual_value,
            "expected_type": self.expected_type,
            "details": self.details,
        }


class VerificationSchema(BaseModel):
    """Schema defining expected output format."""

    type: VerificationType = Field(description="Expected type of output")
    min_value: float | None = Field(default=None, description="Minimum value (for numbers)")
    max_value: float | None = Field(default=None, description="Maximum value (for numbers)")
    min_length: int | None = Field(default=None, description="Minimum length (for strings/lists)")
    max_length: int | None = Field(default=None, description="Maximum length (for strings/lists)")
    pattern: str | None = Field(default=None, description="Regex pattern (for regex type)")
    contains: str | None = Field(default=None, description="Substring to find (for contains type)")
    strip_whitespace: bool = Field(default=True, description="Strip whitespace before validation")


class VerifyOutputInput(BaseModel):
    """Input schema for verify.output operation."""

    schema_config: dict[str, Any] = Field(
        alias="schema",
        description="Verification schema configuration",
    )
    value: Any = Field(description="Value to verify")


class VerifyOutputOutput(BaseModel):
    """Output schema for verify.output operation."""

    passed: bool = Field(description="Whether verification passed")
    message: str = Field(description="Human-readable result message")
    actual_value: Any = Field(description="The actual value that was checked")
    expected_type: str = Field(description="The expected type")
    details: dict[str, Any] | None = Field(default=None, description="Additional details")


def _parse_value(value: Any, expected_type: VerificationType) -> tuple[Any, str | None]:
    """Parse a value to the expected type.

    Args:
        value: The value to parse.
        expected_type: The expected type.

    Returns:
        Tuple of (parsed_value, error_message or None).
    """
    # Handle string input that needs parsing
    if isinstance(value, str):
        value = value.strip()

    if expected_type == VerificationType.STRING:
        return str(value), None

    elif expected_type == VerificationType.INTEGER:
        try:
            # Handle string input
            if isinstance(value, str):
                # Try to parse as integer (handle "42\n" style output)
                cleaned = value.strip()
                return int(cleaned), None
            elif isinstance(value, int | float):
                return int(value), None
            else:
                return None, f"Cannot convert {type(value).__name__} to integer"
        except ValueError:
            return None, f"Cannot parse '{value}' as integer"

    elif expected_type == VerificationType.FLOAT:
        try:
            if isinstance(value, str):
                return float(value.strip()), None
            elif isinstance(value, int | float):
                return float(value), None
            else:
                return None, f"Cannot convert {type(value).__name__} to float"
        except ValueError:
            return None, f"Cannot parse '{value}' as float"

    elif expected_type == VerificationType.BOOLEAN:
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "yes", "1", "on"):
                return True, None
            elif lower in ("false", "no", "0", "off"):
                return False, None
            else:
                return None, f"Cannot parse '{value}' as boolean"
        return None, f"Cannot convert {type(value).__name__} to boolean"

    elif expected_type == VerificationType.LIST:
        if isinstance(value, list):
            return value, None
        if isinstance(value, str):
            # Split by newlines for multiline output
            lines = [line for line in value.strip().split("\n") if line.strip()]
            return lines, None
        return None, f"Cannot convert {type(value).__name__} to list"

    elif expected_type in (
        VerificationType.REGEX,
        VerificationType.CONTAINS,
        VerificationType.NOT_EMPTY,
    ):
        # These work on string representation
        return str(value), None

    elif expected_type == VerificationType.JSON:
        if isinstance(value, dict | list):
            return value, None
        if isinstance(value, str):
            try:
                return json.loads(value), None
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON: {e}"
        return None, f"Cannot parse {type(value).__name__} as JSON"

    return value, None


def verify_output(schema: VerificationSchema, value: Any) -> VerificationResult:
    """Verify that a value matches the expected schema.

    Args:
        schema: The verification schema.
        value: The value to verify.

    Returns:
        VerificationResult with pass/fail status and details.
    """
    # Strip whitespace if configured
    if schema.strip_whitespace and isinstance(value, str):
        value = value.strip()

    # Parse value to expected type
    parsed, parse_error = _parse_value(value, schema.type)
    if parse_error:
        return VerificationResult(
            passed=False,
            message=parse_error,
            actual_value=value,
            expected_type=schema.type.value,
        )

    # Type-specific validation
    if schema.type == VerificationType.INTEGER:
        if schema.min_value is not None and parsed < schema.min_value:
            return VerificationResult(
                passed=False,
                message=f"Value {parsed} is less than minimum {schema.min_value}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"min_value": schema.min_value},
            )
        if schema.max_value is not None and parsed > schema.max_value:
            return VerificationResult(
                passed=False,
                message=f"Value {parsed} is greater than maximum {schema.max_value}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"max_value": schema.max_value},
            )

    elif schema.type == VerificationType.FLOAT:
        if schema.min_value is not None and parsed < schema.min_value:
            return VerificationResult(
                passed=False,
                message=f"Value {parsed} is less than minimum {schema.min_value}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"min_value": schema.min_value},
            )
        if schema.max_value is not None and parsed > schema.max_value:
            return VerificationResult(
                passed=False,
                message=f"Value {parsed} is greater than maximum {schema.max_value}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"max_value": schema.max_value},
            )

    elif schema.type == VerificationType.STRING:
        if schema.min_length is not None and len(parsed) < schema.min_length:
            return VerificationResult(
                passed=False,
                message=f"String length {len(parsed)} is less than minimum {schema.min_length}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"length": len(parsed), "min_length": schema.min_length},
            )
        if schema.max_length is not None and len(parsed) > schema.max_length:
            return VerificationResult(
                passed=False,
                message=f"String length {len(parsed)} is greater than maximum {schema.max_length}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"length": len(parsed), "max_length": schema.max_length},
            )

    elif schema.type == VerificationType.LIST:
        if schema.min_length is not None and len(parsed) < schema.min_length:
            return VerificationResult(
                passed=False,
                message=f"List length {len(parsed)} is less than minimum {schema.min_length}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"length": len(parsed), "min_length": schema.min_length},
            )
        if schema.max_length is not None and len(parsed) > schema.max_length:
            return VerificationResult(
                passed=False,
                message=f"List length {len(parsed)} is greater than maximum {schema.max_length}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"length": len(parsed), "max_length": schema.max_length},
            )

    elif schema.type == VerificationType.REGEX:
        if schema.pattern is None:
            return VerificationResult(
                passed=False,
                message="Regex verification requires a pattern",
                actual_value=parsed,
                expected_type=schema.type.value,
            )
        try:
            if not re.search(schema.pattern, str(parsed)):
                return VerificationResult(
                    passed=False,
                    message=f"Value does not match pattern: {schema.pattern}",
                    actual_value=parsed,
                    expected_type=schema.type.value,
                    details={"pattern": schema.pattern},
                )
        except re.error as e:
            return VerificationResult(
                passed=False,
                message=f"Invalid regex pattern: {e}",
                actual_value=parsed,
                expected_type=schema.type.value,
            )

    elif schema.type == VerificationType.CONTAINS:
        if schema.contains is None:
            return VerificationResult(
                passed=False,
                message="Contains verification requires a substring",
                actual_value=parsed,
                expected_type=schema.type.value,
            )
        if schema.contains not in str(parsed):
            return VerificationResult(
                passed=False,
                message=f"Value does not contain: {schema.contains}",
                actual_value=parsed,
                expected_type=schema.type.value,
                details={"expected_substring": schema.contains},
            )

    elif schema.type == VerificationType.NOT_EMPTY:
        str_value = str(parsed).strip()
        if not str_value:
            return VerificationResult(
                passed=False,
                message="Value is empty",
                actual_value=parsed,
                expected_type=schema.type.value,
            )

    # All checks passed
    return VerificationResult(
        passed=True,
        message="Verification passed",
        actual_value=parsed,
        expected_type=schema.type.value,
    )


@thunk_operation(
    name="verify.output",
    description="Verify that output matches expected schema",
    required_capabilities=frozenset(),  # No special capabilities needed
)
async def verify_output_operation(
    schema: dict[str, Any],
    value: Any,
    *,
    _capabilities: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Verify that a value matches the expected schema.

    Args:
        schema: Verification schema with type and constraints.
        value: The value to verify.
        _capabilities: Capabilities passed by runtime (internal).

    Returns:
        Dictionary with passed, message, actual_value, expected_type.
    """
    # Parse schema
    verification_schema = VerificationSchema(**schema)

    # Verify
    result = verify_output(verification_schema, value)

    return result.to_dict()
