"""Tests for output verification operations."""

from __future__ import annotations

from crows_nest.shell.verification import (
    VerificationSchema,
    VerificationType,
    verify_output,
    verify_output_operation,
)


class TestVerificationTypes:
    """Tests for verification type parsing."""

    def test_string_type(self) -> None:
        """String values are verified correctly."""
        schema = VerificationSchema(type=VerificationType.STRING)
        result = verify_output(schema, "hello world")
        assert result.passed
        assert result.actual_value == "hello world"

    def test_string_min_length(self) -> None:
        """String minimum length is enforced."""
        schema = VerificationSchema(type=VerificationType.STRING, min_length=10)

        # Too short
        result = verify_output(schema, "short")
        assert not result.passed
        assert "less than minimum" in result.message

        # Long enough
        result = verify_output(schema, "long enough string")
        assert result.passed

    def test_string_max_length(self) -> None:
        """String maximum length is enforced."""
        schema = VerificationSchema(type=VerificationType.STRING, max_length=5)

        # Too long
        result = verify_output(schema, "too long")
        assert not result.passed

        # Short enough
        result = verify_output(schema, "ok")
        assert result.passed

    def test_integer_type(self) -> None:
        """Integer values are parsed and verified."""
        schema = VerificationSchema(type=VerificationType.INTEGER)

        # Integer
        result = verify_output(schema, 42)
        assert result.passed
        assert result.actual_value == 42

        # String integer
        result = verify_output(schema, "42")
        assert result.passed
        assert result.actual_value == 42

        # String with whitespace
        result = verify_output(schema, "  42\n")
        assert result.passed
        assert result.actual_value == 42

    def test_integer_min_value(self) -> None:
        """Integer minimum value is enforced."""
        schema = VerificationSchema(type=VerificationType.INTEGER, min_value=0)

        result = verify_output(schema, -1)
        assert not result.passed
        assert "less than minimum" in result.message

        result = verify_output(schema, 0)
        assert result.passed

    def test_integer_max_value(self) -> None:
        """Integer maximum value is enforced."""
        schema = VerificationSchema(type=VerificationType.INTEGER, max_value=100)

        result = verify_output(schema, 101)
        assert not result.passed

        result = verify_output(schema, 100)
        assert result.passed

    def test_integer_invalid(self) -> None:
        """Invalid integer strings fail."""
        schema = VerificationSchema(type=VerificationType.INTEGER)

        result = verify_output(schema, "not a number")
        assert not result.passed
        assert "Cannot parse" in result.message

    def test_float_type(self) -> None:
        """Float values are parsed and verified."""
        schema = VerificationSchema(type=VerificationType.FLOAT)

        result = verify_output(schema, 3.14)
        assert result.passed
        assert abs(result.actual_value - 3.14) < 0.001

        result = verify_output(schema, "3.14")
        assert result.passed

    def test_float_range(self) -> None:
        """Float range constraints work."""
        schema = VerificationSchema(
            type=VerificationType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )

        result = verify_output(schema, 0.5)
        assert result.passed

        result = verify_output(schema, 1.5)
        assert not result.passed

    def test_boolean_type(self) -> None:
        """Boolean values are parsed correctly."""
        schema = VerificationSchema(type=VerificationType.BOOLEAN)

        # True values
        for val in [True, "true", "yes", "1", "on"]:
            result = verify_output(schema, val)
            assert result.passed
            assert result.actual_value is True

        # False values
        for val in [False, "false", "no", "0", "off"]:
            result = verify_output(schema, val)
            assert result.passed
            assert result.actual_value is False

    def test_boolean_invalid(self) -> None:
        """Invalid boolean strings fail."""
        schema = VerificationSchema(type=VerificationType.BOOLEAN)

        result = verify_output(schema, "maybe")
        assert not result.passed

    def test_list_type(self) -> None:
        """List values are verified."""
        schema = VerificationSchema(type=VerificationType.LIST)

        # Python list
        result = verify_output(schema, [1, 2, 3])
        assert result.passed
        assert result.actual_value == [1, 2, 3]

        # Multiline string
        result = verify_output(schema, "line1\nline2\nline3")
        assert result.passed
        assert len(result.actual_value) == 3

    def test_list_length_constraints(self) -> None:
        """List length constraints work."""
        schema = VerificationSchema(
            type=VerificationType.LIST,
            min_length=2,
            max_length=5,
        )

        result = verify_output(schema, [1])  # Too short
        assert not result.passed

        result = verify_output(schema, [1, 2, 3])  # OK
        assert result.passed

        result = verify_output(schema, [1, 2, 3, 4, 5, 6])  # Too long
        assert not result.passed

    def test_regex_type(self) -> None:
        """Regex pattern matching works."""
        schema = VerificationSchema(
            type=VerificationType.REGEX,
            pattern=r"^\d{3}-\d{4}$",
        )

        result = verify_output(schema, "123-4567")
        assert result.passed

        result = verify_output(schema, "1234567")
        assert not result.passed

    def test_regex_missing_pattern(self) -> None:
        """Regex without pattern fails."""
        schema = VerificationSchema(type=VerificationType.REGEX)

        result = verify_output(schema, "test")
        assert not result.passed
        assert "requires a pattern" in result.message

    def test_contains_type(self) -> None:
        """Contains substring check works."""
        schema = VerificationSchema(
            type=VerificationType.CONTAINS,
            contains="success",
        )

        result = verify_output(schema, "Operation completed with success")
        assert result.passed

        result = verify_output(schema, "Operation failed")
        assert not result.passed

    def test_contains_missing_substring(self) -> None:
        """Contains without substring fails."""
        schema = VerificationSchema(type=VerificationType.CONTAINS)

        result = verify_output(schema, "test")
        assert not result.passed
        assert "requires a substring" in result.message

    def test_not_empty_type(self) -> None:
        """Not empty check works."""
        schema = VerificationSchema(type=VerificationType.NOT_EMPTY)

        result = verify_output(schema, "has content")
        assert result.passed

        result = verify_output(schema, "")
        assert not result.passed

        result = verify_output(schema, "   \n\t  ")  # Whitespace only
        assert not result.passed

    def test_json_type(self) -> None:
        """JSON parsing works."""
        schema = VerificationSchema(type=VerificationType.JSON)

        # Dict
        result = verify_output(schema, {"key": "value"})
        assert result.passed

        # JSON string
        result = verify_output(schema, '{"key": "value"}')
        assert result.passed
        assert result.actual_value == {"key": "value"}

        # Invalid JSON
        result = verify_output(schema, "not json")
        assert not result.passed
        assert "Invalid JSON" in result.message


class TestWhitespaceStripping:
    """Tests for whitespace handling."""

    def test_strip_by_default(self) -> None:
        """Whitespace is stripped by default."""
        schema = VerificationSchema(type=VerificationType.INTEGER)

        result = verify_output(schema, "  42  \n")
        assert result.passed
        assert result.actual_value == 42

    def test_no_strip_preserves_internal_whitespace(self) -> None:
        """Internal whitespace is preserved."""
        schema = VerificationSchema(
            type=VerificationType.STRING,
            min_length=5,
        )

        # String with internal whitespace
        result = verify_output(schema, "a b c")
        assert result.passed
        assert result.actual_value == "a b c"


class TestVerifyOutputOperation:
    """Tests for the verify.output thunk operation."""

    async def test_operation_passes(self) -> None:
        """verify.output operation works for valid input."""
        result = await verify_output_operation(
            schema={"type": "integer", "min_value": 0},
            value="42",
        )
        assert result["passed"]
        assert result["actual_value"] == 42

    async def test_operation_fails(self) -> None:
        """verify.output operation returns failure for invalid input."""
        result = await verify_output_operation(
            schema={"type": "integer", "min_value": 100},
            value="42",
        )
        assert not result["passed"]
        assert "less than minimum" in result["message"]

    async def test_operation_with_regex(self) -> None:
        """verify.output works with regex schemas."""
        result = await verify_output_operation(
            schema={"type": "regex", "pattern": r"\d+"},
            value="abc123def",
        )
        assert result["passed"]
