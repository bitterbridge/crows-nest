"""Text Utilities Plugin - Demonstrates operations with multiple parameters.

This plugin provides text manipulation operations showing:
- Multiple input parameters
- Type annotations
- Validation
- Different return types

Usage:
    Copy to ~/.crows-nest/plugins/ and restart the server.
"""

from pydantic import BaseModel, Field

from crows_nest.core.registry import thunk_operation


# Pydantic models can be used for structured output
class TextStats(BaseModel):
    """Statistics about a text."""

    char_count: int = Field(description="Number of characters")
    word_count: int = Field(description="Number of words")
    line_count: int = Field(description="Number of lines")
    avg_word_length: float = Field(description="Average word length")


@thunk_operation(
    name="text.stats",
    description="Compute statistics about a text string",
    required_capabilities=frozenset({"text.analyze"}),
)
async def text_stats(text: str) -> dict:
    """Compute statistics about a text string.

    Args:
        text: The text to analyze

    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    lines = text.split("\n")

    stats = TextStats(
        char_count=len(text),
        word_count=len(words),
        line_count=len(lines),
        avg_word_length=sum(len(w) for w in words) / len(words) if words else 0,
    )

    return stats.model_dump()


@thunk_operation(
    name="text.transform",
    description="Transform text with various operations",
    required_capabilities=frozenset({"text.transform"}),
)
async def text_transform(
    text: str,
    operation: str = "upper",
    reverse: bool = False,
) -> str:
    """Transform text using the specified operation.

    Args:
        text: The text to transform
        operation: Transformation type (upper, lower, title, capitalize)
        reverse: Whether to reverse the result

    Returns:
        Transformed text
    """
    ops = {
        "upper": str.upper,
        "lower": str.lower,
        "title": str.title,
        "capitalize": str.capitalize,
    }

    transform_fn = ops.get(operation)
    if transform_fn is None:
        valid_ops = ", ".join(ops.keys())
        msg = f"Unknown operation '{operation}'. Valid: {valid_ops}"
        raise ValueError(msg)

    result = transform_fn(text)

    if reverse:
        result = result[::-1]

    return result


@thunk_operation(
    name="text.wrap",
    description="Wrap text to a specified width",
    required_capabilities=frozenset({"text.transform"}),
)
async def text_wrap(text: str, width: int = 80, indent: str = "") -> str:
    """Wrap text to a specified width.

    Args:
        text: The text to wrap
        width: Maximum line width (default: 80)
        indent: String to prepend to each line

    Returns:
        Wrapped text
    """
    import textwrap

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )

    paragraphs = text.split("\n\n")
    wrapped_paragraphs = [wrapper.fill(p) for p in paragraphs]

    return "\n\n".join(wrapped_paragraphs)


@thunk_operation(
    name="text.search",
    description="Search for patterns in text",
    required_capabilities=frozenset({"text.analyze"}),
)
async def text_search(
    text: str,
    pattern: str,
    case_sensitive: bool = True,
) -> dict:
    """Search for a pattern in text.

    Args:
        text: The text to search in
        pattern: Pattern to search for
        case_sensitive: Whether to match case (default: True)

    Returns:
        Dictionary with match information
    """
    search_text = text if case_sensitive else text.lower()
    search_pattern = pattern if case_sensitive else pattern.lower()

    matches = []
    start = 0
    while True:
        pos = search_text.find(search_pattern, start)
        if pos == -1:
            break
        matches.append(
            {
                "position": pos,
                "line": text[:pos].count("\n") + 1,
                "context": text[max(0, pos - 20) : pos + len(pattern) + 20],
            }
        )
        start = pos + 1

    return {
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches[:10],  # Limit to first 10
        "has_more": len(matches) > 10,
    }


# Plugin metadata
PLUGIN_NAME = "text_utils"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Text manipulation utilities"
PLUGIN_AUTHOR = "Crow's Nest Examples"
