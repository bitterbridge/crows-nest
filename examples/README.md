# Crow's Nest Examples

This directory contains example plugins and configurations for Crow's Nest.

## Plugins

### hello_world.py

The simplest possible plugin demonstrating:
- Basic operation registration with `@thunk_operation`
- Required capabilities
- Plugin metadata

Operations:
- `hello.greet` - Returns a greeting
- `hello.farewell` - Returns a farewell

### text_utils.py

More advanced plugin demonstrating:
- Multiple parameters with defaults
- Pydantic models for structured output
- Input validation
- Multiple operations in one file

Operations:
- `text.stats` - Compute text statistics
- `text.transform` - Transform text (upper, lower, etc.)
- `text.wrap` - Wrap text to width
- `text.search` - Search for patterns

## Installation

Copy plugin files to one of these locations:

1. `~/.crows-nest/plugins/` (user plugins)
2. Set `CROWS_NEST_PLUGINS_PATH` environment variable
3. Configure in `config.toml`:
   ```toml
   [plugins]
   paths = ["./my-plugins"]
   ```

## Usage

After installing a plugin, restart the server and use the operations:

```bash
# CLI
crows-nest ops list  # See new operations

# API
curl -X POST http://localhost:8000/thunks \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "hello.greet",
    "inputs": {"name": "Developer"},
    "capabilities": ["hello.greet"]
  }'
```

## Creating Your Own Plugin

1. Create a Python file with your operations:

```python
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="my.operation",
    description="Does something useful",
    required_capabilities=frozenset({"my.cap"}),
)
async def my_operation(param: str) -> str:
    return f"Result: {param}"

# Optional metadata
PLUGIN_NAME = "my_plugin"
PLUGIN_VERSION = "1.0.0"
```

2. Place in plugins directory
3. Restart server
4. Use via API or CLI

## Tips

- Use async functions for operations
- Declare all required capabilities
- Add type annotations for better docs
- Return serializable types (dict, list, str, int, etc.)
- Use Pydantic models for complex output
- Handle errors gracefully with meaningful messages
