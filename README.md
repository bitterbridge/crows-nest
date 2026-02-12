# Crow's Nest

A thunk-based multi-agent system for composable, dynamic AI workflows.

## Overview

Crow's Nest is an experimental framework for building composable, dynamic agent systems. Everything is a **thunk** - a lazy, pure, composable unit of computation. Agents are stateless, dynamically created, and communicate through flexible channels.

### Key Concepts

- **Thunks**: Deferred computations with typed inputs, dependencies, and metadata
- **Dynamic Agents**: Agents create agents - no predefined agent types
- **Capability Model**: Inherit-and-restrict permissions flow from parent to child
- **Pluggable Everything**: Storage, communication, and tools are all extensible

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bitterbridge/crows-nest.git
cd crows-nest

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Execute a natural language task
uv run crows-nest ask "list python files in src/"

# With verbose output
uv run crows-nest ask "count lines in README.md" --verbose

# Start the API server with dashboard
uv run crows-nest serve

# Open dashboard at http://localhost:8000/dashboard
```

### CLI Commands

```bash
# Get help
uv run crows-nest --help

# Execute tasks
uv run crows-nest ask "your task here"

# Start API server
uv run crows-nest serve --host 0.0.0.0 --port 8000

# List registered operations
uv run crows-nest ops

# Check system info
uv run crows-nest info
```

## Web Dashboard

The built-in dashboard provides real-time visibility into the system:

- **Overview**: System stats, active thunks, queue depths
- **Thunks**: List and DAG visualization of thunk dependencies
- **Agents**: Agent hierarchy and capabilities
- **Queues**: Message queue monitoring
- **Events**: Real-time event stream via WebSocket
- **Config**: Current configuration display

Access at `http://localhost:8000/dashboard` when the server is running.

## API

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Execute natural language task |
| `/thunks` | POST | Submit a thunk for execution |
| `/thunks/{id}` | GET | Get thunk details |
| `/thunks/{id}/result` | GET | Get thunk execution result |
| `/ops` | GET | List registered operations |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

### Example API Usage

```bash
# Execute a task
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "list files in current directory"}'

# Submit a thunk
curl -X POST http://localhost:8000/thunks \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "shell.run",
    "inputs": {"command": "ls", "args": ["-la"]},
    "capabilities": ["shell.run"]
  }'

# Check health
curl http://localhost:8000/health
```

## Configuration

### Config File

Copy the example configuration:

```bash
cp config.example.toml config.toml
```

### Environment Variables

Environment variables override config file values. Use the `CROWS_NEST_` prefix:

```bash
export CROWS_NEST_DATABASE__PATH=./data/crows_nest.db
export CROWS_NEST_API__HOST=0.0.0.0
export CROWS_NEST_API__PORT=8000
export CROWS_NEST_LLM__PROVIDER=anthropic
export CROWS_NEST_LLM__MODEL=claude-sonnet-4-20250514
```

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE__PATH` | `./crows_nest.db` | SQLite database path |
| `API__HOST` | `127.0.0.1` | API server host |
| `API__PORT` | `8000` | API server port |
| `API__CORS_ENABLED` | `true` | Enable CORS |
| `LLM__PROVIDER` | `mock` | LLM provider (mock, anthropic) |
| `LOG__LEVEL` | `INFO` | Log level |

## Development

### Setup

```bash
# Install all dependencies including dev tools
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Checks

```bash
# All lints and tests
uv run pre-commit run --all-files
uv run pytest

# Individual tools
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run mypy src/           # Type check
uv run bandit -r src/      # Security scan
uv run pytest --cov        # Tests with coverage
```

### Docker

```bash
# Build dev image (includes test tooling)
docker build --target dev -t crows-nest:dev .

# Run tests in container
docker run --rm crows-nest:dev

# Build production image
docker build --target runtime -t crows-nest:latest .

# Run production image
docker run -p 8000:8000 crows-nest:latest
```

## Architecture

```
src/crows_nest/
├── core/           # Thunk runtime, registry, configuration
├── cli/            # Click CLI interface
├── api/            # FastAPI server and dashboard API
├── dashboard/      # Web dashboard (HTML, CSS, JS)
├── agents/         # Agent factory and hierarchy
├── orchestrator/   # Task planning and execution
├── shell/          # Shell command operations
├── persistence/    # Storage backends (SQLite)
├── memory/         # Agent memory system
├── queue/          # Message queue system
├── mocks/          # Mock LLM for testing
├── observability/  # Logging, tracing, metrics
└── plugins/        # Plugin discovery and loading
```

### Core Components

- **Thunk**: Immutable unit of computation with inputs, operation, and metadata
- **Runtime**: Executes thunks, manages dependencies, caches results
- **Registry**: Stores operation definitions and handlers
- **Persistence**: Saves thunks and results to SQLite
- **Orchestrator**: Converts natural language to execution plans

## Plugin System

Plugins can add new operations:

```python
# my_plugin.py
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="my.custom_op",
    description="My custom operation",
    required_capabilities=frozenset({"my.custom"}),
)
async def custom_operation(input_value: str) -> str:
    return f"Processed: {input_value}"
```

Load plugins by placing them in the plugins directory or via configuration.

## Security

- **Shell Command Allowlist**: Only safe read-only commands allowed
- **Capability System**: Operations require explicit capabilities
- **Rate Limiting**: API endpoints are rate-limited (60/min, 10/sec)
- **Input Validation**: All API inputs validated with length/pattern constraints
- **Path Traversal Protection**: Working directories validated

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-report=html

# Run specific test file
uv run pytest tests/test_security.py -v

# Run tests matching pattern
uv run pytest -k "shell" -v
```

## License

MIT
