# Crow's Nest

A thunk-based multi-agent system built on Python and atomic-agents.

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

### Running

```bash
# CLI
uv run crows-nest --help
uv run cn --help  # alias

# API server
uv run crows-nest serve

# Or directly
uv run uvicorn crows_nest.api:app --reload
```

### Configuration

Copy the example configuration:

```bash
cp config.example.toml config.toml
```

Environment variables override config file values. Use the `CROWS_NEST_` prefix:

```bash
export CROWS_NEST_LLM__PROVIDER=anthropic
export CROWS_NEST_LLM__MODEL=claude-sonnet-4-20250514
export CROWS_NEST_API__PORT=9000
```

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
├── core/           # Thunk runtime, registry, evaluation
├── cli/            # Click CLI interface
├── api/            # FastAPI server
├── agents/         # Agent factory and operations
├── persistence/    # Storage backends (SQLite, etc.)
├── mocks/          # Mock LLM for testing
├── observability/  # Logging, tracing, metrics
└── plugins/        # Plugin discovery and loading
```

## License

MIT
