# Crow's Nest Architecture

This document describes the architecture and design of Crow's Nest.

## Overview

Crow's Nest is built around the concept of **thunks** - lazy, pure, composable units of computation. The system is designed for:

- **Composability**: Complex workflows built from simple primitives
- **Observability**: Full visibility into execution via events and tracing
- **Security**: Capability-based access control
- **Extensibility**: Plugin system for custom operations

## Core Concepts

### Thunks

A thunk represents a deferred computation:

```python
@dataclass(frozen=True)
class Thunk:
    id: UUID                    # Unique identifier
    operation: str              # Operation name (e.g., "shell.run")
    inputs: dict[str, Any]      # Operation inputs
    metadata: ThunkMetadata     # Capabilities, tags, timestamps
```

Thunks are:
- **Immutable**: Once created, never modified
- **Pure**: Same inputs always produce same outputs
- **Lazy**: Only executed when explicitly forced
- **Composable**: Can reference other thunks via ThunkRef

### ThunkRef (Dependencies)

Thunks can depend on other thunks using ThunkRef:

```python
@dataclass
class ThunkRef:
    thunk_id: UUID
    output_path: str  # JSONPath to extract from result
```

This enables DAG-style execution where thunks wait for dependencies.

### Operations

Operations are the building blocks of computation:

```python
@thunk_operation(
    name="shell.run",
    description="Execute shell command",
    required_capabilities=frozenset({"shell.run"}),
)
async def shell_run(command: str, args: list[str]) -> ShellResult:
    ...
```

Operations:
- Have a unique name (dot-notation namespace)
- Declare required capabilities
- Are async Python functions
- Are registered in a global registry

### Capabilities

Capabilities control what operations can do:

```
shell.run       - Execute shell commands
agent.spawn     - Create child agents
memory.read     - Read from memory
memory.write    - Write to memory
llm.complete    - Call LLM for completions
```

Capabilities flow from parent to child and can only be restricted, never expanded.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  /ask    │  │ /thunks  │  │/dashboard│  │ /metrics │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ TaskPlanner  │  │ TaskExecutor │  │   Progress   │          │
│  │  (NL→Plan)   │  │ (Plan→Result)│  │   Tracking   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Runtime                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Thunk     │  │   Registry   │  │   Runtime    │          │
│  │  (immutable) │  │ (operations) │  │  (executor)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Persistence│  │  Queue   │  │  Memory  │  │  Events  │        │
│  │ (SQLite) │  │(messages)│  │ (agents) │  │(pub/sub) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### API Layer (`src/crows_nest/api/`)

FastAPI server providing:
- REST endpoints for thunk submission and querying
- `/ask` endpoint for natural language tasks
- Dashboard API for web UI
- WebSocket for real-time events
- Health, readiness, and metrics endpoints

### Orchestrator (`src/crows_nest/orchestrator/`)

Converts natural language to execution:

1. **TaskPlanner**: Analyzes query, creates execution plan
   - Pattern matching for common commands
   - Optional LLM for complex planning

2. **TaskExecutor**: Executes plan steps
   - Creates thunks with proper dependencies
   - Tracks progress
   - Handles verification

### Core Runtime (`src/crows_nest/core/`)

The execution engine:

- **Thunk**: Data structure for computations
- **Registry**: Stores operation definitions
- **Runtime**: Executes thunks
  - Resolves dependencies (DAG traversal)
  - Checks capabilities
  - Caches results

### Infrastructure

**Persistence** (`src/crows_nest/persistence/`):
- SQLite backend for thunks and results
- Async operations via aiosqlite

**Queue** (`src/crows_nest/queue/`):
- In-memory message queue
- Pub/sub for agent communication

**Memory** (`src/crows_nest/memory/`):
- Per-agent memory storage
- Decay-based importance scoring
- Query by type, text, importance

**Events** (`src/crows_nest/events/`):
- System-wide event bus
- Used for dashboard updates

## Data Flow

### Natural Language Task Execution

```
User: "count python files"
         │
         ▼
    ┌─────────┐
    │   /ask  │  API endpoint
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Planner │  Pattern match → shell.run + wc
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │Executor │  Create thunks with dependencies
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Runtime │  Force thunks in dependency order
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Result  │  "42 python files"
    └─────────┘
```

### Thunk Execution

```
Thunk Created
     │
     ▼
┌─────────────┐
│ Check Deps  │  Are all dependencies resolved?
└──────┬──────┘
       │
   ┌───┴───┐
   No      Yes
   │       │
   ▼       ▼
 Wait   ┌─────────────┐
        │Check Caps   │  Does caller have capabilities?
        └──────┬──────┘
               │
           ┌───┴───┐
           No      Yes
           │       │
           ▼       ▼
         Error  ┌─────────────┐
                │  Execute    │  Run operation handler
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │Store Result │  Save to persistence
                └─────────────┘
```

## Agent Hierarchy

Agents form a tree with capability inheritance:

```
Root Agent (all capabilities)
├── Worker Agent (shell.run, memory.*)
│   ├── Shell Agent (shell.run only)
│   └── Memory Agent (memory.* only)
└── Planner Agent (llm.complete, agent.spawn)
    └── Sub-planner (llm.complete only)
```

Each agent:
- Has a unique ID
- Tracks its parent
- Can only grant capabilities it has
- Maintains its own memory

## Security Model

### Defense in Depth

1. **Capability Checks**: Operations verify capabilities
2. **Command Allowlist**: Shell commands must be in allowlist
3. **Argument Validation**: Dangerous patterns blocked
4. **Path Restrictions**: Certain paths blocked
5. **Rate Limiting**: API requests throttled
6. **Input Validation**: All inputs validated

### Capability Flow

```
Parent Agent
    │
    │ spawn(caps={"shell.run", "memory.read"})
    │
    ▼
Child Agent (can only have shell.run, memory.read)
    │
    │ spawn(caps={"shell.run"})  ✓ allowed
    │ spawn(caps={"llm.complete"}) ✗ rejected
    │
    ▼
Grandchild Agent (can only have shell.run)
```

## Extensibility

### Adding Operations

```python
from crows_nest.core.registry import thunk_operation

@thunk_operation(
    name="my.operation",
    description="Does something",
    required_capabilities=frozenset({"my.cap"}),
)
async def my_operation(input: str) -> str:
    return process(input)
```

### Plugin Discovery

Plugins are discovered from:
1. `CROWS_NEST_PLUGINS_PATH` environment variable
2. `~/.crows-nest/plugins/` directory
3. Package entry points

## Performance Considerations

- **Caching**: Results cached by thunk ID
- **Lazy Evaluation**: Thunks only execute when forced
- **Dependency Resolution**: Topological sort for parallel execution
- **Output Truncation**: Large outputs truncated to 1MB
- **Connection Pooling**: Database connections reused
