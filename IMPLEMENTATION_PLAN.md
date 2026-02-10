# Crow's Nest Implementation Plan

A thunk-based multi-agent system built on Python + atomic-agents.

## Overview

**Goal:** Build a flexible, composable agent system where everything is a thunk - lazy, pure, and composable. Agents are stateless, dynamically created, and communicate through multiple channels.

**MVP Target:** CLI/API where a user can request a simple UNIX command pipeline, and an orchestrator agent delegates execution, verifies completion, and reports back.

---

## Stage 1: Project Scaffolding & Quality Gates

**Goal:** Establish project structure, tooling, and CI pipeline. All quality gates operational.

**Success Criteria:**
- [x] `uv run pytest` passes with 0 tests (scaffold complete)
- [x] `uv run ruff check .` passes
- [x] `uv run ruff format --check .` passes
- [x] `uv run mypy src/` passes
- [x] `uv run bandit -r src/` passes (0 issues)
- [x] Pre-commit hooks run all checks
- [ ] Docker build succeeds (Docker not running locally, Dockerfile ready)
- [ ] GHA workflow builds container and runs checks (will verify on push)

**Tests:**
- Smoke test that imports the package
- Pre-commit runs without errors on clean repo

**Deliverables:**
```
crows-nest/
├── src/
│   └── crows_nest/
│       ├── __init__.py
│       ├── core/           # Thunk runtime (Stage 2)
│       ├── cli/            # Click CLI (Stage 4)
│       ├── api/            # FastAPI server (Stage 4)
│       ├── agents/         # Agent factory (Stage 3)
│       ├── persistence/    # Storage backends (Stage 2)
│       ├── mocks/          # Mock LLM system (Stage 3)
│       ├── observability/  # Logging, tracing (Stage 4)
│       └── plugins/        # Plugin loader (Stage 5+)
├── tests/
│   ├── conftest.py
│   └── test_smoke.py
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── config.example.toml
└── README.md
```

**Status:** Complete

---

## Stage 2: Core Thunk Primitives

**Goal:** Implement the thunk runtime - the foundation everything else builds on.

**Success Criteria:**
- [x] `Thunk` dataclass with id, operation, inputs, dependencies, metadata
- [x] `ThunkRegistry` for registering operations
- [x] `ThunkRuntime` forces thunks, resolves dependencies in correct order
- [x] Thunks can produce other thunks (thunk-returning-thunk pattern)
- [x] `PersistenceBackend` protocol defined
- [x] SQLite backend implements persistence protocol
- [x] All thunk lifecycle events are traceable (created, forced, completed, failed)
- [x] 80%+ test coverage on core module (achieved: 92%)

**Tests:** (72 tests total)
- Unit: Thunk creation, serialization, deserialization
- Unit: Registry registration, lookup, validation
- Unit: Runtime forces simple thunk
- Unit: Runtime resolves linear dependency chain (A → B → C)
- Unit: Runtime resolves diamond dependency (A → B, A → C, B+C → D)
- Unit: Thunk returns thunk, runtime handles correctly
- Integration: SQLite persistence round-trip

**Key Types:**
```python
@dataclass
class Thunk[T]:
    id: UUID
    operation: str
    inputs: dict[str, Any]  # JSON-serializable
    dependencies: list[ThunkRef]
    metadata: ThunkMetadata

class ThunkRef:
    """Reference to another thunk's output."""
    thunk_id: UUID
    output_path: str | None  # JSONPath for extracting subset

class ThunkMetadata:
    created_at: datetime
    created_by: str | None  # parent thunk or "user"
    trace_id: str
    capabilities: set[str]  # inherit-and-restrict

class ThunkResult[T]:
    thunk_id: UUID
    status: Literal["success", "failure"]
    value: T | None
    error: ThunkError | None
    forced_at: datetime
    duration_ms: int
```

**Status:** Complete

---

## Stage 3: Mock LLM & Agent Foundation

**Goal:** Build mock LLM system and integrate atomic-agents for real LLM calls. Establish agent-as-thunk pattern.

**Success Criteria:**
- [ ] Mock LLM provider generates JSON Schema-conformant responses
- [ ] Mock provider is injectable (swap real ↔ mock easily)
- [ ] atomic-agents integration working with Anthropic provider
- [ ] `AgentFactory` creates agents dynamically from config thunks
- [ ] Agent execution wrapped as thunk operation
- [ ] Capability inheritance works (child agent gets subset of parent capabilities)
- [ ] 80%+ test coverage

**Tests:**
- Unit: Mock LLM generates valid response for simple schema
- Unit: Mock LLM generates valid response for nested schema
- Unit: Mock LLM generates valid response for array schema
- Unit: Mock LLM generates valid response for enum schema
- Integration: atomic-agents agent runs with mock provider
- Integration: atomic-agents agent runs with real provider (skipped in CI without keys)
- Unit: AgentFactory creates agent from config dict
- Unit: Capability restriction prevents child from gaining capabilities
- Unit: Capability restriction allows subset inheritance

**Key Types:**
```python
class MockLLMProvider:
    """Generates schema-conformant mock responses."""
    def complete(self, messages, output_schema: type[BaseIOSchema]) -> BaseIOSchema: ...

class AgentConfig(BaseIOSchema):
    """Configuration for dynamically creating an agent."""
    system_prompt: str
    capabilities: set[str]
    output_schema: dict  # JSON Schema
    max_tokens: int = 1000
    temperature: float = 0.7

# Thunk operations
@thunk_operation("agent.create")
async def create_agent(config: AgentConfig, parent_capabilities: set[str]) -> AgentRef: ...

@thunk_operation("agent.run")
async def run_agent(agent: AgentRef, input: dict) -> AgentOutput: ...
```

**Status:** Not Started

---

## Stage 4: CLI, API & Observability

**Goal:** User-facing interfaces and production observability.

**Success Criteria:**
- [ ] `crows-nest` CLI with `--help`, `--version`
- [ ] `cn` alias works
- [ ] CLI can execute a thunk from JSON file
- [ ] CLI can list registered operations
- [ ] FastAPI server starts with health endpoint
- [ ] API can submit thunk for execution
- [ ] API can query thunk status/result
- [ ] Config loads from TOML, env vars override
- [ ] OpenTelemetry tracing captures thunk lifecycle
- [ ] Structured JSON logging in place
- [ ] 80%+ test coverage

**Tests:**
- Unit: Config loads from TOML
- Unit: Config env vars override TOML values
- Integration: CLI --help works
- Integration: CLI executes simple thunk
- Integration: API health endpoint returns 200
- Integration: API submit thunk returns thunk ID
- Integration: API get thunk status returns correct state
- Unit: Tracing creates spans for thunk operations
- Unit: Logging outputs valid JSON

**CLI Commands:**
```bash
crows-nest run <thunk.json>      # Execute thunk from file
crows-nest run -                  # Execute thunk from stdin
crows-nest status <thunk-id>      # Get thunk status
crows-nest result <thunk-id>      # Get thunk result
crows-nest ops list               # List registered operations
crows-nest serve                  # Start API server
crows-nest config show            # Show resolved config
```

**API Endpoints:**
```
POST   /thunks              # Submit thunk for execution
GET    /thunks/{id}         # Get thunk details
GET    /thunks/{id}/status  # Get execution status
GET    /thunks/{id}/result  # Get result (blocks or returns immediately)
GET    /ops                 # List registered operations
GET    /health              # Health check
GET    /ready               # Readiness check
```

**Status:** Not Started

---

## Stage 5: MVP - UNIX Pipeline Agent

**Goal:** End-to-end flow where user requests a task, orchestrator delegates to shell agents, verifies, and reports.

**Success Criteria:**
- [ ] Shell execution thunks (`shell.run`, `shell.pipe`)
- [ ] Safety constraints on shell execution (allowlist, no arbitrary code)
- [ ] `pid0` orchestrator receives task, creates execution plan as thunks
- [ ] Orchestrator delegates to worker agents
- [ ] Verification thunk checks output
- [ ] Full trace from request → delegation → execution → verification → response
- [ ] Works via both CLI and API
- [ ] 80%+ test coverage

**Tests:**
- Unit: `shell.run` executes allowed command
- Unit: `shell.run` rejects disallowed command
- Unit: `shell.pipe` chains commands correctly
- Integration: pid0 receives "list files" → creates ls thunk → executes → returns
- Integration: pid0 receives "count lines in file" → creates cat | wc pipeline → executes
- Integration: Full flow via CLI
- Integration: Full flow via API
- E2E: Docker container can execute MVP flow

**Key Operations:**
```python
@thunk_operation("shell.run")
async def shell_run(
    command: str,
    args: list[str],
    capabilities: set[str],  # must include "shell.run"
) -> ShellResult: ...

@thunk_operation("shell.pipe")
async def shell_pipe(
    commands: list[ShellCommand],
    capabilities: set[str],
) -> ShellResult: ...

@thunk_operation("orchestrate.plan")
async def plan_task(
    task_description: str,
    available_capabilities: set[str],
) -> list[Thunk]: ...

@thunk_operation("verify.output")
async def verify_output(
    expected: VerificationSchema,
    actual: Any,
) -> VerificationResult: ...
```

**Example Flow:**
```
User: "List all Python files and count them"
  ↓
pid0.plan_task("List all Python files and count them")
  ↓
Returns thunks:
  1. shell.run(command="find", args=[".", "-name", "*.py"])
  2. shell.run(command="wc", args=["-l"], stdin=ThunkRef(1))
  3. verify.output(expected={type: "integer", min: 0}, actual=ThunkRef(2))
  ↓
Runtime forces thunks in order
  ↓
pid0 formats result: "Found 42 Python files"
```

**Status:** Not Started

---

## Future Stages (Not Yet Planned)

- **Stage 6:** Plugin system (entry points, directory scanning)
- **Stage 7:** Additional communication patterns (message queue, events)
- **Stage 8:** Agent-creates-agent pattern ("hiring committee")
- **Stage 9:** Persistent agent memory/context management
- **Stage 10:** Web UI for monitoring/debugging

---

## Technical Decisions Reference

| Area | Decision |
|------|----------|
| Python | 3.12+ |
| Package manager | uv |
| LLM abstraction | atomic-agents |
| CLI | Click |
| API | FastAPI |
| Config | TOML + env vars |
| Persistence | SQLite (pluggable) |
| Observability | OpenTelemetry + structured JSON logs |
| Linting | ruff, mypy (strict src), bandit, codespell |
| Testing | pytest, ~80% coverage, fixtures + factories |
| Mocking | JSON Schema-conformant noise |
| Docstrings | Google style |
| Container | Multistage Dockerfile |
| CI | GHA, tests in container |

---

## Commands Reference

```bash
# Development
uv sync                          # Install dependencies
uv run pytest                    # Run tests
uv run pytest --cov              # Run tests with coverage
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy src/                 # Type check
uv run pre-commit run --all-files  # Run all pre-commit hooks

# Docker
docker build -t crows-nest .
docker run crows-nest pytest     # Run tests in container

# Application
crows-nest serve                 # Start API server
crows-nest run task.json         # Execute a thunk
```
