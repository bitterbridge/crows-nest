# Dynamic Plugin Generation via Thunks

## Overview

A system for agents to generate, verify, approve, and load plugins at runtime - all expressed as thunks with explicit dependency chains.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ plugin.generate │ ──▶ │ sandbox.verify  │ ──▶ │  user.approve   │ ──▶ │   plugin.load   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
  PluginArtifact        VerificationReport      ApprovalDecision          LoadResult
    (code, meta)         (tests, behavior)       (yes/no, reason)        (registered)
```

## Goals

1. **Security**: No plugin loads without explicit human approval
2. **Auditability**: Full provenance chain for every loaded plugin
3. **Determinism**: Replaying thunk graphs produces same results (cached approvals)
4. **Composability**: Mix and match verification/approval steps based on risk

## Data Models

### PluginArtifact

The output of plugin generation - code that hasn't been loaded yet.

```python
@dataclass(frozen=True)
class PluginArtifact:
    """Generated plugin code, not yet loaded."""

    id: UUID
    name: str                          # e.g., "json_parser"
    description: str                   # What it does
    source_code: str                   # The actual Python code
    operations: list[OperationSpec]    # What operations it registers
    requested_capabilities: set[str]   # Capabilities it will need
    generated_by: UUID                 # Agent that requested generation
    generated_at: datetime
    generation_context: dict           # Why it was generated (for audit)

    def hash(self) -> str:
        """Content hash for verification."""
        return hashlib.sha256(self.source_code.encode()).hexdigest()


@dataclass(frozen=True)
class OperationSpec:
    """Specification of an operation the plugin will register."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    required_capabilities: set[str]
    return_type: str
```

### VerificationReport

Results from running the plugin in a sandbox.

```python
@dataclass(frozen=True)
class VerificationReport:
    """Results from sandbox verification."""

    artifact_id: UUID
    artifact_hash: str                 # Verify we tested the right code

    # Sandbox execution results
    sandbox_id: str
    executed_at: datetime
    duration_ms: int

    # What happened
    loaded_successfully: bool
    operations_registered: list[str]

    # Behavior analysis
    test_results: list[TestResult]
    resource_usage: ResourceUsage
    violations: list[Violation]

    # Verdict
    passed: bool
    risk_level: RiskLevel              # LOW, MEDIUM, HIGH, CRITICAL
    summary: str


@dataclass(frozen=True)
class TestResult:
    """Result of a single test case."""

    test_name: str
    passed: bool
    input: dict
    expected_output: Any
    actual_output: Any
    error: str | None
    duration_ms: int


@dataclass(frozen=True)
class Violation:
    """Security or resource violation detected."""

    type: ViolationType                # NETWORK, FILESYSTEM, RESOURCE, CAPABILITY
    severity: Severity                 # WARNING, ERROR, CRITICAL
    description: str
    evidence: dict


class RiskLevel(Enum):
    LOW = "low"           # Simple data transformation, no I/O
    MEDIUM = "medium"     # File I/O within working directory
    HIGH = "high"         # Network, shell, or broad file access
    CRITICAL = "critical" # Requests dangerous capabilities
```

### ApprovalDecision

Human decision on whether to load the plugin.

```python
@dataclass(frozen=True)
class ApprovalDecision:
    """Human decision on plugin approval."""

    artifact_id: UUID
    artifact_hash: str                 # What was approved
    verification_id: UUID | None       # Link to verification (if done)

    # Decision
    approved: bool
    reason: str                        # Human explanation
    conditions: list[str]              # Any conditions on approval

    # Audit
    decided_by: str                    # User identifier
    decided_at: datetime
    decision_context: dict             # What info was shown to user

    # Scope
    scope: ApprovalScope               # How broadly does this apply?
    expires_at: datetime | None        # Optional expiration


class ApprovalScope(Enum):
    ONCE = "once"                      # This execution only
    SESSION = "session"                # Until restart
    PERMANENT = "permanent"            # Persisted approval
    HASH_PERMANENT = "hash_permanent"  # Permanent for this exact code hash
```

### LoadResult

Result of actually loading the plugin.

```python
@dataclass(frozen=True)
class LoadResult:
    """Result of loading a plugin."""

    artifact_id: UUID
    approval_id: UUID

    loaded: bool
    error: str | None

    # What was registered
    operations_registered: list[str]
    capabilities_granted: set[str]

    # Runtime info
    loaded_at: datetime
    load_id: UUID                      # Unique ID for this load instance
```

## Operations

### plugin.generate

Generate plugin code from a specification.

```python
@registry.operation(
    "plugin.generate",
    required_capabilities=frozenset({"plugin.generate"}),
)
async def generate_plugin(
    spec: str,                         # Natural language description
    target_operations: list[str],      # Operation names to create
    example_inputs: list[dict],        # Example inputs for testing
    example_outputs: list[Any],        # Expected outputs
    *,
    llm: LLMProvider,
    context: AgentContext,
) -> PluginArtifact:
    """
    Generate plugin code from specification.

    The LLM generates Python code that:
    1. Defines operation handler functions
    2. Uses only standard library + allowed dependencies
    3. Follows the plugin contract

    Does NOT load the plugin - just produces the artifact.
    """
    ...
```

### sandbox.verify

Run the plugin in an isolated environment.

```python
@registry.operation(
    "sandbox.verify",
    required_capabilities=frozenset({"sandbox.run"}),
)
async def verify_in_sandbox(
    artifact: PluginArtifact,
    test_cases: list[TestCase] | None = None,
    timeout_seconds: float = 30.0,
    resource_limits: ResourceLimits | None = None,
) -> VerificationReport:
    """
    Execute plugin in sandbox and analyze behavior.

    Sandbox provides:
    - Isolated filesystem (tmpfs)
    - No network access
    - Resource limits (CPU, memory, time)
    - Capability interception (logs what it tries to do)

    Runs:
    1. Load plugin, verify it registers expected operations
    2. Run test cases (from artifact + provided)
    3. Monitor resource usage
    4. Detect any violations
    """
    ...
```

### user.approve

Request human approval with full context.

```python
@registry.operation(
    "user.approve",
    required_capabilities=frozenset({"user.prompt"}),
)
async def request_approval(
    request_type: str,                 # "plugin.load", "shell.dangerous", etc.
    description: str,                  # What's being requested
    artifact: PluginArtifact | None,   # The plugin (if applicable)
    verification: VerificationReport | None,  # Verification results
    risk_level: RiskLevel,
    additional_context: dict,
    timeout_seconds: float | None = None,  # None = wait forever
) -> ApprovalDecision:
    """
    Request human approval for an action.

    Presents to user:
    - What: description of the request
    - Why: context from the agent
    - Risk: verification results, risk level
    - Code: source code (for plugins)

    Blocks until user responds or timeout.

    User can:
    - Approve (with optional conditions)
    - Deny (with reason)
    - Request more info (returns to agent)
    """
    ...
```

### plugin.load

Load an approved plugin.

```python
@registry.operation(
    "plugin.load",
    required_capabilities=frozenset({"plugin.load"}),
)
async def load_plugin(
    artifact: PluginArtifact,
    approval: ApprovalDecision,
    scope: LoadScope = LoadScope.SESSION,
) -> LoadResult:
    """
    Load an approved plugin into the registry.

    Preconditions:
    - approval.approved must be True
    - approval.artifact_hash must match artifact.hash()
    - approval must not be expired

    The plugin is loaded into:
    - SESSION: Current runtime only, lost on restart
    - PERSISTENT: Saved to plugins directory, reloaded on restart
      (requires separate approval for auto-reload)
    """
    ...
```

## User Interaction

### CLI Integration

```
$ crows-nest serve

[12:34:56] Approval requested: Load plugin 'json_parser'
           Requested by: agent-abc123
           Risk level: LOW

           Description: Parse and validate JSON with custom schema

           Operations:
             - json.parse(data: str) -> dict
             - json.validate(data: dict, schema: dict) -> bool

           Verification: PASSED (4/4 tests, no violations)

           [A]pprove  [D]eny  [V]iew code  [?] More info
           >
```

### Dashboard Integration

- Real-time approval queue in dashboard
- Full code review interface
- Verification report visualization
- Approval history and audit log

### API Integration

```python
# Webhook for external approval systems
POST /api/approvals/pending
GET /api/approvals/{id}
POST /api/approvals/{id}/decide
```

## Capability Model

```
Capability              Required For              Typical Holder
─────────────────────────────────────────────────────────────────
plugin.generate         Generate plugin code      Worker agents (restricted)
sandbox.run             Execute in sandbox        Verification agents
user.prompt             Request human input       Orchestrator only
plugin.load             Load into registry        Orchestrator only
plugin.reload           Reload on restart         Root only (manual)
```

## Persistence & Restart

### What's Persisted

1. **PluginArtifacts**: Stored in `plugins/artifacts/{id}.json`
2. **VerificationReports**: Stored with thunk results
3. **ApprovalDecisions**: Stored in `plugins/approvals/{id}.json`
4. **LoadRecords**: Which plugins were loaded, stored in `plugins/loaded.json`

### On Restart

1. Previously loaded plugins are NOT auto-loaded
2. To reload, either:
   - Run `crows-nest plugins reload` (prompts for re-approval)
   - Create explicit `plugin.reload` thunk referencing original approval
3. Permanent approvals (HASH_PERMANENT) skip re-prompting if code unchanged

### plugin.reload Operation

```python
@registry.operation(
    "plugin.reload",
    required_capabilities=frozenset({"plugin.reload"}),
)
async def reload_plugin(
    artifact_id: UUID,
    require_reverification: bool = False,
) -> LoadResult:
    """
    Reload a previously approved plugin.

    Checks:
    - Original approval exists and is valid
    - Code hash matches (if HASH_PERMANENT)
    - Re-verifies if require_reverification=True

    If approval was ONCE or SESSION, prompts for new approval.
    """
    ...
```

## Example Flow

### Agent Needs a JSON Parser

```python
# 1. Agent realizes it needs JSON parsing
need = "Parse JSON API responses with nested structure validation"

# 2. Generate plugin (creates PluginArtifact)
gen_thunk = Thunk.create(
    "plugin.generate",
    {
        "spec": need,
        "target_operations": ["json.parse", "json.validate"],
        "example_inputs": [...],
        "example_outputs": [...],
    },
    capabilities={"plugin.generate"},
)

# 3. Verify in sandbox (creates VerificationReport)
verify_thunk = Thunk.create(
    "sandbox.verify",
    {"artifact": ThunkRef(gen_thunk.id)},
    capabilities={"sandbox.run"},
)

# 4. Request approval (creates ApprovalDecision)
approve_thunk = Thunk.create(
    "user.approve",
    {
        "request_type": "plugin.load",
        "description": "Load JSON parsing plugin",
        "artifact": ThunkRef(gen_thunk.id),
        "verification": ThunkRef(verify_thunk.id),
        "risk_level": "LOW",
    },
    capabilities={"user.prompt"},
)

# 5. Load if approved (creates LoadResult)
load_thunk = Thunk.create(
    "plugin.load",
    {
        "artifact": ThunkRef(gen_thunk.id),
        "approval": ThunkRef(approve_thunk.id),
    },
    capabilities={"plugin.load"},
)

# Execute the chain
result = await runtime.force_all(
    {t.id: t for t in [gen_thunk, verify_thunk, approve_thunk, load_thunk]},
    target=load_thunk.id,
)
```

## Security Considerations

### Sandbox Isolation

- Use `seccomp` or similar for syscall filtering
- No network access by default
- Restricted filesystem (tmpfs only)
- Resource limits (CPU, memory, file descriptors)
- No access to environment variables or secrets

### Code Analysis

Before sandbox execution, static analysis:
- No `eval`, `exec`, `compile` with dynamic input
- No `__import__` with dynamic names
- No subprocess/os.system calls
- Whitelist of allowed imports

### Approval Context

Always show user:
- Full source code
- What capabilities it requests
- What the verification found
- Which agent requested it and why

### Audit Trail

Every plugin load has immutable record:
- Who generated it
- What tests were run
- Who approved it
- When it was loaded
- What it's been used for

## Future Extensions

### Capability Escalation Requests

```python
@registry.operation("capability.request")
async def request_capability(
    capability: str,
    reason: str,
    duration: timedelta | None,
) -> CapabilityGrant:
    """Agent requests temporary capability escalation."""
    ...
```

### Plugin Revocation

```python
@registry.operation("plugin.revoke")
async def revoke_plugin(
    load_id: UUID,
    reason: str,
) -> RevokeResult:
    """Immediately unload a plugin and invalidate its approval."""
    ...
```

### Automated Approval Policies

```yaml
# approval-policies.yaml
policies:
  - name: "auto-approve-low-risk"
    conditions:
      risk_level: LOW
      verification: PASSED
      operations_match: true
    action: APPROVE
    scope: HASH_PERMANENT

  - name: "deny-network-access"
    conditions:
      requested_capabilities:
        contains: "network.*"
    action: DENY
    reason: "Network access not permitted for generated plugins"
```

## Implementation Plan

### Stage 17: Dynamic Plugin Foundation

1. Implement `PluginArtifact` and related models
2. Implement `plugin.generate` operation (LLM integration)
3. Basic artifact storage

### Stage 18: Sandbox Verification

1. Implement sandbox execution environment
2. Implement `sandbox.verify` operation
3. Test case generation from examples

### Stage 19: Approval Flow

1. Implement `user.approve` operation
2. CLI approval interface
3. Dashboard approval queue
4. Approval persistence

### Stage 20: Plugin Loading

1. Implement `plugin.load` operation
2. Implement `plugin.reload` operation
3. Restart behavior
4. Audit logging

---

*This design keeps everything as thunks - auditable, composable, and explicit. No magic, just a clear chain of custody from "agent wants capability" to "capability is available".*
