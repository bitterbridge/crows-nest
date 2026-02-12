# CLI Reference

Complete reference for the Crow's Nest command-line interface.

## Installation

After installing Crow's Nest, the CLI is available as:

```bash
crows-nest <command>   # Full name
cn <command>           # Short alias
```

## Global Options

| Option | Description |
|--------|-------------|
| `--config, -c PATH` | Path to config file |
| `--version` | Show version |
| `--help` | Show help message |

## Commands

### ask

Execute a natural language task.

```bash
crows-nest ask "your task here" [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--verbose, -v` | Show execution progress |
| `--json` | Output result as JSON |

**Examples:**
```bash
# Simple task
crows-nest ask "list python files"

# With progress output
crows-nest ask "count lines in all .py files" -v

# JSON output for scripting
crows-nest ask "show current directory" --json
```

---

### serve

Start the API server.

```bash
crows-nest serve [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--host, -h HOST` | Host to bind to |
| `--port, -p PORT` | Port to bind to |
| `--reload` | Enable auto-reload for development |

**Examples:**
```bash
# Start with defaults
crows-nest serve

# Custom host and port
crows-nest serve -h 0.0.0.0 -p 9000

# Development mode with auto-reload
crows-nest serve --reload
```

---

### run

Execute a thunk from a JSON file.

```bash
crows-nest run [THUNK_FILE] [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `THUNK_FILE` | JSON file with thunk definition (default: stdin) |

**Options:**
| Option | Description |
|--------|-------------|
| `--wait/--no-wait` | Wait for completion (default: wait) |

**Examples:**
```bash
# Run from file
crows-nest run my_thunk.json

# Run from stdin
echo '{"operation": "shell.run", ...}' | crows-nest run -

# Submit without waiting
crows-nest run my_thunk.json --no-wait
```

---

### status

Get the status of a thunk.

```bash
crows-nest status THUNK_ID
```

**Output:**
```json
{
  "thunk_id": "...",
  "status": "success",
  "forced_at": "2024-...",
  "duration_ms": 42
}
```

---

### result

Get the result of a thunk.

```bash
crows-nest result THUNK_ID
```

---

### ops list

List all registered operations.

```bash
crows-nest ops list
```

**Output:**
```
shell.run
  Execute a shell command safely
  Requires: shell.run
  Parameters:
    - command: str (required)
    - args: list[str] (default: [])
```

---

### config show

Show the resolved configuration.

```bash
crows-nest config show
```

### config path

Show config file search paths and which file is in use.

```bash
crows-nest config path
```

---

### plugins list

List all loaded plugins.

```bash
crows-nest plugins list
```

### plugins info

Show detailed information about a plugin.

```bash
crows-nest plugins info PLUGIN_NAME
```

---

### queues list

List all message queues.

```bash
crows-nest queues list
```

### queues stats

Show detailed statistics for a queue.

```bash
crows-nest queues stats QUEUE_NAME
```

### queues dead-letters

Show dead letter messages for a queue.

```bash
crows-nest queues dead-letters QUEUE_NAME
```

---

### agents list

List all agents.

```bash
crows-nest agents list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--all, -a` | Include terminated agents |

### agents tree

Show agent hierarchy as a tree.

```bash
crows-nest agents tree [AGENT_ID]
```

If `AGENT_ID` is provided, shows tree rooted at that agent. Otherwise shows all roots.

**Output:**
```
Agent hierarchy (1 root):
└── ● a1b2c3d4... (depth=0)
    ├── ● e5f6g7h8... (depth=1)
    │   └── ○ i9j0k1l2... (depth=2)
    └── ● m3n4o5p6... (depth=1)
```

### agents info

Show detailed information about an agent.

```bash
crows-nest agents info AGENT_ID
```

### agents capabilities

Show capabilities for an agent.

```bash
crows-nest agents capabilities AGENT_ID
```

### agents terminate

Terminate an agent.

```bash
crows-nest agents terminate AGENT_ID [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--cascade/--no-cascade` | Also terminate children (default: cascade) |
| `--force, -f` | Skip confirmation prompt |

---

### memory list

List memories for an agent.

```bash
crows-nest memory list AGENT_ID [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--type, -t TYPE` | Filter by memory type (episodic, semantic, procedural) |
| `--scope, -s SCOPE` | Filter by scope |
| `--limit, -n N` | Maximum results (default: 10) |
| `--min-importance FLOAT` | Minimum importance threshold |

### memory stats

Show memory statistics for an agent.

```bash
crows-nest memory stats AGENT_ID
```

### memory get

Get a specific memory by ID.

```bash
crows-nest memory get AGENT_ID MEMORY_ID
```

### memory forget

Delete a specific memory.

```bash
crows-nest memory forget AGENT_ID MEMORY_ID [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--force, -f` | Skip confirmation |

### memory decay

Remove low-importance memories based on decay score.

```bash
crows-nest memory decay AGENT_ID [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--threshold FLOAT` | Decay threshold (default: 0.1) |
| `--half-life FLOAT` | Half-life in seconds (default: 86400) |
| `--dry-run` | Show what would be deleted |

---

## Configuration

The CLI uses the following config file search order:

1. `./config.toml`
2. `./crows_nest.toml`
3. `~/.config/crows-nest/config.toml`
4. `/etc/crows-nest/config.toml`

Override with `--config` or environment variables.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |

## Examples

### Complete Workflow

```bash
# Start the server
crows-nest serve &

# Execute a task
crows-nest ask "count python files in src/"

# Check thunk history via dashboard
open http://localhost:8000/dashboard

# List agents
crows-nest agents list

# Show agent memory
crows-nest memory list <agent-id> --type episodic
```

### Scripting with JSON Output

```bash
# Get result as JSON
result=$(crows-nest ask "list files" --json)

# Parse with jq
echo "$result" | jq '.output'

# Check success
if echo "$result" | jq -e '.success' > /dev/null; then
  echo "Task succeeded"
fi
```
