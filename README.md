# ocel-generator

A Python CLI tool for generating synthetic [OCEL 2.0](https://www.ocel-standard.org/) event logs that simulate LangChain multi-agent workflow executions. Designed for testing process mining and conformance checking algorithms.

## Features

- **OCEL 2.0 compliant** output validated against the official JSON schema
- **Multiple workflow patterns**: sequential, supervisor, and parallel agent architectures
- **Configurable deviation injection** for conformance checking scenarios
- **Reproducible generation** with seed support
- **Rich CLI** with progress feedback and summary tables

## Installation

```bash
# Clone the repository
git clone https://github.com/juliensimon/ocel-generator.git
cd ocel-generator

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

For conformance checking support with pm4py:
```bash
uv sync --extra conformance
```

## Quick Start

Generate 100 sequential workflow runs with 20% noise:

```bash
ocelgen generate --pattern sequential --runs 100 --noise 0.2
```

This produces three files:
- `output.jsonocel` — the OCEL 2.0 event log
- `normative_model.json` — the expected workflow template
- `manifest.json` — generation metadata and injected deviations

## CLI Commands

### generate

Generate synthetic OCEL 2.0 event logs.

```bash
ocelgen generate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --pattern` | `sequential` | Workflow pattern (`sequential`, `supervisor`, `parallel`) |
| `-n, --runs` | `100` | Number of workflow runs to generate |
| `-N, --noise` | `0.2` | Fraction of runs with deviations (0.0–1.0) |
| `--max-deviations` | `3` | Maximum deviations per deviant run |
| `--seed` | random | Random seed for reproducibility |
| `-o, --output` | `output.jsonocel` | Output file path |

### validate

Validate an OCEL 2.0 JSON file against the official schema.

```bash
ocelgen validate path/to/file.jsonocel
```

### list-patterns

List available workflow patterns with descriptions.

```bash
ocelgen list-patterns
```

## Workflow Patterns

| Pattern | Description |
|---------|-------------|
| `sequential` | Linear chain of agents passing results forward |
| `supervisor` | Central supervisor delegates to worker agents |
| `parallel` | Agents execute concurrently with fan-out/fan-in |

## Deviation Types

The generator can inject various deviations to create non-conformant traces:

| Deviation | Description |
|-----------|-------------|
| `skipped_activity` | Required step omitted from execution |
| `inserted_activity` | Unexpected step added to workflow |
| `wrong_resource` | Step executed by incorrect agent |
| `swapped_order` | Steps executed out of expected order |
| `wrong_tool` | Incorrect tool used for a step |
| `repeated_activity` | Step executed multiple times |
| `timeout` | Step exceeded expected duration |
| `wrong_routing` | Incorrect routing decision in supervisor pattern |
| `missing_handoff` | Agent handoff not properly recorded |
| `extra_llm_call` | Unexpected LLM invocation |

## Output Format

### OCEL 2.0 Event Log

The generated `.jsonocel` file follows the [OCEL 2.0 JSON specification](https://www.ocel-standard.org/) with:

- **Object types**: `run`, `agent`, `tool`, `llm`
- **Event types**: `start_run`, `llm_call`, `tool_call`, `agent_handoff`, `end_run`, etc.
- **Relationships**: Events linked to relevant objects via `o2o` and `e2o` mappings

### Normative Model

The `normative_model.json` contains the expected workflow template for conformance checking, including step sequences, allowed agents, and expected tools.

### Manifest

The `manifest.json` records generation parameters and a complete list of injected deviations with their locations, enabling ground-truth evaluation of conformance checking algorithms.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Type checking
mypy src

# Linting
ruff check src tests
```

## License

MIT
