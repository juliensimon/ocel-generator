"""Generate Hugging Face dataset card for the unified agent traces dataset."""

from __future__ import annotations

from ocelgen.scenarios.domain import DomainScenario

GITHUB_URL = "https://github.com/juliensimon/ocel-generator"


def generate_dataset_card(
    scenarios: list[DomainScenario],
    namespace: str,
    domain_stats: dict[str, dict],
) -> str:
    """Generate a unified HF dataset card for all domains.

    Args:
        scenarios: List of all domain scenarios.
        namespace: HF namespace (e.g. "juliensimon").
        domain_stats: {domain_name: {"num_events": int, "num_objects": int}}
    """
    repo_name = f"{namespace}/open-agent-traces"
    total_events = sum(s["num_events"] for s in domain_stats.values())
    sum(s["num_objects"] for s in domain_stats.values())
    total_runs = sum(s.runs for s in scenarios)

    # Build configs YAML
    configs_yaml = ""
    for s in scenarios:
        stats = domain_stats.get(s.name, {})
        configs_yaml += f"""  - config_name: {s.name}
    data_files:
      - split: train
        path: data/{s.name}/train.parquet
"""

    # Build domain table
    domain_rows = ""
    for s in scenarios:
        stats = domain_stats.get(s.name, {})
        events = stats.get("num_events", 0)
        domain_rows += f"| `{s.name}` | {s.pattern} | {s.runs} | {s.noise:.0%} | {events:,} | {s.description} |\n"

    # Build per-domain details
    domain_details = ""
    for s in scenarios:
        agent_rows = ""
        for role, persona in s.agent_personas.items():
            agent_rows += f"  | `{role}` | {persona} |\n"
        tool_rows = ""
        for tool, desc in s.tool_descriptions.items():
            tool_rows += f"  | `{tool}` | {desc} |\n"

        domain_details += f"""
<details>
<summary><strong>{s.name}</strong> ({s.pattern})</summary>

  {s.description}

  | Agent | Persona |
  |-------|---------|
{agent_rows}
  | Tool | Description |
  |------|-------------|
{tool_rows}
  Example queries:
{chr(10).join(f'  - "{q}"' for q in s.user_queries[:5])}

</details>
"""

    return f"""---
configs:
{configs_yaml}
license: mit
language:
  - en
task_categories:
  - text-generation
  - text-classification
tags:
  - agent-traces
  - ocel
  - multi-agent
  - process-mining
  - synthetic
  - llm-agents
  - conformance-checking
  - ai-agents
  - workflow-traces
  - agent-observability
  - tool-use
  - chain-of-thought
  - anomaly-detection
pretty_name: Open Agent Traces
size_categories:
  - 10K<n<100K
---

# Open Agent Traces — Synthetic Multi-Agent Workflow Dataset

**{total_events:,} LLM-enriched agent trace events** across **{total_runs} workflow runs** in **10 domains** and **3 workflow patterns** (sequential, supervisor, parallel).

Built for teams developing AI agent infrastructure: observability platforms, evaluation frameworks, process mining tools, and anomaly detection systems.

> Source code and generator: **[ocelgen]({GITHUB_URL})** on GitHub

![Parallel workflow trace example](docs/parallel-workflow-example.png)

```python
from datasets import load_dataset

ds = load_dataset("{repo_name}", "incident-response")

for event in ds["train"]:
    if event["run_id"] == "run-0000":
        print(f"{{event['event_type']:25s}} | {{event['agent_role']:12s}} | {{event['reasoning'][:60] if event['reasoning'] else ''}}")
```

## What's inside

Each trace captures the full execution of a multi-agent workflow — the same data you'd see in production agent observability tools:

- **Agent reasoning** — chain-of-thought for every agent step
- **LLM prompts and completions** — realistic request/response pairs with calibrated token counts
- **Tool calls with inputs and outputs** — structured JSON for each tool invocation
- **Inter-agent messages** — handoff content between workflow steps
- **Deviation labels** — ground-truth annotations marking conformant vs anomalous behavior
- **Realistic timestamps** — seconds-scale LLM latencies, not synthetic milliseconds

The traces follow the **[OCEL 2.0](https://www.ocel-standard.org/) standard** (Object-Centric Event Logs), making them compatible with process mining tools and conformance checking algorithms.

## Domains

10 configurations, each representing a different domain and workflow pattern:

| Config | Pattern | Runs | Noise | Events | Description |
|--------|---------|------|-------|--------|-------------|
{domain_rows}

**Workflow patterns:**
- **sequential** — linear chain of agents (A → B → C)
- **supervisor** — central agent delegates to specialist workers
- **parallel** — fan-out to concurrent agents, then aggregate

{domain_details}

## Schema

Each row is one event in the OCEL 2.0 trace:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | string | Unique event identifier |
| `event_type` | string | `run_started`, `agent_invoked`, `llm_request_sent`, `llm_response_received`, `tool_called`, `tool_returned`, `message_sent`, `routing_decided`, `agent_completed`, `run_completed`, `error_occurred`, `retry_started` |
| `timestamp` | string | ISO 8601 with realistic inter-event durations (seconds-scale) |
| `run_id` | string | Workflow run identifier |
| `sequence_number` | int | Monotonic order within the run |
| `is_deviation` | bool | Whether this event is part of an injected deviation |
| `deviation_type` | string | `skipped_activity`, `inserted_activity`, `wrong_resource`, `swapped_order`, `wrong_tool`, `repeated_activity`, `timeout`, `wrong_routing`, `missing_handoff`, `extra_llm_call` |
| `step_id` | string | Workflow step identifier |
| `agent_role` | string | Agent role (e.g. `researcher`, `supervisor`, `coder`) |
| `model_name` | string | LLM model (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `prompt` | string | LLM prompt text (on `llm_response_received` events) |
| `completion` | string | LLM completion text |
| `tool_name` | string | Name of the tool called |
| `tool_input` | string | Tool input as JSON |
| `tool_output` | string | Tool output as JSON |
| `message_content` | string | Inter-agent handoff message |
| `reasoning` | string | Agent chain-of-thought reasoning |
| `input_tokens` | int | Input token count (calibrated to content) |
| `output_tokens` | int | Output token count (calibrated to content) |
| `latency_ms` | int | LLM or tool call latency in ms |
| `cost_usd` | float | Estimated invocation cost |
| `is_conformant` | bool | Whether the run follows the expected workflow |
| `pattern` | string | `sequential`, `supervisor`, or `parallel` |
| `domain` | string | Domain name (same as config name) |
| `user_query` | string | User request that initiated the run |

## Usage examples

```python
from datasets import load_dataset

# Load one domain
ds = load_dataset("{repo_name}", "customer-support-triage")

# Get all LLM completions
completions = ds["train"].filter(lambda x: x["event_type"] == "llm_response_received")
for row in completions:
    print(f"Prompt: {{row['prompt'][:100]}}...")
    print(f"Completion: {{row['completion'][:100]}}...")

# Analyze deviations
deviant = ds["train"].filter(lambda x: x["is_deviation"])
print(f"Deviation types: {{set(e for e in deviant['deviation_type'] if e)}}")

# Cross-domain comparison
for domain in ["customer-support-triage", "incident-response", "code-review-pipeline"]:
    ds = load_dataset("{repo_name}", domain)
    agents = set(row["agent_role"] for row in ds["train"] if row["agent_role"])
    print(f"{{domain}}: {{agents}}")
```

## Use cases

- **Agent observability and debugging** — build and test monitoring dashboards for multi-agent workflows, with the same data platforms like LangSmith, Arize, and Braintrust display
- **Agent evaluation and benchmarking** — compare agent reasoning across sequential (LangChain-style), supervisor (CrewAI-style), and parallel (LangGraph-style) architectures
- **Conformance checking and anomaly detection** — train models to detect deviant agent behavior using labeled ground-truth deviations
- **Process mining** — apply OCEL 2.0 conformance checking algorithms to multi-agent systems
- **Agent framework testing** — validate orchestration frameworks against realistic trace data across 10 enterprise domains

## Files per domain

| Path | Format | Description |
|------|--------|-------------|
| `data/{{domain}}/train.parquet` | Parquet | Flat tabular (one row per event) |
| `ocel/{{domain}}/output.jsonocel` | OCEL 2.0 JSON | Native object-centric event log |
| `ocel/{{domain}}/normative_model.json` | JSON | Expected workflow template |
| `ocel/{{domain}}/manifest.json` | JSON | Generation metadata + deviation ground truth |

## How it was built

Generated with **[ocelgen]({GITHUB_URL})** — a two-pass architecture:

1. **Structural generation** — OCEL 2.0 traces with configurable workflow patterns, deviation injection (10 types), and deterministic seeding
2. **LLM enrichment** — each agent step enriched via [OpenRouter](https://openrouter.ai) with domain-specific prompts; outputs chain across steps for coherence

Quality measures:
- Token counts calibrated to actual content length (1.3x word-to-token ratio)
- Realistic timestamps (seconds-scale LLM latencies)
- 50 unique queries per domain (LLM-expanded from seed set)
- Deviation-aware content (deviant steps reflect failures in their reasoning)
- Parallel aggregator coherence (aggregator sees all workers' outputs)

## Citation

```bibtex
@misc{{open-agent-traces-2026,
  title={{Open Agent Traces: Synthetic Multi-Agent Workflow Datasets}},
  author={{Julien Simon}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

MIT — source code at [{GITHUB_URL}]({GITHUB_URL})
"""


# Keep single-domain version for backward compat with tests
def generate_single_domain_card(
    scenario: DomainScenario,
    namespace: str,
    num_events: int,
    num_objects: int,
) -> str:
    """Generate a dataset card for a single domain (used in tests)."""
    return generate_dataset_card(
        scenarios=[scenario],
        namespace=namespace,
        domain_stats={scenario.name: {"num_events": num_events, "num_objects": num_objects}},
    )
