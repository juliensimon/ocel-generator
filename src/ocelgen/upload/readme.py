"""Generate Hugging Face dataset card (README.md) for agent trace datasets."""

from __future__ import annotations

from ocelgen.scenarios.domain import DomainScenario


def generate_dataset_card(
    scenario: DomainScenario,
    namespace: str,
    num_events: int,
    num_objects: int,
) -> str:
    """Generate a HF dataset card markdown string."""
    repo_name = f"{namespace}/agent-traces-{scenario.name}"

    return f"""---
dataset_info:
  features:
    - name: event_id
      dtype: string
    - name: event_type
      dtype: string
    - name: timestamp
      dtype: string
    - name: run_id
      dtype: string
    - name: sequence_number
      dtype: int64
    - name: is_deviation
      dtype: bool
    - name: deviation_type
      dtype: string
    - name: step_id
      dtype: string
    - name: agent_role
      dtype: string
    - name: model_name
      dtype: string
    - name: prompt
      dtype: string
    - name: completion
      dtype: string
    - name: tool_name
      dtype: string
    - name: tool_input
      dtype: string
    - name: tool_output
      dtype: string
    - name: message_content
      dtype: string
    - name: reasoning
      dtype: string
    - name: input_tokens
      dtype: int64
    - name: output_tokens
      dtype: int64
    - name: latency_ms
      dtype: int64
    - name: cost_usd
      dtype: float64
    - name: is_conformant
      dtype: bool
    - name: pattern
      dtype: string
    - name: domain
      dtype: string
    - name: user_query
      dtype: string
  splits:
    - name: train
      num_examples: {num_events}
license: mit
tags:
  - agent-traces
  - ocel
  - multi-agent
  - process-mining
  - synthetic
---

# Agent Traces: {scenario.name}

Synthetic multi-agent workflow traces with LLM-enriched content for the **{scenario.name}** domain.

## Description

{scenario.description}

- **Workflow pattern:** {scenario.pattern}
- **Runs:** {scenario.runs}
- **Noise rate:** {scenario.noise} (fraction of runs with injected deviations)
- **Events:** {num_events}
- **Objects:** {num_objects}
- **Seed:** {scenario.seed} (reproducible)

## Schema

Each row represents one event in the OCEL 2.0 trace:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | string | Unique event identifier |
| `event_type` | string | Event type (e.g. `agent_invoked`, `llm_request_sent`) |
| `timestamp` | string | ISO 8601 timestamp |
| `run_id` | string | Which workflow run this event belongs to |
| `sequence_number` | int | Order within the run |
| `is_deviation` | bool | Whether this event is part of an injected deviation |
| `deviation_type` | string | Type of deviation (if any) |
| `agent_role` | string | Role of the agent (resolved from relationship) |
| `prompt` | string | LLM prompt text (enriched) |
| `completion` | string | LLM completion text (enriched) |
| `tool_name` | string | Tool that was called |
| `tool_input` | string | Tool input as JSON (enriched) |
| `tool_output` | string | Tool output as JSON (enriched) |
| `reasoning` | string | Agent chain-of-thought reasoning (enriched) |
| `is_conformant` | bool | Whether the run follows the normative workflow |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_name}")
print(ds["train"][0])

# Filter to just agent invocations
agent_events = ds["train"].filter(lambda x: x["event_type"] == "agent_invoked")

# Get all deviant runs
deviant = ds["train"].filter(lambda x: not x["is_conformant"])
```

## Files

- `data/train.parquet` — Flat tabular format (one row per event)
- `ocel/output.jsonocel` — Native OCEL 2.0 JSON format
- `ocel/normative_model.json` — Expected workflow template
- `ocel/manifest.json` — Generation metadata and deviation ground truth

## Generation

Generated with [ocelgen](https://github.com/juliensimon/ocel-generator) using two-pass architecture:
1. Structural OCEL 2.0 trace generation with configurable deviation injection
2. LLM enrichment via OpenRouter for realistic prompts, completions, and tool I/O

Part of the [{namespace}/open-agent-traces](https://huggingface.co/collections/{namespace}/open-agent-traces) collection.

## License

MIT
"""
