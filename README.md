# ocelgen — Synthetic Agent Traces Dataset Generator

Generate realistic, LLM-enriched multi-agent workflow trace datasets in [OCEL 2.0](https://www.ocel-standard.org/) format. Designed for building agent observability tools, testing process mining algorithms, and training anomaly detection models.

**Dataset on Hugging Face:** [`juliensimon/open-agent-traces`](https://huggingface.co/datasets/juliensimon/open-agent-traces) — 17,000+ events across 10 domains

## Why synthetic agent traces?

Real agent traces from production systems are scarce, proprietary, and hard to share. ocelgen fills this gap by generating structurally valid, semantically rich traces that look like real multi-agent executions — complete with LLM prompts, completions, tool calls, agent reasoning, and labeled deviations.

## Features

- **OCEL 2.0 compliant** output validated against the official JSON schema
- **LLM-enriched content** — realistic prompts, completions, tool I/O, and chain-of-thought reasoning generated via OpenRouter
- **10 built-in domains** — customer support, code review, market research, legal analysis, data pipeline debugging, content generation, financial analysis, incident response, academic paper review, e-commerce product enrichment
- **3 workflow patterns** — sequential chains, supervisor/worker delegation, parallel fan-out/fan-in
- **Configurable deviation injection** — 10 deviation types (skipped steps, wrong tools, timeouts, etc.) with ground-truth labels for conformance checking
- **Reproducible generation** with deterministic seeding
- **Hugging Face Hub integration** — generate, enrich, and upload datasets in one command

## Quick start

```bash
# Install
git clone https://github.com/juliensimon/ocel-generator.git
cd ocel-generator
uv sync

# Generate structural traces
ocelgen generate --pattern sequential --runs 100 --noise 0.2

# Enrich with LLM content (requires OPENAI_API_KEY for OpenRouter)
ocelgen enrich output.jsonocel --domain customer-support-triage

# Or run the full pipeline: generate + enrich + upload to HF
ocelgen pipeline --domain customer-support-triage --namespace your-hf-username

# Generate all 10 domains
ocelgen pipeline --all --namespace your-hf-username
```

## Using the pre-built dataset

The dataset is available on Hugging Face with 10 domain configurations:

```python
from datasets import load_dataset

# Load a specific domain
ds = load_dataset("juliensimon/open-agent-traces", "incident-response")

# Browse a workflow run
for event in ds["train"]:
    if event["run_id"] == "run-0000":
        print(f"{event['event_type']:25s} | {event['agent_role']:12s} | {event['reasoning'][:60] if event['reasoning'] else ''}")

# Analyze deviations
deviant = ds["train"].filter(lambda x: x["is_deviation"])
print(f"Deviation types: {set(e for e in deviant['deviation_type'] if e)}")
```

Available domains: `customer-support-triage`, `code-review-pipeline`, `market-research`, `legal-document-analysis`, `data-pipeline-debugging`, `content-generation`, `financial-analysis`, `incident-response`, `academic-paper-review`, `ecommerce-product-enrichment`

## CLI commands

| Command | Description |
|---------|-------------|
| `ocelgen generate` | Generate structural OCEL 2.0 event logs |
| `ocelgen enrich` | Enrich traces with LLM-generated content |
| `ocelgen upload` | Upload enriched traces to Hugging Face Hub |
| `ocelgen pipeline` | End-to-end: generate + enrich + upload |
| `ocelgen validate` | Validate OCEL 2.0 JSON against the schema |
| `ocelgen list-patterns` | List available workflow patterns |
| `ocelgen list-domains` | List available domain scenarios |

## Workflow patterns

| Pattern | Description | Agents |
|---------|-------------|--------|
| `sequential` | Linear chain: A &rarr; B &rarr; C | 3 agents |
| `supervisor` | Central supervisor delegates to workers | 5 agents |
| `parallel` | Fan-out to concurrent agents, then aggregate | 5 agents |

## Deviation types for conformance checking

ocelgen injects labeled deviations into traces, creating ground-truth data for evaluating conformance checking algorithms:

| Deviation | Description |
|-----------|-------------|
| `skipped_activity` | Required step omitted |
| `inserted_activity` | Unexpected step added |
| `wrong_resource` | Step handled by wrong agent |
| `swapped_order` | Steps executed out of order |
| `wrong_tool` | Incorrect tool used |
| `repeated_activity` | Step executed multiple times |
| `timeout` | Step exceeded expected duration |
| `wrong_routing` | Incorrect supervisor routing |
| `missing_handoff` | Agent handoff not recorded |
| `extra_llm_call` | Unnecessary LLM invocation |

## Architecture

Two-pass generation:

1. **Structural pass** — generates OCEL 2.0 compliant traces with events, objects, and relationships. Deviation injection mutates conformant traces to create labeled anomalies. Fast, free, deterministic.

2. **Enrichment pass** — walks each trace and calls an LLM (via OpenRouter) to fill in realistic content: prompts, completions, tool inputs/outputs, agent reasoning, and inter-agent messages. Each step's output chains into the next step's context for coherence.

Quality measures:
- Token counts calibrated to actual content length
- Realistic timestamps (seconds-scale LLM latencies)
- Unique queries per run (LLM-expanded from seed set)
- Deviation-aware content (deviant steps reflect failures in reasoning)
- Parallel aggregator coherence (aggregator sees all workers' outputs)

## Use cases

- **Agent observability** — build and test dashboards for multi-agent workflow monitoring
- **Process mining** — apply OCEL 2.0 conformance checking algorithms
- **Anomaly detection** — train classifiers on conformant vs deviant agent behavior
- **Agent evaluation** — benchmark reasoning quality across domains
- **Trace analysis research** — study information flow in multi-agent architectures

## Development

```bash
uv sync --extra dev
pytest                # Run tests
mypy src              # Type checking
ruff check src tests  # Linting
```

## License

MIT
