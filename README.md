# ocelgen — Synthetic Agent Traces Dataset Generator

Generate realistic, LLM-enriched multi-agent workflow trace datasets in [OCEL 2.0](https://www.ocel-standard.org/) format. Built for the AI agent ecosystem: observability tooling, process mining research, anomaly detection, and agent evaluation.

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-open--agent--traces-yellow)](https://huggingface.co/datasets/juliensimon/open-agent-traces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Pre-built dataset:** [`juliensimon/open-agent-traces`](https://huggingface.co/datasets/juliensimon/open-agent-traces) — 17,000+ events, 10 domains, 3 workflow patterns

## Example: Market Research Parallel Workflow

![Parallel workflow trace](docs/parallel-workflow-example.png)

A single parallel-pattern run showing the full agent execution trace with **real enriched content**: the planner fans out to three concurrent workers (researcher, analyst, writer), each making LLM calls and tool invocations with domain-specific inputs/outputs, then the aggregator synthesizes all results into a final report.

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

## Use cases for AI agent development

### Agent observability and debugging
Build and test monitoring dashboards that visualize multi-agent workflow execution. The traces include realistic timestamps, token counts, and cost estimates — exactly what tools like LangSmith, Arize, Braintrust, and Weights & Biases need to display.

### Agent evaluation and benchmarking
Compare agent reasoning quality across domains and patterns. The dataset covers sequential chains (LangChain-style), supervisor/worker delegation (CrewAI-style), and parallel fan-out (LangGraph-style) — the three most common agentic architectures.

### Conformance checking and anomaly detection
Train classifiers to distinguish conformant from deviant agent behavior. Each dataset includes labeled deviations (wrong tools, skipped steps, timeouts) with ground-truth annotations — ready for supervised ML.

### Process mining on agent workflows
Apply OCEL 2.0 process mining algorithms to multi-agent systems. The traces use the official Object-Centric Event Log standard with proper object types (agents, LLM calls, tool calls, messages) and relationships.

### Agent framework testing
Test agent orchestration frameworks (LangGraph, CrewAI, AutoGen, Smolagents) against realistic trace data. The 10 domains cover common enterprise use cases: customer support, code review, incident response, data pipeline debugging, and more.

## Model and endpoint configuration

ocelgen uses any **OpenAI-compatible API endpoint** for enrichment. Set `OPENAI_API_KEY` and optionally override the base URL:

```bash
# OpenRouter (default)
export OPENAI_API_KEY="sk-or-v1-..."
ocelgen enrich output.jsonocel --domain incident-response --model google/gemini-2.0-flash-001

# OpenAI directly
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
ocelgen enrich output.jsonocel --domain incident-response --model gpt-4o-mini

# Local models (Ollama, vLLM, etc.)
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:11434/v1"
ocelgen enrich output.jsonocel --domain incident-response --model llama3
```

**Model recommendations:**

| Model | Speed | Cost/500 runs | Best for |
|-------|-------|---------------|----------|
| `google/gemini-2.0-flash-001` | Fast | ~$2 | Default — good balance |
| `openai/gpt-4o-mini` | Fast | ~$3 | High quality at low cost |
| `anthropic/claude-haiku` | Fast | ~$2 | Concise, structured output |
| `openai/gpt-4o` | Slower | ~$15 | Maximum content quality |
| Local (Llama 3, Mistral) | Varies | Free | Privacy, offline use |

## Documentation

- [Quick Start](docs/quickstart.md) — generate your first dataset in 5 minutes
- [User Guide](docs/user-guide.md) — CLI reference, patterns, domains, enrichment details
- [Dataset on Hugging Face](https://huggingface.co/datasets/juliensimon/open-agent-traces) — pre-built dataset, ready to use

## Development

```bash
uv sync --extra dev
pytest                # Run tests
mypy src              # Type checking
ruff check src tests  # Linting
```

## License

MIT
