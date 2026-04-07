# ocelgen — Open Agent Traces Dataset Generator

Generate realistic multi-agent workflow trace datasets with LLM-enriched content, semantic validation, and PM4Py compatibility. Built for the AI agent ecosystem.

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-open--agent--traces-yellow)](https://huggingface.co/datasets/juliensimon/open-agent-traces)
[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)
[![CI](https://github.com/juliensimon/ocel-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/ocel-generator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![OCEL 2.0](https://img.shields.io/badge/OCEL-2.0-orange.svg)](https://www.ocel-standard.org/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-lightgrey.svg)](docs/user-guide.md#enrichment-details)

![Parallel workflow trace — market research domain](docs/parallel-workflow-example.png)

## The problem

Real agent traces are scarce. Production multi-agent systems generate rich execution data — LLM prompts, tool calls, agent reasoning, handoff messages — but these traces are proprietary and rarely shared. Teams building agent observability, evaluation, and debugging tools lack open datasets to develop against.

## The solution

ocelgen generates **structurally valid, semantically rich** agent traces that look and feel like real multi-agent executions:

- **Full trace content** — LLM prompts and completions, tool call inputs/outputs, agent reasoning, inter-agent messages
- **10 enterprise domains** — customer support, code review, incident response, financial analysis, and 6 more (plus custom domains via YAML)
- **3 workflow patterns** — sequential, supervisor/worker, parallel fan-out/fan-in
- **Labeled deviations** — 10 types of anomalies (wrong tools, skipped steps, timeouts) with ground-truth annotations
- **Semantic validation** — referential integrity, temporal ordering, type attribute checks, and workflow conformance beyond JSON schema
- **OCEL 2.0 standard** — validated with [PM4Py](https://pm4py.fit.fraunhofer.de/) across all 10 domains
- **Any LLM backend** — OpenRouter, OpenAI, Anthropic, local models via `--base-url`

## Quick start

```bash
pip install open-agent-traces
```

### LLM setup

Enrichment requires an OpenAI-compatible endpoint. Pick one:

**Cloud (OpenRouter, OpenAI, etc.)**
```bash
export OPENAI_API_KEY="your-key"
# Default: OpenRouter with Gemini Flash. Override with --model:
ocelgen enrich output.jsonocel -d customer-support-triage --model anthropic/claude-sonnet-4
```

**Local (llama.cpp, Ollama, vLLM, etc.)**
```bash
# Point ocelgen at the local endpoint (no API key needed)
ocelgen enrich output.jsonocel -d customer-support-triage \
  --model local-model --base-url http://localhost:8080/v1
```

### Generate and enrich

```bash
# Generate structural traces
ocelgen generate --pattern sequential --runs 50 --noise 0.2

# Enrich with LLM-generated content
ocelgen enrich output.jsonocel --domain customer-support-triage

# Or run the full pipeline (generate + enrich + upload to HF)
ocelgen pipeline --domain customer-support-triage --namespace your-hf-username

# Use custom domains defined in YAML
ocelgen pipeline --domain my-domain --config domains.yaml --namespace your-hf-username
```

### Development setup

```bash
git clone https://github.com/juliensimon/ocel-generator.git && cd ocel-generator
uv sync --extra dev
uv run pre-commit install
```

## Use the pre-built dataset

Skip generation — load the dataset directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("juliensimon/open-agent-traces", "incident-response")

for event in ds["train"]:
    if event["run_id"] == "run-0000":
        print(f"{event['event_type']:25s} | {event['agent_role']:12s} | {event['reasoning'][:60] if event['reasoning'] else ''}")
```

10 domains available: `customer-support-triage` · `code-review-pipeline` · `market-research` · `legal-document-analysis` · `data-pipeline-debugging` · `content-generation` · `financial-analysis` · `incident-response` · `academic-paper-review` · `ecommerce-product-enrichment`

## Validate traces

ocelgen includes semantic validators that go beyond JSON schema — referential integrity, temporal ordering, type attribute declarations, and workflow conformance:

```python
from ocelgen.generation.engine import generate
from ocelgen.validation import (
    validate_referential_integrity,
    validate_temporal_order,
    validate_type_attributes,
    validate_workflow_conformance,
)

result = generate("sequential", num_runs=50, noise_rate=0.3, seed=42)

assert validate_referential_integrity(result.log) == []
assert validate_type_attributes(result.log) == []
assert validate_workflow_conformance(result.log, result.template) == []
```

With the optional `pm4py` extra, you can also load traces directly in the reference OCEL 2.0 process mining library:

```bash
pip install open-agent-traces[conformance]
```

```python
import pm4py
ocel = pm4py.read.read_ocel2_json("output.jsonocel")
```

## Who is this for?

- **Agent observability teams** — build dashboards with realistic trace data (timestamps, token counts, costs)
- **ML researchers** — train anomaly detectors on labeled conformant vs deviant traces
- **Process mining researchers** — apply OCEL 2.0 conformance checking to agent workflows
- **Agent framework developers** — test LangGraph, CrewAI, AutoGen, Smolagents against realistic traces
- **Evaluation teams** — benchmark agent reasoning quality across domains and architectures

## Examples

The [`examples/`](examples/) folder contains runnable scripts:

| Script | What it shows |
|--------|---------------|
| [`basic_generation.py`](examples/basic_generation.py) | Generate logs via Python API, inspect results, write files |
| [`validate_traces.py`](examples/validate_traces.py) | Run all 5 semantic validators across all 3 patterns |
| [`inspect_run.py`](examples/inspect_run.py) | Walk a single run's event timeline, LLM calls, tools, costs |
| [`explore_with_pm4py.py`](examples/explore_with_pm4py.py) | Download from HF, query with pm4py and datasets library |
| [`conformance_demo.py`](examples/conformance_demo.py) | Generate and load with pm4py |

## Documentation

- **[Quick Start](docs/quickstart.md)** — first dataset in 5 minutes
- **[User Guide](docs/user-guide.md)** — CLI reference, patterns, domains, custom YAML config, validation, PM4Py
- **[Dataset on Hugging Face](https://huggingface.co/datasets/juliensimon/open-agent-traces)** — 17,000+ events across 10 domains, ready to use

## License

MIT
