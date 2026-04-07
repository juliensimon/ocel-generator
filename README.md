# ocelgen

**Generate realistic multi-agent workflow traces on demand.** Any domain, any pattern, any LLM. Validated against OCEL 2.0 and PM4Py.

[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)
[![CI](https://github.com/juliensimon/ocel-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/ocel-generator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![OCEL 2.0](https://img.shields.io/badge/OCEL-2.0-orange.svg)](https://www.ocel-standard.org/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-open--agent--traces-yellow)](https://huggingface.co/datasets/juliensimon/open-agent-traces)

```bash
pip install open-agent-traces

ocelgen generate --pattern sequential --runs 50 --noise 0.2 --seed 42
```

**1,500+ events in under 2 seconds.** No API key needed for structural traces.

```
run-0000: "My order arrived damaged, what are my options?"
├── run_started                                              08:00:00.007
├── agent_invoked          researcher    gpt-4o              08:00:00.052
│   ├── llm_request_sent   "Search for refund policy..."     08:00:00.067
│   ├── llm_response       "The refund policy states..."     08:00:00.749
│   ├── tool_called        web_search    → policy found      08:00:01.705
│   └── tool_called        file_reader   → order history     08:00:01.898
├── agent_invoked          analyst       gpt-4o              08:00:02.281
│   ├── llm_request_sent   "Analyze refund eligibility..."   08:00:02.334
│   ├── llm_response       "Customer is eligible for..."     08:00:06.747
│   └── tool_called        calculator    → refund amount     08:00:08.819
├── agent_invoked          summarizer    claude-3.5-sonnet   08:00:09.680
│   ├── llm_request_sent   "Draft resolution response..."    08:00:09.717
│   └── llm_response       "Dear customer, we apologize..."  08:00:10.363
└── run_completed                                            08:00:10.369
    cost: $0.038 | 3,950 input + 2,516 output tokens | 5 LLM calls | 3 tool calls
```

## What it generates

Each trace includes LLM prompts and completions, tool call inputs and outputs, agent reasoning chains, inter-agent messages, calibrated token counts, realistic timestamps, and cost estimates — the same data you'd see in LangSmith, Arize, or Braintrust.

**3 workflow patterns:**

```
Sequential:    Research → Analyze → Summarize
Supervisor:    Supervisor → [Worker A, Worker B, Worker C] → Aggregate
Parallel:      Split → [Worker A ‖ Worker B ‖ Worker C] → Aggregate
```

**10 deviation types** with ground-truth labels for anomaly detection: skipped steps, wrong tools, swapped order, timeouts, missing handoffs, extra LLM calls, wrong routing, repeated activities, inserted activities, wrong resources.

**10 built-in enterprise domains** — or [define your own](#define-your-own-domains) in YAML:

| Domain | Pattern | What it simulates |
|--------|---------|-------------------|
| `customer-support-triage` | sequential | Classify ticket, research KB, draft response |
| `code-review-pipeline` | supervisor | Delegate to linter, security reviewer, style checker |
| `incident-response` | supervisor | Route to diagnostics, mitigation, communications |
| `data-pipeline-debugging` | supervisor | Log analyzer, schema checker, fix proposer |
| `market-research` | parallel | Competitor analyst, trend researcher, report writer |
| `content-generation` | parallel | Researcher, writer, editor working concurrently |
| `academic-paper-review` | parallel | Methodology, novelty, writing reviewers |
| `legal-document-analysis` | sequential | Extract clauses, check compliance, summarize risks |
| `financial-analysis` | sequential | Gather filings, compute ratios, write investment memo |
| `ecommerce-product-enrichment` | sequential | Scrape specs, normalize attributes, generate descriptions |

## Enrich with any LLM

Plug in any OpenAI-compatible endpoint to fill traces with realistic content:

```bash
# Cloud (OpenRouter — default)
export OPENAI_API_KEY="your-key"
ocelgen enrich output.jsonocel --domain customer-support-triage

# Local (llama.cpp, Ollama, vLLM — no API key needed)
ocelgen enrich output.jsonocel -d customer-support-triage \
  --model local-model --base-url http://localhost:8080/v1

# Full pipeline: generate + enrich + upload to Hugging Face
ocelgen pipeline --domain customer-support-triage --namespace your-hf-username
```

Enrichment chains context across agent steps, reflects deviations in the generated content, recalibrates token counts and timestamps, and expands seed queries via LLM for diversity across runs.

## Validated, not just generated

Every trace is checked by 5 validation layers — tested across all 10 domains, all 3 patterns, and [the live HF dataset](https://huggingface.co/datasets/juliensimon/open-agent-traces):

| Validator | What it checks |
|-----------|---------------|
| JSON Schema | OCEL 2.0 structural compliance |
| Referential integrity | Every relationship points to an existing object |
| Type attributes | Every attribute matches its declared type schema |
| Temporal ordering | Causal pairs in order, run boundaries correct |
| Workflow conformance | Conformant runs follow the template (parallel-aware) |

```python
from ocelgen.generation.engine import generate
from ocelgen.validation import (
    validate_referential_integrity,
    validate_workflow_conformance,
)

result = generate("sequential", num_runs=50, noise_rate=0.3, seed=42)
assert validate_referential_integrity(result.log) == []
assert validate_workflow_conformance(result.log, result.template) == []
```

Traces load directly in [PM4Py](https://pm4py.fit.fraunhofer.de/) — the reference OCEL 2.0 process mining library:

```bash
pip install open-agent-traces[conformance]
```

```python
import pm4py
ocel = pm4py.read.read_ocel2_json("output.jsonocel")
```

## Define your own domains

Create custom domains in YAML — they merge with the 10 built-ins:

```yaml
domains:
  - name: "hr-onboarding"
    description: "HR onboarding: collect docs, run checks, provision access"
    pattern: "sequential"
    runs: 30
    noise: 0.15
    seed: 50001
    user_queries:
      - "New hire starting March 15 as Senior Engineer"
    agent_personas:
      researcher: "You are an HR coordinator collecting new hire documentation"
      analyst: "You are a compliance officer verifying background checks"
      summarizer: "You are an IT provisioner setting up accounts and access"
    tool_descriptions:
      web_search: "Search HR knowledge base for onboarding checklists"
      file_reader: "Read employee records and compliance documents"
```

```bash
ocelgen pipeline --domain hr-onboarding --config domains.yaml --namespace your-hf-username
```

## Pre-built dataset

Don't want to generate? Load 17,000+ events directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("juliensimon/open-agent-traces", "incident-response")

for event in ds["train"]:
    if event["run_id"] == "run-0000":
        print(f"{event['event_type']:25s} | {event['agent_role']:12s} | {event['reasoning'][:60] if event['reasoning'] else ''}")
```

## Who is this for?

- **Agent observability teams** — build and test monitoring dashboards with realistic trace data
- **ML researchers** — train anomaly detectors on labeled conformant vs deviant traces
- **Process mining researchers** — apply OCEL 2.0 conformance checking to multi-agent systems
- **Agent framework developers** — test LangGraph, CrewAI, AutoGen, Smolagents pipelines
- **Evaluation teams** — benchmark agent reasoning quality across domains and architectures

## Examples

| Script | What it shows |
|--------|---------------|
| [`basic_generation.py`](examples/basic_generation.py) | Generate logs via Python API, inspect results, write files |
| [`validate_traces.py`](examples/validate_traces.py) | Run all 5 semantic validators across all 3 patterns |
| [`inspect_run.py`](examples/inspect_run.py) | Walk a single run's event timeline, LLM calls, tools, costs |
| [`explore_with_pm4py.py`](examples/explore_with_pm4py.py) | Download from HF, query with pm4py and datasets library |
| [`conformance_demo.py`](examples/conformance_demo.py) | Generate and load with pm4py |

## Documentation

- **[Quick Start](docs/quickstart.md)** — first dataset in 5 minutes
- **[User Guide](docs/user-guide.md)** — CLI reference, patterns, domains, custom YAML, validation, PM4Py
- **[Dataset on HF](https://huggingface.co/datasets/juliensimon/open-agent-traces)** — 17,000+ events across 10 domains

## Development

```bash
git clone https://github.com/juliensimon/ocel-generator.git && cd ocel-generator
uv sync --extra dev
uv run pre-commit install   # ruff + mypy + pytest on every commit
uv run pytest               # 265 tests, 98% coverage
```

## License

MIT
