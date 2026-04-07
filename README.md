# Open Agent Traces

**17,000+ realistic multi-agent workflow events across 10 enterprise domains.** Ready to use from [Hugging Face](https://huggingface.co/datasets/juliensimon/open-agent-traces) or generate your own.

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-open--agent--traces-yellow)](https://huggingface.co/datasets/juliensimon/open-agent-traces)
[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)
[![CI](https://github.com/juliensimon/ocel-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/ocel-generator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![OCEL 2.0](https://img.shields.io/badge/OCEL-2.0-orange.svg)](https://www.ocel-standard.org/)

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

Every trace includes LLM prompts and completions, tool call inputs and outputs, agent reasoning chains, inter-agent messages, calibrated token counts, realistic timestamps, and cost estimates — the same data you'd see in production agent observability tools.

## Use it now

**From Hugging Face (no install needed):**

```python
from datasets import load_dataset

ds = load_dataset("juliensimon/open-agent-traces", "incident-response")

for event in ds["train"]:
    if event["run_id"] == "run-0000":
        print(f"{event['event_type']:25s} | {event['agent_role']:12s} | {event['reasoning'][:60] if event['reasoning'] else ''}")
```

**Or generate your own (no API key needed for structural traces):**

```bash
pip install open-agent-traces

ocelgen generate --pattern sequential --runs 50 --noise 0.2 --seed 42
```

This produces a complete [OCEL 2.0](https://www.ocel-standard.org/) event log in under 2 seconds — 1,500+ events with realistic structure, deviation labels, and ground-truth manifests.

## What's inside

**500 workflow runs** across **10 domains** and **3 workflow patterns**, with **20% labeled anomalies** for conformance checking:

| Domain | Pattern | Events | What it simulates |
|--------|---------|--------|-------------------|
| `customer-support-triage` | sequential | 1,483 | Classify ticket, research KB, draft response |
| `code-review-pipeline` | supervisor | 2,035 | Delegate to linter, security reviewer, style checker |
| `incident-response` | supervisor | 1,976 | Route to diagnostics, mitigation, communications |
| `data-pipeline-debugging` | supervisor | 2,033 | Log analyzer, schema checker, fix proposer |
| `market-research` | parallel | 1,671 | Competitor analyst, trend researcher, report writer |
| `content-generation` | parallel | 1,668 | Researcher, writer, editor working concurrently |
| `academic-paper-review` | parallel | 1,695 | Methodology, novelty, writing reviewers |
| `legal-document-analysis` | sequential | 1,498 | Extract clauses, check compliance, summarize risks |
| `financial-analysis` | sequential | 1,471 | Gather filings, compute ratios, write investment memo |
| `ecommerce-product-enrichment` | sequential | 1,489 | Scrape specs, normalize attributes, generate descriptions |

**Workflow patterns:**
- **Sequential** — linear chain (A &rarr; B &rarr; C)
- **Supervisor** — central agent delegates to specialist workers
- **Parallel** — fan-out to concurrent agents, then aggregate

**10 deviation types** with ground-truth labels: skipped steps, wrong tools, swapped order, timeouts, missing handoffs, extra LLM calls, wrong routing, repeated activities, inserted activities, wrong resources.

## Enrich with any LLM

Add realistic prompts, completions, and tool I/O using any OpenAI-compatible endpoint:

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

Enrichment chains context across agent steps, detects deviations and reflects them in the generated content, recalculates token counts to match actual output, and rewrites timestamps with realistic LLM latencies.

## Validated, not just generated

Every trace is checked by **5 validation layers** — tested across all 10 domains on the [live HF dataset](https://huggingface.co/datasets/juliensimon/open-agent-traces):

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

Compatible with [PM4Py](https://pm4py.fit.fraunhofer.de/) — the reference OCEL 2.0 process mining library:

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

## Who is this for?

- **Agent observability teams** — build and test monitoring dashboards with the same data LangSmith, Arize, and Braintrust display
- **ML researchers** — train anomaly detectors on labeled conformant vs deviant traces
- **Process mining researchers** — apply OCEL 2.0 conformance checking algorithms to multi-agent systems
- **Agent framework developers** — test LangGraph, CrewAI, AutoGen, Smolagents against realistic traces
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
- **[Dataset on Hugging Face](https://huggingface.co/datasets/juliensimon/open-agent-traces)** — 17,000+ events across 10 domains

## Development

```bash
git clone https://github.com/juliensimon/ocel-generator.git && cd ocel-generator
uv sync --extra dev
uv run pre-commit install   # ruff + mypy + pytest on every commit
uv run pytest               # 265 tests, 98% coverage
```

## License

MIT
