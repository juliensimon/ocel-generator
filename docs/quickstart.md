# Quick Start

Generate your first synthetic agent traces dataset in under 5 minutes.

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key (for LLM enrichment — optional for structural generation)

## Installation

```bash
pip install open-agent-traces
```

For development (with linting, testing, and pre-commit hooks):

```bash
git clone https://github.com/juliensimon/ocel-generator.git
cd ocel-generator
uv sync --extra dev
uv run pre-commit install
```

## Step 1: Generate structural traces

Generate 20 sequential workflow runs with 20% noise (deviations):

```bash
ocelgen generate --pattern sequential --runs 20 --noise 0.2 --seed 42
```

This creates three files:
- `output.jsonocel` — the OCEL 2.0 event log
- `normative_model.json` — the expected workflow template
- `manifest.json` — generation metadata and injected deviations

## Step 2: Enrich with LLM content

Set your OpenRouter API key:

```bash
export OPENAI_API_KEY="sk-or-v1-your-key-here"
```

Enrich the traces with realistic prompts, completions, and tool I/O:

```bash
ocelgen enrich output.jsonocel --domain customer-support-triage
```

This produces `enriched-output.jsonocel` with LLM-generated content for each agent step.

## Step 3: Explore the data

```python
import json

with open("enriched-output.jsonocel") as f:
    log = json.load(f)

# See what's inside
print(f"Events: {len(log['events'])}")
print(f"Objects: {len(log['objects'])}")

# Look at an enriched LLM call
for obj in log["objects"]:
    if obj["type"] == "llm_call":
        attrs = {a["name"]: a["value"] for a in obj["attributes"]}
        if attrs.get("prompt"):
            print(f"\nPrompt: {attrs['prompt'][:200]}")
            print(f"Completion: {attrs['completion'][:200]}")
            break
```

## Step 4: Upload to Hugging Face (optional)

```bash
ocelgen pipeline --domain customer-support-triage --namespace your-hf-username
```

This runs the full pipeline (generate + enrich + flatten + upload) and creates a dataset on HF Hub.

## Step 5: Use custom domains (optional)

Define your own domains in a YAML file:

```yaml
# my-domains.yaml
domains:
  - name: "hr-onboarding"
    description: "HR onboarding: collect docs, run checks, provision access"
    pattern: "sequential"
    runs: 30
    noise: 0.15
    seed: 50001
    user_queries:
      - "New hire John Smith starting March 15 as Senior Engineer"
      - "Onboard contractor Maria Garcia for 6-month engagement"
    agent_personas:
      researcher: "You are an HR coordinator collecting new hire documentation"
      analyst: "You are a compliance officer verifying background checks"
      summarizer: "You are an IT provisioner setting up accounts and access"
    tool_descriptions:
      web_search: "Search HR knowledge base for onboarding checklists"
      file_reader: "Read employee records and compliance documents"
```

Then pass it with `--config`:

```bash
ocelgen enrich output.jsonocel --domain hr-onboarding --config my-domains.yaml
ocelgen pipeline --domain hr-onboarding --config my-domains.yaml --namespace your-hf-username
```

Custom domains merge with the 10 built-ins. To override a built-in, use the same `name`.

## Step 6: Validate the traces

ocelgen includes semantic validators that go beyond JSON schema:

```python
from ocelgen.generation.engine import generate
from ocelgen.validation import (
    validate_referential_integrity,
    validate_temporal_order,
    validate_type_attributes,
    validate_workflow_conformance,
)

result = generate("sequential", num_runs=50, noise_rate=0.3, seed=42)

# All references resolve, types are declared, conformant runs match the template
assert validate_referential_integrity(result.log) == []
assert validate_type_attributes(result.log) == []
assert validate_workflow_conformance(result.log, result.template) == []
```

Or validate a file from the CLI:

```bash
ocelgen validate output.jsonocel
```

## Step 7: Load with pm4py (optional)

Install the conformance extra and load traces in the reference OCEL 2.0 library:

```bash
uv sync --extra conformance
```

```python
import pm4py

ocel = pm4py.read.read_ocel2_json("output.jsonocel")
print(f"Events: {len(ocel.events)}")
print(f"Objects: {len(ocel.objects)}")

# Event types are in 'ocel:activity' (not 'ocel:type')
print(ocel.events["ocel:activity"].value_counts())
```

## Examples

The [`examples/`](../examples/) folder contains runnable scripts showing common workflows:

```bash
# Run all semantic validators across all patterns
python examples/validate_traces.py

# Walk a single run's event timeline, LLM calls, tools, costs
python examples/inspect_run.py

# Download from HF and explore with pm4py
python examples/explore_with_pm4py.py
```

## Next steps

- Read the [User Guide](user-guide.md) for detailed configuration options
- Try different [workflow patterns](user-guide.md#workflow-patterns): `sequential`, `supervisor`, `parallel`
- Explore all 10 [built-in domains](user-guide.md#domains) or [define your own](user-guide.md#custom-domains)
- Use the [pre-built dataset](https://huggingface.co/datasets/juliensimon/open-agent-traces) directly
