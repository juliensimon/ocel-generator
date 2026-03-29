# Quick Start

Generate your first synthetic agent traces dataset in under 5 minutes.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [OpenRouter](https://openrouter.ai) API key (for LLM enrichment)

## Installation

```bash
git clone https://github.com/juliensimon/ocel-generator.git
cd ocel-generator
uv sync
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

## Next steps

- Read the [User Guide](user-guide.md) for detailed configuration options
- Try different [workflow patterns](user-guide.md#workflow-patterns): `sequential`, `supervisor`, `parallel`
- Explore all 10 [built-in domains](user-guide.md#domains)
- Use the [pre-built dataset](https://huggingface.co/datasets/juliensimon/open-agent-traces) directly
