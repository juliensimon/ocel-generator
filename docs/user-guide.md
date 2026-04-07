# User Guide

Complete reference for ocelgen — generating, enriching, and publishing synthetic agent trace datasets.

## CLI Reference

### `ocelgen generate`

Generate structural OCEL 2.0 event logs (no LLM calls needed).

```bash
ocelgen generate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --pattern` | `sequential` | Workflow pattern: `sequential`, `supervisor`, `parallel` |
| `-n, --runs` | `100` | Number of workflow runs |
| `-N, --noise` | `0.2` | Fraction of runs with deviations (0.0–1.0) |
| `--max-deviations` | `3` | Max deviations per deviant run |
| `--seed` | random | Random seed for reproducibility |
| `-o, --output` | `output.jsonocel` | Output file path |

**Examples:**

```bash
# 100 sequential runs, 20% noise
ocelgen generate -p sequential -n 100 -N 0.2

# 50 supervisor runs, no noise, reproducible
ocelgen generate -p supervisor -n 50 -N 0.0 --seed 42

# 200 parallel runs, heavy noise
ocelgen generate -p parallel -n 200 -N 0.5 -o parallel_traces.jsonocel
```

### `ocelgen enrich`

Enrich a structural trace with LLM-generated content via OpenRouter.

Requires `OPENAI_API_KEY` environment variable set to your OpenRouter key.

```bash
ocelgen enrich <path> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --domain` | required | Domain scenario name |
| `-m, --model` | `google/gemini-2.0-flash-001` | LLM model for enrichment |
| `--base-url` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL |
| `-o, --output` | `enriched-<input>` | Output file path |
| `-c, --config` | — | YAML file or directory with custom domain definitions |

**Examples:**

```bash
# Enrich with default model (OpenRouter)
ocelgen enrich output.jsonocel --domain customer-support-triage

# Use a different model
ocelgen enrich output.jsonocel --domain incident-response --model openai/gpt-4o-mini

# Use a local LLM (no API key needed)
ocelgen enrich output.jsonocel --domain customer-support-triage \
  --model local-model --base-url http://localhost:8080/v1

# Custom output path
ocelgen enrich output.jsonocel --domain code-review-pipeline -o enriched.jsonocel

# Use a custom domain from YAML
ocelgen enrich output.jsonocel --domain my-domain --config domains.yaml
```

### `ocelgen pipeline`

End-to-end: generate, enrich, and optionally upload to Hugging Face Hub.

```bash
ocelgen pipeline [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --domain` | — | Single domain to process |
| `--all` | `false` | Process all domains |
| `-n, --namespace` | required | HF namespace for upload |
| `-m, --model` | `google/gemini-2.0-flash-001` | LLM model |
| `--base-url` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL |
| `--collection` | `open-agent-traces` | HF collection slug for grouping datasets |
| `--skip-upload` | `false` | Generate and enrich without uploading |
| `-c, --config` | — | YAML file or directory with custom domain definitions |

**Examples:**

```bash
# Single domain, upload to HF
ocelgen pipeline --domain customer-support-triage --namespace juliensimon

# All domains (built-in + custom)
ocelgen pipeline --all --namespace juliensimon --config my-domains/

# Generate and enrich without uploading
ocelgen pipeline --domain incident-response --namespace test --skip-upload

# Custom domain from YAML
ocelgen pipeline --domain my-domain --config domains.yaml --namespace test --skip-upload

# Use a local LLM (no API key needed)
ocelgen pipeline --domain customer-support-triage --namespace test --skip-upload \
  --model local-model --base-url http://localhost:8080/v1
```

### `ocelgen validate`

Validate an OCEL 2.0 file against the official schema.

```bash
ocelgen validate path/to/file.jsonocel
```

### `ocelgen list-patterns`

Show available workflow patterns with step counts.

### `ocelgen upload`

Upload an enriched trace to Hugging Face Hub.

```bash
ocelgen upload <path> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --domain` | required | Domain scenario name |
| `-n, --namespace` | required | HF namespace |
| `--collection` | `open-agent-traces` | HF collection slug for grouping datasets |
| `-c, --config` | — | YAML file or directory with custom domain definitions |

### `ocelgen list-domains`

Show available domain scenarios with pattern, run count, and noise level.

```bash
ocelgen list-domains [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config` | — | YAML file or directory with custom domain definitions |

## Workflow Patterns

### Sequential

A linear chain of agents, where each completes before the next begins.

```
Research → Analyze → Summarize
```

3 agents, each with 1–2 LLM calls and 0–3 tool calls per step.

**Domains using this pattern:** customer-support-triage, legal-document-analysis, financial-analysis, ecommerce-product-enrichment

### Supervisor

A central supervisor agent delegates tasks to specialist workers, then aggregates results.

```
Supervisor → [Worker A, Worker B, Worker C] → Supervisor (aggregate)
```

5 agents. The supervisor makes routing decisions, each worker has tools, and the aggregator combines results.

**Domains:** code-review-pipeline, data-pipeline-debugging, incident-response

### Parallel

A planner fans out work to concurrent agents, then an aggregator merges results.

```
Planner → [Worker A || Worker B || Worker C] → Aggregator
```

5 agents with overlapping timestamps for parallel workers.

**Domains:** market-research, content-generation, academic-paper-review

## Domains

| Domain | Pattern | Description |
|--------|---------|-------------|
| `customer-support-triage` | sequential | Classify ticket, research KB, draft response |
| `code-review-pipeline` | supervisor | Delegate to linter, security reviewer, style checker |
| `market-research` | parallel | Fan-out to competitor analyst, trend researcher, report writer |
| `legal-document-analysis` | sequential | Extract clauses, check compliance, summarize risks |
| `data-pipeline-debugging` | supervisor | Route to log analyzer, schema checker, fix proposer |
| `content-generation` | parallel | Fan-out to researcher, writer, editor |
| `financial-analysis` | sequential | Gather filings, compute ratios, write investment memo |
| `incident-response` | supervisor | Route to diagnostics, mitigation, communications |
| `academic-paper-review` | parallel | Fan-out to methodology, novelty, writing reviewers |
| `ecommerce-product-enrichment` | sequential | Scrape specs, normalize attributes, generate descriptions |

## Custom Domains

You can define your own domain scenarios in YAML files and use them alongside (or instead of) the 10 built-in domains.

### YAML schema

```yaml
domains:
  - name: "my-domain"            # unique identifier (required)
    description: "What this domain simulates"  # (required)
    pattern: "sequential"        # sequential | supervisor | parallel (required)
    runs: 50                     # number of workflow runs (required)
    noise: 0.20                  # fraction of runs with deviations, 0.0–1.0 (required)
    seed: 42                     # random seed for reproducibility (required)
    user_queries:                # seed queries recycled across runs (optional)
      - "First example query"
      - "Second example query"
    agent_personas:              # role → persona description (optional)
      researcher: "You are a researcher investigating the topic"
      analyst: "You are an analyst evaluating findings"
      summarizer: "You are a writer drafting the final report"
    tool_descriptions:           # tool_name → description (optional)
      web_search: "Search for relevant information"
      calculator: "Perform calculations"
```

A single YAML file can contain multiple domains under the `domains` key.

### Usage

Pass a YAML file or directory with `--config / -c`:

```bash
# List built-in + custom domains
ocelgen list-domains --config my-domains.yaml

# Enrich with a custom domain
ocelgen enrich output.jsonocel --domain my-domain --config my-domains.yaml

# Full pipeline with custom domain
ocelgen pipeline --domain my-domain --config my-domains.yaml --namespace test --skip-upload

# Load all YAML files from a directory
ocelgen pipeline --all --config ./domains/ --namespace test --skip-upload
```

### Merging behavior

Custom domains merge with the 10 built-in domains:
- **New names** are added to the registry
- **Matching names** override the built-in (e.g., defining `customer-support-triage` in YAML replaces the built-in version)
- When loading a directory, files are processed alphabetically; later files override earlier ones for the same domain name

## Deviation Types

Deviations are injected into a configurable fraction of runs to create non-conformant traces. Each deviation is labeled in the event attributes (`is_deviation=true`, `deviation_type=<type>`), providing ground truth for conformance checking evaluation.

| Deviation | What it does |
|-----------|-------------|
| `skipped_activity` | Removes all events for a non-start/non-end step |
| `inserted_activity` | Adds an unexpected agent invocation |
| `wrong_resource` | Swaps the agent handling a step to the wrong one |
| `swapped_order` | Swaps timestamps of two consecutive agent invocations |
| `wrong_tool` | Replaces a tool call with a different tool |
| `repeated_activity` | Duplicates an agent invocation (simulates retry) |
| `timeout` | Adds a timeout error event after an agent invocation |
| `wrong_routing` | Inserts a routing decision that selects the wrong agent |
| `missing_handoff` | Removes an inter-agent message event |
| `extra_llm_call` | Inserts an unnecessary extra LLM call |

## Enrichment Details

The enrichment pass calls an LLM once per agent step via any OpenAI-compatible endpoint (OpenRouter by default, or a local server via `--base-url`). It:

1. **Expands seed queries** — if the domain has fewer seed queries than runs, the LLM generates additional unique queries
2. **Chains context** — each step's output is passed as context to the next step, producing coherent traces
3. **Detects deviations** — deviant steps get modified prompts that tell the LLM to generate failure-reflecting content
4. **Recalculates metrics** — token counts, latencies, and costs are calibrated to the actual enriched content
5. **Rewrites timestamps** — event timestamps are adjusted to reflect realistic LLM latencies (1–5s per call)

### Model and endpoint configuration

ocelgen works with any OpenAI-compatible API. Use `--model` and `--base-url` to configure:

**Cloud providers (via OpenRouter — default):**
```bash
export OPENAI_API_KEY="your-openrouter-key"
ocelgen enrich output.jsonocel -d customer-support-triage
ocelgen enrich output.jsonocel -d customer-support-triage --model openai/gpt-4o
```

**Direct provider APIs:**
```bash
export OPENAI_API_KEY="your-openai-key"
ocelgen enrich output.jsonocel -d customer-support-triage \
  --model gpt-4o --base-url https://api.openai.com/v1
```

**Local models (llama.cpp, Ollama, vLLM — no API key needed):**
```bash
ocelgen enrich output.jsonocel -d customer-support-triage \
  --model local-model --base-url http://localhost:8080/v1
```

### Supported models

Any model available on OpenRouter works. Recommended:

| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| `google/gemini-2.0-flash-001` | Fast | Low | Good (default) |
| `openai/gpt-4o-mini` | Fast | Low | Good |
| `anthropic/claude-haiku` | Fast | Low | Good |
| `openai/gpt-4o` | Slower | Higher | Excellent |

### Cost estimate

With the default model (Gemini Flash), enriching 50 runs costs approximately:
- Sequential pattern (3 steps/run): ~$0.15–0.30
- Supervisor pattern (5 steps/run): ~$0.25–0.50
- Parallel pattern (5 steps/run): ~$0.25–0.50

All 10 domains (500 runs total): ~$2–5

## Output Format

### OCEL 2.0 JSON (`.jsonocel`)

The native format follows the [OCEL 2.0 JSON specification](https://www.ocel-standard.org/) with:

- **Object types:** `run`, `agent`, `agent_invocation`, `llm_call`, `tool_call`, `message`, `task`
- **Event types:** `run_started`, `agent_invoked`, `llm_request_sent`, `llm_response_received`, `tool_called`, `tool_returned`, `message_sent`, `routing_decided`, `agent_completed`, `run_completed`, `error_occurred`, `retry_started`
- **Relationships:** Events linked to objects via `e2o` relationships with qualifiers

### Parquet (tabular)

The flattened format has one row per event with denormalized columns for agent role, LLM content, tool I/O, and run metadata. Used for HF Hub datasets.

## Using the Pre-Built Dataset

Skip generation entirely and use the pre-built dataset:

```python
from datasets import load_dataset

# Load a specific domain
ds = load_dataset("juliensimon/open-agent-traces", "incident-response")

# All available configs
configs = [
    "customer-support-triage", "code-review-pipeline", "market-research",
    "legal-document-analysis", "data-pipeline-debugging", "content-generation",
    "financial-analysis", "incident-response", "academic-paper-review",
    "ecommerce-product-enrichment",
]
```

See the [dataset page](https://huggingface.co/datasets/juliensimon/open-agent-traces) for full documentation and examples.

## Validation

### CLI validation

```bash
ocelgen validate output.jsonocel
```

This checks the file against the OCEL 2.0 JSON schema. For deeper validation, use the Python API.

### Semantic validators

ocelgen includes four semantic validation layers beyond JSON schema:

```python
from ocelgen.generation.engine import generate
from ocelgen.validation import (
    validate_ocel_dict,
    validate_referential_integrity,
    validate_temporal_order,
    validate_type_attributes,
    validate_workflow_conformance,
)
from ocelgen.export.ocel_json import ocel_log_to_dict

result = generate("sequential", num_runs=50, noise_rate=0.3, seed=42)
log = result.log

# 1. JSON Schema — structural compliance
errors = validate_ocel_dict(ocel_log_to_dict(log))

# 2. Referential integrity — every relationship points to an existing object
errors = validate_referential_integrity(log)

# 3. Type attributes — every attribute is declared in its type schema
errors = validate_type_attributes(log)

# 4. Temporal ordering — events respect causal order within each run
errors = validate_temporal_order(log)

# 5. Workflow conformance — conformant runs follow the normative template
errors = validate_workflow_conformance(log, result.template)
```

Each validator returns a list of error messages (empty if valid).

| Validator | What it checks |
|-----------|---------------|
| `validate_ocel_dict` | OCEL 2.0 JSON schema compliance (Draft 7) |
| `validate_referential_integrity` | Dangling references, duplicate IDs, undeclared types |
| `validate_type_attributes` | Attributes match their eventType/objectType declarations |
| `validate_temporal_order` | run_started first, run_completed last, causal pairs in order |
| `validate_workflow_conformance` | Conformant runs match template steps (parallel-group aware) |

### PM4Py compatibility

Install the conformance extra to load traces in [pm4py](https://pm4py.fit.fraunhofer.de/) — the reference OCEL 2.0 process mining library:

```bash
pip install open-agent-traces[conformance]
```

```python
import pm4py

ocel = pm4py.read.read_ocel2_json("output.jsonocel")
print(f"Events: {len(ocel.events)}, Objects: {len(ocel.objects)}")

# Note: pm4py uses 'ocel:activity' for event types (not 'ocel:type')
print(ocel.events["ocel:activity"].value_counts())

# Object types use 'ocel:type'
print(ocel.objects["ocel:type"].value_counts())

# Relationships are in ocel.relations
print(f"Relationships: {len(ocel.relations)}")
```

## Examples

Runnable scripts in the [`examples/`](../examples/) folder:

| Script | Description |
|--------|-------------|
| [`basic_generation.py`](../examples/basic_generation.py) | Generate logs via Python API, inspect results, write files |
| [`validate_traces.py`](../examples/validate_traces.py) | Run all 5 semantic validators across all 3 patterns |
| [`inspect_run.py`](../examples/inspect_run.py) | Walk a single run's event timeline, LLM calls, tools, costs, deviations |
| [`explore_with_pm4py.py`](../examples/explore_with_pm4py.py) | Download from HF, query with pm4py and datasets library |
| [`conformance_demo.py`](../examples/conformance_demo.py) | Generate and load with pm4py |

```bash
# Run an example
python examples/validate_traces.py
```
