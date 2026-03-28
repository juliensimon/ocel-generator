# Enriched Agent Traces — Design Spec

**Date:** 2026-03-28
**Status:** Approved
**Goal:** Extend ocelgen to produce LLM-enriched synthetic agent trace datasets and upload them as a Hugging Face collection of 10 datasets.

## Context

Clem Delangue called for more open agent traces datasets. ocelgen already generates structurally valid OCEL 2.0 traces for multi-agent workflows, but they lack semantic content (prompts, completions, tool I/O are empty or faker-generated). This spec adds an enrichment layer that uses a cheap LLM to fill in realistic content, plus a pipeline to upload the results to Hugging Face Hub.

## Approach: Two-Pass Generation

**Pass 1 (existing):** Generate structural OCEL traces using the current engine — fast, free, deterministic, OCEL-compliant, with deviation injection.

**Pass 2 (new):** Walk the generated trace and call an LLM via OpenRouter to fill in content attributes — prompts, completions, tool args/results, reasoning, handoff messages. One LLM call per agent step, chaining outputs to maintain coherence.

This preserves the existing engine entirely while adding enrichment as a separate, retryable layer.

## The 10 Datasets

| # | Domain | Pattern | Runs | Noise | Description |
|---|--------|---------|------|-------|-------------|
| 1 | customer-support-triage | sequential | 50 | 0.20 | Classify ticket, research KB, draft response |
| 2 | code-review-pipeline | supervisor | 50 | 0.20 | Supervisor assigns to linter, security reviewer, style checker |
| 3 | market-research | parallel | 50 | 0.20 | Planner fans out to competitor analyst, trend researcher, report writer |
| 4 | legal-document-analysis | sequential | 50 | 0.15 | Extract clauses, check compliance, summarize risks |
| 5 | data-pipeline-debugging | supervisor | 50 | 0.25 | Supervisor routes to log analyzer, schema checker, fix proposer |
| 6 | content-generation | parallel | 50 | 0.20 | Planner fans out to researcher, writer, editor |
| 7 | financial-analysis | sequential | 50 | 0.20 | Gather filings, compute ratios, write investment memo |
| 8 | incident-response | supervisor | 50 | 0.30 | On-call supervisor routes to diagnostics, mitigation, comms |
| 9 | academic-paper-review | parallel | 50 | 0.15 | Fan-out to methodology reviewer, novelty assessor, writing critic |
| 10 | ecommerce-product-enrichment | sequential | 50 | 0.20 | Scrape specs, normalize attributes, generate descriptions |

500 total runs. ~4,500 LLM enrichment calls. Estimated cost: $2-5 with a cheap model.

## Enrichment Architecture

### New attributes added to existing OCEL object types

| Object Type | New Attributes |
|---|---|
| `llm_call` | `prompt`, `completion` |
| `tool_call` | `tool_input`, `tool_output` |
| `message` | `content` |
| `agent_invocation` | `reasoning` |
| `run` | `user_query` (replaced from faker to domain-realistic) |

### Enrichment flow per run

1. Select a `user_query` from the domain's seed query bank (cycling across runs)
2. Walk the run's events in sequence order
3. For each agent step (agent_invoked → agent_completed span):
   - Build a meta-prompt with: domain context, agent persona, tools, previous step output, structural metadata (how many LLM/tool calls to generate)
   - Call the LLM once, requesting structured JSON response
   - Patch the step's `llm_call`, `tool_call`, `message`, and `agent_invocation` objects with the returned content
4. Accumulate each step's output as context for the next step

### Meta-prompt structure

```
You are simulating a {role} agent in a {domain} workflow.
Pattern: {pattern_description}
User query: {user_query}
Your tools: {tool_list_with_descriptions}
Previous agent output: {previous_step_output}
Number of LLM calls to simulate: {n}
Number of tool calls to simulate: {n}

Generate realistic content as JSON:
{
  "reasoning": "...",
  "llm_calls": [{"prompt": "...", "completion": "..."}],
  "tool_calls": [{"input": {...}, "output": {...}}],
  "output_to_next_agent": "..."
}
```

### LLM client

- OpenRouter via OpenAI-compatible API (`OPENAI_API_KEY` env var, base URL `https://openrouter.ai/api/v1`)
- Default model: `google/gemini-2.0-flash-001` (cheap, fast). Configurable via `--model` flag.
- Retry: 3 attempts with exponential backoff on transient failures

## Domain Scenarios

### Data structure

```python
@dataclass
class DomainScenario:
    name: str                    # e.g. "customer-support-triage"
    description: str             # One-line for the LLM system prompt
    pattern: str                 # "sequential", "supervisor", "parallel"
    runs: int                    # 50
    noise: float                 # 0.15-0.3
    seed: int                    # Fixed for reproducibility
    user_queries: list[str]      # 20-30 seed queries per domain
    agent_personas: dict[AgentRole, str]  # Role -> persona description
    tool_descriptions: dict[ToolKind, str]  # Tool -> contextual description
```

### Query strategy

20-30 hand-written seed queries per domain. Queries cycle across runs (run N uses query N % len(queries)). The LLM naturally introduces variation through its completions even when the query repeats.

### Tool descriptions

Same `ToolKind` enum, described differently per domain:

- Customer Support: `web_search` = "Search knowledge base for return/refund policy"
- Financial Analysis: `web_search` = "Search SEC EDGAR for company filings"
- Incident Response: `web_search` = "Search runbooks and past incident postmortems"

All 10 domains defined in a single `registry.py` file.

## HF Upload & Dataset Format

### Tabular schema (one row per event)

| Column | Type | Source |
|---|---|---|
| `event_id` | string | OCEL event id |
| `event_type` | string | e.g. `agent_invoked`, `llm_request_sent` |
| `timestamp` | string (ISO 8601) | OCEL event time |
| `run_id` | string | From event attributes |
| `sequence_number` | int | From event attributes |
| `is_deviation` | bool | From event attributes |
| `deviation_type` | string | From event attributes |
| `agent_role` | string | Resolved from related agent object |
| `model_name` | string | Resolved from related agent object |
| `step_id` | string | From event attributes (if present) |
| `prompt` | string | From related llm_call object |
| `completion` | string | From related llm_call object |
| `tool_name` | string | From related tool_call object |
| `tool_input` | string | JSON string |
| `tool_output` | string | JSON string |
| `message_content` | string | From related message object |
| `reasoning` | string | From related agent_invocation object |
| `input_tokens` | int | From related llm_call / agent_invocation |
| `output_tokens` | int | From related llm_call / agent_invocation |
| `latency_ms` | int | From related llm_call / tool_call |
| `cost_usd` | float | From related agent_invocation |
| `is_conformant` | bool | From run object |
| `pattern` | string | From run object |
| `domain` | string | Added during upload |
| `user_query` | string | From run object |

### Naming

- **Repos:** `{namespace}/agent-traces-{domain-slug}`
- **Collection:** `{namespace}/open-agent-traces`
- **Each repo contains:** Parquet file (tabular), `.jsonocel` (native OCEL), `normative_model.json`, `manifest.json`, auto-generated README

### Upload mechanism

Uses `huggingface_hub` library: `create_repo`, `upload_file`, `add_collection_item`. Raw OCEL files uploaded as supplementary artifacts. Upload is idempotent — re-running overwrites.

## CLI Commands

### `ocelgen enrich`

```bash
ocelgen enrich output.jsonocel \
    --domain customer-support-triage \
    --model google/gemini-2.0-flash-001 \
    --output enriched.jsonocel
```

### `ocelgen upload`

```bash
ocelgen upload enriched.jsonocel \
    --domain customer-support-triage \
    --repo-namespace juliensimon \
    --collection open-agent-traces
```

### `ocelgen pipeline`

```bash
# Single domain
ocelgen pipeline --domain customer-support-triage --repo-namespace juliensimon

# All 10 domains
ocelgen pipeline --all --repo-namespace juliensimon
```

### `ocelgen list-domains`

Lists all 10 domain scenarios with pattern, run count, noise level.

### Pipeline orchestration

For each domain: generate → enrich → write OCEL → flatten to Parquet → upload to HF → add to collection. Progress via `rich` progress bars.

## Error handling

- Enrichment retries failed LLM calls 3 times with exponential backoff
- If a run's enrichment fails entirely, it's kept as structural-only (content attributes left empty) — the dataset is still valid
- Upload is idempotent

## New file structure

```
src/ocelgen/
    enrichment/
        __init__.py
        client.py        # OpenRouter LLM client (OpenAI-compatible)
        enricher.py      # Walk trace, call LLM, patch OCEL attributes
        prompts.py       # Meta-prompt templates
    scenarios/
        __init__.py
        domain.py        # DomainScenario dataclass
        registry.py      # All 10 domain definitions
    upload/
        __init__.py
        flatten.py       # OcelLog -> flat rows (list of dicts)
        hf_upload.py     # Create repo, upload parquet + OCEL, manage collection
        readme.py        # Generate dataset card markdown
```

## Modified files

- `cli.py` — 4 new commands (`enrich`, `upload`, `pipeline`, `list-domains`)
- `pyproject.toml` — 3 new dependencies (`openai>=1.0`, `huggingface_hub>=0.20`, `pyarrow>=15.0`)
- `models/ocel.py` — New attribute definitions for enriched content fields

## Unchanged

All existing generation, deviation, pattern, and export code remains untouched.

## Dependencies

```toml
dependencies = [
    "pydantic>=2.6",
    "typer>=0.12",
    "rich>=13.0",
    "jsonschema>=4.21",
    "faker>=24.0",
    "openai>=1.0",
    "huggingface_hub>=0.20",
    "pyarrow>=15.0",
]
```
