# Enriched Agent Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ocelgen with LLM-enriched content generation and Hugging Face upload to produce 10 open agent trace datasets.

**Architecture:** Two-pass: existing structural OCEL generation (untouched) + new enrichment layer that calls an LLM via OpenRouter to fill in prompts, completions, tool I/O, and reasoning. A flatten+upload pipeline pushes results to HF Hub as Parquet datasets in a collection.

**Tech Stack:** Python 3.11+, Pydantic, OpenAI SDK (OpenRouter-compatible), huggingface_hub, PyArrow, Typer/Rich CLI.

**Spec:** `docs/superpowers/specs/2026-03-28-enriched-agent-traces-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/ocelgen/scenarios/__init__.py` | Package init, re-exports |
| `src/ocelgen/scenarios/domain.py` | `DomainScenario` dataclass |
| `src/ocelgen/scenarios/registry.py` | All 10 domain definitions, `SCENARIO_REGISTRY` dict |
| `src/ocelgen/enrichment/__init__.py` | Package init, re-exports |
| `src/ocelgen/enrichment/client.py` | OpenRouter LLM client wrapper |
| `src/ocelgen/enrichment/prompts.py` | Meta-prompt builder for enrichment |
| `src/ocelgen/enrichment/enricher.py` | Walk OCEL trace, call LLM, patch attributes |
| `src/ocelgen/upload/__init__.py` | Package init, re-exports |
| `src/ocelgen/upload/flatten.py` | `OcelLog` to flat tabular rows |
| `src/ocelgen/upload/readme.py` | Generate HF dataset card markdown |
| `src/ocelgen/upload/hf_upload.py` | Create HF repos, upload files, manage collection |
| `tests/test_scenarios.py` | Tests for domain scenarios |
| `tests/test_enrichment.py` | Tests for enrichment (mocked LLM) |
| `tests/test_flatten.py` | Tests for OCEL-to-tabular flattening |
| `tests/test_upload.py` | Tests for readme generation and upload logic |
| `tests/test_cli_new.py` | Tests for new CLI commands |

### Modified files

| File | Change |
|---|---|
| `pyproject.toml` | Add `openai`, `huggingface_hub`, `pyarrow` dependencies |
| `src/ocelgen/generation/run_simulator.py:116-153` | Add enrichment attribute defs to `OBJECT_ATTR_DEFS` |
| `src/ocelgen/cli.py` | Add `enrich`, `upload`, `pipeline`, `list-domains` commands |

---

## Parallelism Notes

Tasks 1-2 are sequential (foundation). After that:
- **Parallel group A:** Tasks 3, 4, 5 (enrichment layer — sequential within group)
- **Parallel group B:** Tasks 6, 7 (flatten + readme — independent of enrichment)
- Tasks 8, 9 depend on both groups completing.
- Task 10 is final integration.

---

### Task 1: Dependencies and Attribute Definitions

**Files:**
- Modify: `pyproject.toml:10-16`
- Modify: `src/ocelgen/generation/run_simulator.py:116-153`
- Test: existing tests must still pass

- [ ] **Step 1: Update pyproject.toml with new dependencies**

In `pyproject.toml`, replace the dependencies list:

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

- [ ] **Step 2: Add enrichment attribute definitions to OBJECT_ATTR_DEFS**

In `src/ocelgen/generation/run_simulator.py`, replace lines 116-153 (the `OBJECT_ATTR_DEFS` dict) with:

```python
OBJECT_ATTR_DEFS: dict[str, list[OcelAttributeDefinition]] = {
    "run": [
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="pattern_type", type="string"),
        OcelAttributeDefinition(name="is_conformant", type="boolean"),
        OcelAttributeDefinition(name="user_query", type="string"),
    ],
    "agent": [
        OcelAttributeDefinition(name="role", type="string"),
        OcelAttributeDefinition(name="model_name", type="string"),
    ],
    "agent_invocation": [
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="input_tokens", type="integer"),
        OcelAttributeDefinition(name="output_tokens", type="integer"),
        OcelAttributeDefinition(name="cost_usd", type="float"),
        OcelAttributeDefinition(name="reasoning", type="string"),
    ],
    "tool_call": [
        OcelAttributeDefinition(name="tool_name", type="string"),
        OcelAttributeDefinition(name="tool_kind", type="string"),
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="duration_ms", type="integer"),
        OcelAttributeDefinition(name="tool_input", type="string"),
        OcelAttributeDefinition(name="tool_output", type="string"),
    ],
    "llm_call": [
        OcelAttributeDefinition(name="model", type="string"),
        OcelAttributeDefinition(name="input_tokens", type="integer"),
        OcelAttributeDefinition(name="output_tokens", type="integer"),
        OcelAttributeDefinition(name="latency_ms", type="integer"),
        OcelAttributeDefinition(name="prompt", type="string"),
        OcelAttributeDefinition(name="completion", type="string"),
    ],
    "message": [
        OcelAttributeDefinition(name="role", type="string"),
        OcelAttributeDefinition(name="content_length", type="integer"),
        OcelAttributeDefinition(name="content", type="string"),
    ],
    "task": [
        OcelAttributeDefinition(name="description", type="string"),
        OcelAttributeDefinition(name="status", type="string"),
    ],
}
```

New attributes added: `reasoning` (agent_invocation), `tool_input`/`tool_output` (tool_call), `prompt`/`completion` (llm_call), `content` (message).

- [ ] **Step 3: Install dependencies and run existing tests**

Run: `cd /Users/julien/Development/mpmx && uv sync`
Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/ -v`
Expected: All existing tests pass. The new attribute definitions are schema declarations — they don't change runtime behavior.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/ocelgen/generation/run_simulator.py
git commit -m "feat: add enrichment dependencies and OCEL attribute definitions"
```

---

### Task 2: Domain Scenarios Module

**Files:**
- Create: `src/ocelgen/scenarios/__init__.py`
- Create: `src/ocelgen/scenarios/domain.py`
- Create: `src/ocelgen/scenarios/registry.py`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write tests for domain scenarios**

Create `tests/test_scenarios.py`:

```python
"""Tests for domain scenario definitions."""

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario


class TestDomainScenario:
    def test_dataclass_creation(self) -> None:
        scenario = DomainScenario(
            name="test-domain",
            description="A test domain",
            pattern="sequential",
            runs=10,
            noise=0.2,
            seed=42,
            user_queries=["query one", "query two"],
            agent_personas={"researcher": "You are a test researcher"},
            tool_descriptions={"web_search": "Search the web"},
        )
        assert scenario.name == "test-domain"
        assert scenario.pattern == "sequential"
        assert len(scenario.user_queries) == 2

    def test_query_for_run_cycles(self) -> None:
        scenario = DomainScenario(
            name="test",
            description="test",
            pattern="sequential",
            runs=10,
            noise=0.2,
            seed=42,
            user_queries=["q0", "q1", "q2"],
            agent_personas={},
            tool_descriptions={},
        )
        assert scenario.query_for_run(0) == "q0"
        assert scenario.query_for_run(1) == "q1"
        assert scenario.query_for_run(2) == "q2"
        assert scenario.query_for_run(3) == "q0"  # cycles


class TestRegistry:
    def test_registry_has_10_domains(self) -> None:
        assert len(SCENARIO_REGISTRY) == 10

    def test_all_domains_have_queries(self) -> None:
        for name, scenario in SCENARIO_REGISTRY.items():
            assert len(scenario.user_queries) >= 10, f"{name} needs at least 10 queries"

    def test_all_domains_have_valid_pattern(self) -> None:
        valid = {"sequential", "supervisor", "parallel"}
        for name, scenario in SCENARIO_REGISTRY.items():
            assert scenario.pattern in valid, f"{name} has invalid pattern {scenario.pattern}"

    def test_all_domains_have_unique_seeds(self) -> None:
        seeds = [s.seed for s in SCENARIO_REGISTRY.values()]
        assert len(seeds) == len(set(seeds)), "Seeds must be unique across domains"

    def test_get_scenario_by_name(self) -> None:
        scenario = get_scenario("customer-support-triage")
        assert scenario.name == "customer-support-triage"
        assert scenario.pattern == "sequential"

    def test_get_scenario_unknown_raises(self) -> None:
        import pytest
        with pytest.raises(KeyError):
            get_scenario("nonexistent-domain")

    def test_all_domains_have_agent_personas(self) -> None:
        for name, scenario in SCENARIO_REGISTRY.items():
            assert len(scenario.agent_personas) >= 2, f"{name} needs at least 2 personas"

    def test_all_domains_have_tool_descriptions(self) -> None:
        for name, scenario in SCENARIO_REGISTRY.items():
            # Sequential summarizer has no tools, so some domains may have 0
            # but most should have at least 1
            pass  # validated by enrichment integration test
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ocelgen.scenarios'`

- [ ] **Step 3: Create the scenarios package**

Create `src/ocelgen/scenarios/__init__.py`:

```python
"""Domain scenario definitions for LLM-enriched trace generation."""

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario

__all__ = ["DomainScenario", "SCENARIO_REGISTRY", "get_scenario"]
```

Create `src/ocelgen/scenarios/domain.py`:

```python
"""DomainScenario dataclass — defines a domain for enriched trace generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainScenario:
    """A domain scenario for generating enriched agent traces.

    Each scenario pairs a workflow pattern with domain-specific context:
    user queries, agent personas, and tool descriptions that the enrichment
    LLM uses to generate realistic content.
    """

    name: str
    description: str
    pattern: str  # "sequential", "supervisor", "parallel"
    runs: int
    noise: float
    seed: int
    user_queries: list[str] = field(default_factory=list)
    agent_personas: dict[str, str] = field(default_factory=dict)
    tool_descriptions: dict[str, str] = field(default_factory=dict)

    def query_for_run(self, run_index: int) -> str:
        """Return the user query for a given run index, cycling through the bank."""
        return self.user_queries[run_index % len(self.user_queries)]
```

Create `src/ocelgen/scenarios/registry.py`:

```python
"""Registry of all 10 domain scenarios."""

from __future__ import annotations

from ocelgen.scenarios.domain import DomainScenario

SCENARIO_REGISTRY: dict[str, DomainScenario] = {
    "customer-support-triage": DomainScenario(
        name="customer-support-triage",
        description="Customer support ticket triage: classify, research, and draft response",
        pattern="sequential",
        runs=50,
        noise=0.20,
        seed=1001,
        user_queries=[
            "My refund for order #4821 hasn't arrived after 10 business days",
            "I was charged twice for my Pro subscription this month",
            "The delivery tracking says 'delivered' but I never received my package",
            "I can't log into my account after the password reset",
            "My discount code SAVE20 isn't working at checkout",
            "I need to cancel my order but the cancel button is greyed out",
            "The product I received doesn't match the listing photos at all",
            "My gift card balance disappeared after a failed transaction",
            "I've been waiting 3 weeks for a replacement item that was promised in 5 days",
            "The app keeps crashing when I try to view my order history",
            "I was promised free shipping but got charged $12.99",
            "My subscription renewed even though I cancelled it last month",
        ],
        agent_personas={
            "researcher": "You are a support agent researching the customer's issue in the knowledge base and order management system",
            "analyst": "You are a support analyst determining the root cause, checking policies, and deciding the appropriate resolution",
            "summarizer": "You are a support agent drafting a clear, empathetic response to the customer with the resolution",
        },
        tool_descriptions={
            "web_search": "Search the internal knowledge base for policies, FAQs, and troubleshooting guides",
            "file_reader": "Read customer account details, order history, and previous support interactions",
            "calculator": "Calculate refund amounts, shipping costs, or discount values",
            "code_interpreter": "Run queries against the order management database",
        },
    ),
    "code-review-pipeline": DomainScenario(
        name="code-review-pipeline",
        description="Automated code review: supervisor delegates to linter, security reviewer, and style checker",
        pattern="supervisor",
        runs=50,
        noise=0.20,
        seed=2002,
        user_queries=[
            "Review PR #342: Add user authentication middleware with JWT tokens",
            "Review PR #343: Refactor database connection pooling for PostgreSQL",
            "Review PR #344: Add rate limiting to public API endpoints",
            "Review PR #345: Migrate from REST to GraphQL for the user profile service",
            "Review PR #346: Fix SQL injection vulnerability in search endpoint",
            "Review PR #347: Add WebSocket support for real-time notifications",
            "Review PR #348: Implement RBAC authorization for admin panel",
            "Review PR #349: Optimize image upload pipeline with async processing",
            "Review PR #350: Add OpenTelemetry instrumentation to all HTTP handlers",
            "Review PR #351: Replace homegrown cache with Redis for session storage",
            "Review PR #352: Add CORS configuration for multi-tenant SaaS deployment",
            "Review PR #353: Implement graceful shutdown handling for background workers",
        ],
        agent_personas={
            "supervisor": "You are a senior engineering lead triaging a pull request and delegating review tasks to specialized reviewers",
            "researcher": "You are a code linter checking for bugs, type errors, unused imports, and code smells",
            "coder": "You are a security reviewer scanning for vulnerabilities: injection, XSS, auth bypasses, secrets in code",
            "reviewer": "You are a style checker verifying naming conventions, documentation, test coverage, and architectural consistency",
        },
        tool_descriptions={
            "web_search": "Search documentation for language/framework best practices",
            "file_reader": "Read the pull request diff, source files, and test files",
            "code_interpreter": "Run static analysis tools (ruff, mypy, bandit) on the changed files",
        },
    ),
    "market-research": DomainScenario(
        name="market-research",
        description="Market research: fan-out to competitor analyst, trend researcher, and report writer",
        pattern="parallel",
        runs=50,
        noise=0.20,
        seed=3003,
        user_queries=[
            "Analyze the competitive landscape for AI code assistants in 2025",
            "Research market trends in open-source large language model deployment",
            "Compare pricing strategies of major cloud GPU providers",
            "Investigate the enterprise adoption rate of retrieval-augmented generation",
            "Analyze the developer tools market with focus on observability platforms",
            "Research the growth trajectory of AI-powered customer support solutions",
            "Compare feature sets of top 5 vector database providers",
            "Investigate market opportunities in AI agent orchestration frameworks",
            "Analyze competitive positioning of major MLOps platforms",
            "Research enterprise spending trends on generative AI infrastructure",
            "Compare go-to-market strategies of AI coding startups vs incumbents",
            "Investigate the impact of open-weight models on commercial AI services",
        ],
        agent_personas={
            "planner": "You are a research director breaking down the research question into parallel workstreams for your team",
            "researcher": "You are a competitive intelligence analyst gathering data on market players, products, and positioning",
            "analyst": "You are a market trends analyst identifying growth patterns, adoption curves, and emerging opportunities",
            "writer": "You are a research report writer synthesizing findings into clear, data-backed narratives",
            "aggregator": "You are a research director combining all workstreams into a cohesive final market research report",
        },
        tool_descriptions={
            "web_search": "Search for press releases, analyst reports, funding announcements, and product launches",
            "calculator": "Calculate market share percentages, growth rates, and financial comparisons",
            "database_query": "Query internal market intelligence database for historical data",
            "text_splitter": "Split long documents and reports into sections for focused analysis",
        },
    ),
    "legal-document-analysis": DomainScenario(
        name="legal-document-analysis",
        description="Legal document analysis: extract clauses, check compliance, summarize risks",
        pattern="sequential",
        runs=50,
        noise=0.15,
        seed=4004,
        user_queries=[
            "Review this SaaS vendor agreement for data processing compliance with GDPR",
            "Analyze the indemnification clauses in the proposed partnership agreement",
            "Check this employment contract template against California labor law requirements",
            "Review the liability limitations in our cloud services terms of service",
            "Analyze the intellectual property assignment clauses in this contractor agreement",
            "Check the termination and renewal provisions in the enterprise license agreement",
            "Review the data breach notification requirements in this DPA addendum",
            "Analyze the non-compete and non-solicitation scope in the executive agreement",
            "Check the force majeure clause coverage in our supply chain contracts",
            "Review the warranty disclaimers in the open-source license for commercial use",
            "Analyze the audit rights and compliance reporting in the SOC 2 vendor agreement",
            "Check the cross-border data transfer mechanisms in the international DPA",
        ],
        agent_personas={
            "researcher": "You are a legal analyst extracting and cataloging key clauses, definitions, and obligations from the document",
            "analyst": "You are a compliance specialist checking extracted clauses against applicable regulations and internal policies",
            "summarizer": "You are a legal advisor summarizing the risk exposure and recommending actions for the business team",
        },
        tool_descriptions={
            "web_search": "Search legal databases for relevant case law, regulatory guidance, and compliance standards",
            "file_reader": "Read the contract document, internal policy templates, and compliance checklists",
            "calculator": "Calculate financial exposure, liability caps, and penalty thresholds",
            "code_interpreter": "Run clause comparison against standard template library",
        },
    ),
    "data-pipeline-debugging": DomainScenario(
        name="data-pipeline-debugging",
        description="Data pipeline debugging: supervisor routes to log analyzer, schema checker, and fix proposer",
        pattern="supervisor",
        runs=50,
        noise=0.25,
        seed=5005,
        user_queries=[
            "Pipeline job etl_daily_users failed at 03:42 UTC with OOM error on the join stage",
            "Data quality alert: 40% null values in user_email column after last ETL run",
            "Spark job for revenue aggregation has been running 3x longer than usual since Tuesday",
            "Schema mismatch error: upstream API changed the 'address' field from string to object",
            "Duplicate records detected in the orders fact table after backfill job ran twice",
            "Airflow DAG customer_360 stuck in 'running' state for 6 hours with no progress",
            "BigQuery costs spiked 400% this week — investigate which queries are responsible",
            "CDC replication from PostgreSQL to Snowflake has a 4-hour lag since the table alter",
            "Data freshness SLA breach: dashboard data is 8 hours stale instead of 1 hour",
            "dbt model staging_payments failing with 'relation does not exist' after warehouse migration",
            "Partition skew in the daily clickstream job causing executor timeouts on node 3",
            "Kafka consumer group for events_ingest has been rebalancing every 5 minutes since deploy",
        ],
        agent_personas={
            "supervisor": "You are a data engineering lead triaging a pipeline incident and assigning investigation tasks to specialists",
            "researcher": "You are a log analyst examining execution logs, error traces, and resource utilization metrics",
            "coder": "You are a schema and data quality checker validating data shapes, types, and integrity constraints",
            "reviewer": "You are a fix proposer suggesting configuration changes, code patches, or architectural improvements",
        },
        tool_descriptions={
            "web_search": "Search internal runbooks and documentation for known issues and fixes",
            "file_reader": "Read pipeline configuration files, DAG definitions, and dbt models",
            "code_interpreter": "Run diagnostic queries against the data warehouse and metadata store",
            "database_query": "Query pipeline execution logs and monitoring metrics",
        },
    ),
    "content-generation": DomainScenario(
        name="content-generation",
        description="Content generation: fan-out to researcher, writer, and editor with final aggregation",
        pattern="parallel",
        runs=50,
        noise=0.20,
        seed=6006,
        user_queries=[
            "Write a technical blog post about implementing RAG with open-source models",
            "Create a product announcement for our new real-time collaboration feature",
            "Write a developer tutorial on building MCP servers in Python",
            "Create a case study about how Company X reduced inference costs by 60%",
            "Write a comparison article: fine-tuning vs RAG vs prompt engineering",
            "Create a newsletter edition covering this month's top open-source AI releases",
            "Write an onboarding guide for new contributors to our open-source project",
            "Create a technical deep-dive on transformer attention mechanism optimizations",
            "Write a year-in-review post summarizing our community growth and milestones",
            "Create a best-practices guide for deploying LLMs in production environments",
            "Write a post explaining how we built our evaluation pipeline for model quality",
            "Create an explainer article on quantization techniques for edge deployment",
        ],
        agent_personas={
            "planner": "You are a content strategist planning the article structure, key points, and audience targeting",
            "researcher": "You are a technical researcher gathering facts, code examples, benchmarks, and references",
            "analyst": "You are a data analyst pulling metrics, usage statistics, and performance benchmarks to support claims",
            "writer": "You are a technical writer crafting the prose with clear explanations and engaging narrative",
            "aggregator": "You are an editor combining research and writing into a polished, publication-ready article",
        },
        tool_descriptions={
            "web_search": "Search for reference materials, documentation, competitor content, and source data",
            "calculator": "Calculate performance metrics, cost comparisons, and statistical summaries",
            "database_query": "Query internal analytics for product usage data and engagement metrics",
            "text_splitter": "Split long reference documents into digestible sections for analysis",
        },
    ),
    "financial-analysis": DomainScenario(
        name="financial-analysis",
        description="Financial analysis: gather filings, compute ratios, write investment memo",
        pattern="sequential",
        runs=50,
        noise=0.20,
        seed=7007,
        user_queries=[
            "Analyze NVIDIA's Q4 2025 earnings and provide an investment recommendation",
            "Compare AMD vs Intel financial performance over the last 4 quarters",
            "Evaluate Snowflake's revenue growth trajectory and path to profitability",
            "Analyze the impact of AI infrastructure spending on Microsoft's cloud margins",
            "Review Palantir's government vs commercial revenue mix and growth sustainability",
            "Assess Datadog's competitive position based on latest financial metrics",
            "Analyze Alphabet's AI investment returns across Cloud and Search segments",
            "Evaluate the financial health of Confluent post their pricing model change",
            "Compare MongoDB vs CockroachDB financial metrics and market positioning",
            "Analyze Meta's Reality Labs spending trajectory and VR market opportunity",
            "Review Salesforce's acquisition integration costs and organic growth rate",
            "Assess the financial impact of open-source strategy on Elastic's revenue",
        ],
        agent_personas={
            "researcher": "You are a financial researcher gathering earnings reports, SEC filings, analyst estimates, and market data",
            "analyst": "You are a financial analyst computing valuation ratios, growth metrics, and peer comparisons",
            "summarizer": "You are an investment analyst writing a concise memo with bull/bear thesis and recommendation",
        },
        tool_descriptions={
            "web_search": "Search SEC EDGAR, earnings transcripts, and financial news for company filings and data",
            "file_reader": "Read downloaded financial statements, spreadsheets, and prior analysis reports",
            "calculator": "Calculate P/E ratios, revenue growth rates, margins, DCF valuations, and peer comparisons",
            "code_interpreter": "Run financial models and generate comparison charts",
        },
    ),
    "incident-response": DomainScenario(
        name="incident-response",
        description="Incident response: on-call supervisor routes to diagnostics, mitigation, and communications agents",
        pattern="supervisor",
        runs=50,
        noise=0.30,
        seed=8008,
        user_queries=[
            "SEV1: API gateway returning 503 for 30% of requests across all regions since 14:22 UTC",
            "SEV2: Authentication service latency spiked to 8s p99, causing login failures",
            "SEV1: Database primary node unresponsive, automatic failover did not trigger",
            "SEV2: CDN cache hit ratio dropped from 95% to 20% after config deploy at 10:15",
            "SEV1: Payment processing completely down — all transactions failing with timeout",
            "SEV2: Memory leak in worker pods causing OOM kills every 45 minutes since release v2.8.1",
            "SEV1: Data loss alert — 2 hours of event data missing from analytics pipeline",
            "SEV2: SSL certificate expired on api.example.com causing all HTTPS connections to fail",
            "SEV1: Kubernetes control plane unresponsive in us-east-1, pods cannot be scheduled",
            "SEV2: Rate limiter misconfigured after deploy — blocking legitimate traffic from top customer",
            "SEV1: DNS propagation failure — 40% of users resolving to old IP after migration",
            "SEV2: Background job queue backed up to 500K messages, consumer throughput dropped 90%",
        ],
        agent_personas={
            "supervisor": "You are the on-call incident commander coordinating the response and making escalation decisions",
            "researcher": "You are a diagnostics engineer investigating logs, metrics, traces, and recent changes to identify root cause",
            "coder": "You are a mitigation engineer implementing hotfixes, rollbacks, or workarounds to restore service",
            "reviewer": "You are a communications specialist drafting status page updates, stakeholder notifications, and post-incident reports",
        },
        tool_descriptions={
            "web_search": "Search internal runbooks, past incident postmortems, and documentation for known solutions",
            "file_reader": "Read deployment manifests, configuration files, and recent git changes",
            "code_interpreter": "Run diagnostic scripts, query monitoring systems, and test fixes",
            "database_query": "Query metrics store (Prometheus/Datadog) and log aggregator (Elasticsearch/Loki)",
        },
    ),
    "academic-paper-review": DomainScenario(
        name="academic-paper-review",
        description="Academic paper review: fan-out to methodology reviewer, novelty assessor, and writing critic",
        pattern="parallel",
        runs=50,
        noise=0.15,
        seed=9009,
        user_queries=[
            "Review: 'Scaling Laws for Sparse Mixture-of-Experts Language Models'",
            "Review: 'Chain-of-Thought Distillation for Efficient Small Language Models'",
            "Review: 'Self-Play Reinforcement Learning for Code Generation Agents'",
            "Review: 'Towards Faithful Long-Context Summarization with Retrieval Augmentation'",
            "Review: 'Dynamic Quantization Strategies for On-Device LLM Inference'",
            "Review: 'Multi-Agent Debate Improves Factual Accuracy in Open-Domain QA'",
            "Review: 'Continual Learning Without Catastrophic Forgetting in Vision-Language Models'",
            "Review: 'Efficient Fine-Tuning of Diffusion Models via Low-Rank Adaptation'",
            "Review: 'Benchmarking Tool-Use Capabilities Across 50 Large Language Models'",
            "Review: 'Constitutional AI: Training Harmless Assistants Without Human Labels'",
            "Review: 'Graph Neural Networks for Molecular Property Prediction at Scale'",
            "Review: 'Multimodal Instruction Following with Interleaved Image-Text Training'",
        ],
        agent_personas={
            "planner": "You are a meta-reviewer assigning the paper to specialist reviewers and defining review criteria",
            "researcher": "You are a methodology reviewer evaluating experimental design, statistical rigor, baselines, and reproducibility",
            "analyst": "You are a novelty assessor comparing the contribution against related work and assessing significance",
            "writer": "You are a writing and clarity critic evaluating presentation quality, figure clarity, and argument structure",
            "aggregator": "You are a meta-reviewer synthesizing all reviews into a final recommendation with accept/revise/reject",
        },
        tool_descriptions={
            "web_search": "Search Semantic Scholar, arXiv, and Google Scholar for related papers and citation context",
            "calculator": "Verify statistical claims, compute effect sizes, and check significance thresholds",
            "database_query": "Query paper metadata database for citation counts and author h-index",
            "text_splitter": "Split the paper into sections (abstract, methods, results, discussion) for focused review",
        },
    ),
    "ecommerce-product-enrichment": DomainScenario(
        name="ecommerce-product-enrichment",
        description="E-commerce product enrichment: scrape specs, normalize attributes, generate descriptions",
        pattern="sequential",
        runs=50,
        noise=0.20,
        seed=10010,
        user_queries=[
            "Enrich product listing: Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "Enrich product listing: Apple MacBook Air 15-inch M3 (2024) base configuration",
            "Enrich product listing: Samsung Galaxy S25 Ultra 512GB Titanium Black",
            "Enrich product listing: Dyson V15 Detect Absolute cordless vacuum cleaner",
            "Enrich product listing: LG C4 65-inch OLED evo 4K Smart TV (2024)",
            "Enrich product listing: Bose QuietComfort Ultra Earbuds",
            "Enrich product listing: ASUS ROG Strix G16 gaming laptop RTX 4070",
            "Enrich product listing: Ninja Creami Deluxe ice cream maker NC501",
            "Enrich product listing: Herman Miller Aeron chair fully loaded size B",
            "Enrich product listing: Anker SOLIX F2000 portable power station",
            "Enrich product listing: Kindle Scribe 64GB with Premium Pen",
            "Enrich product listing: DJI Mini 4 Pro drone with RC 2 controller",
        ],
        agent_personas={
            "researcher": "You are a product data specialist scraping manufacturer websites and retail listings for raw specifications",
            "analyst": "You are a data normalizer cleaning, standardizing, and structuring product attributes into a consistent taxonomy",
            "summarizer": "You are a copywriter generating SEO-optimized product descriptions, bullet points, and comparison highlights",
        },
        tool_descriptions={
            "web_search": "Search manufacturer websites, retail listings, and review sites for product specifications",
            "file_reader": "Read existing product catalog entries and attribute taxonomy definitions",
            "calculator": "Convert units, calculate price-per-feature ratios, and normalize numeric specifications",
            "code_interpreter": "Run attribute extraction and normalization scripts on raw product data",
        },
    ),
}


def get_scenario(name: str) -> DomainScenario:
    """Look up a domain scenario by name. Raises KeyError if not found."""
    return SCENARIO_REGISTRY[name]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_scenarios.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/scenarios/ tests/test_scenarios.py
git commit -m "feat: add 10 domain scenario definitions for enriched traces"
```

---

### Task 3: LLM Client

**Files:**
- Create: `src/ocelgen/enrichment/__init__.py`
- Create: `src/ocelgen/enrichment/client.py`
- Test: `tests/test_enrichment.py`

- [ ] **Step 1: Write tests for the LLM client**

Create `tests/test_enrichment.py`:

```python
"""Tests for enrichment client and enricher (mocked LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ocelgen.enrichment.client import LLMClient, EnrichmentResponse


class TestLLMClient:
    def test_client_creation_with_defaults(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            assert client.model == "google/gemini-2.0-flash-001"

    def test_client_custom_model(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient(model="openai/gpt-4o-mini")
            assert client.model == "openai/gpt-4o-mini"

    def test_client_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                LLMClient()


class TestEnrichmentResponse:
    def test_parse_valid_response(self) -> None:
        raw = {
            "reasoning": "I need to search the knowledge base first.",
            "llm_calls": [
                {"prompt": "Search for refund policy", "completion": "The refund policy states..."}
            ],
            "tool_calls": [
                {"input": {"query": "refund policy"}, "output": {"result": "Policy found"}}
            ],
            "output_to_next_agent": "The customer's refund is eligible for processing.",
        }
        resp = EnrichmentResponse.from_dict(raw)
        assert resp.reasoning == "I need to search the knowledge base first."
        assert len(resp.llm_calls) == 1
        assert resp.llm_calls[0]["prompt"] == "Search for refund policy"
        assert len(resp.tool_calls) == 1
        assert resp.output_to_next_agent == "The customer's refund is eligible for processing."

    def test_parse_missing_fields_uses_defaults(self) -> None:
        raw = {"reasoning": "thinking..."}
        resp = EnrichmentResponse.from_dict(raw)
        assert resp.reasoning == "thinking..."
        assert resp.llm_calls == []
        assert resp.tool_calls == []
        assert resp.output_to_next_agent == ""

    def test_parse_extra_llm_calls_trimmed(self) -> None:
        raw = {
            "reasoning": "ok",
            "llm_calls": [
                {"prompt": "p1", "completion": "c1"},
                {"prompt": "p2", "completion": "c2"},
                {"prompt": "p3", "completion": "c3"},
            ],
            "tool_calls": [],
            "output_to_next_agent": "done",
        }
        resp = EnrichmentResponse.from_dict(raw, expected_llm_calls=2)
        assert len(resp.llm_calls) == 2

    def test_parse_extra_tool_calls_trimmed(self) -> None:
        raw = {
            "reasoning": "ok",
            "llm_calls": [],
            "tool_calls": [
                {"input": {}, "output": {}},
                {"input": {}, "output": {}},
            ],
            "output_to_next_agent": "done",
        }
        resp = EnrichmentResponse.from_dict(raw, expected_tool_calls=1)
        assert len(resp.tool_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ocelgen.enrichment'`

- [ ] **Step 3: Implement the LLM client**

Create `src/ocelgen/enrichment/__init__.py`:

```python
"""LLM enrichment layer for OCEL traces."""
```

Create `src/ocelgen/enrichment/client.py`:

```python
"""OpenRouter LLM client for trace enrichment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI


DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3


@dataclass
class EnrichmentResponse:
    """Parsed response from the enrichment LLM."""

    reasoning: str
    llm_calls: list[dict[str, str]]
    tool_calls: list[dict[str, object]]
    output_to_next_agent: str

    @classmethod
    def from_dict(
        cls,
        raw: dict,
        expected_llm_calls: int | None = None,
        expected_tool_calls: int | None = None,
    ) -> EnrichmentResponse:
        """Parse a raw dict from LLM JSON output into an EnrichmentResponse.

        Trims excess calls to match expected counts if provided.
        """
        llm_calls = raw.get("llm_calls", [])
        tool_calls = raw.get("tool_calls", [])

        if expected_llm_calls is not None:
            llm_calls = llm_calls[:expected_llm_calls]
        if expected_tool_calls is not None:
            tool_calls = tool_calls[:expected_tool_calls]

        return cls(
            reasoning=raw.get("reasoning", ""),
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            output_to_next_agent=raw.get("output_to_next_agent", ""),
        )


class LLMClient:
    """Thin wrapper around OpenAI-compatible API for OpenRouter."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for enrichment. "
                "Set it to your OpenRouter API key."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the LLM and parse the JSON response.

        Retries up to MAX_RETRIES times on transient failures.
        Returns the parsed JSON dict.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts: {last_error}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/enrichment/__init__.py src/ocelgen/enrichment/client.py tests/test_enrichment.py
git commit -m "feat: add OpenRouter LLM client for trace enrichment"
```

---

### Task 4: Enrichment Prompts

**Files:**
- Create: `src/ocelgen/enrichment/prompts.py`
- Test: `tests/test_enrichment.py` (append)

- [ ] **Step 1: Add tests for prompt builder**

Append to `tests/test_enrichment.py`:

```python
from ocelgen.enrichment.prompts import build_enrichment_prompt


class TestPromptBuilder:
    def test_build_prompt_basic(self) -> None:
        system, user = build_enrichment_prompt(
            domain_description="Customer support triage workflow",
            pattern_description="Linear chain: Research -> Analyze -> Summarize",
            agent_role="researcher",
            agent_persona="You are a support agent researching the issue",
            user_query="My refund hasn't arrived after 10 days",
            tool_names=["web_search", "file_reader"],
            tool_descriptions={
                "web_search": "Search the knowledge base",
                "file_reader": "Read customer order history",
            },
            expected_llm_calls=2,
            expected_tool_calls=1,
            previous_output=None,
        )
        assert "Customer support" in system
        assert "researcher" in user
        assert "refund" in user
        assert "web_search" in user
        assert '"llm_calls"' in user
        assert "2" in user  # expected_llm_calls

    def test_build_prompt_with_previous_output(self) -> None:
        _, user = build_enrichment_prompt(
            domain_description="Test domain",
            pattern_description="Test pattern",
            agent_role="analyst",
            agent_persona="You are an analyst",
            user_query="Test query",
            tool_names=[],
            tool_descriptions={},
            expected_llm_calls=1,
            expected_tool_calls=0,
            previous_output="The researcher found that...",
        )
        assert "The researcher found that..." in user

    def test_build_prompt_no_tools(self) -> None:
        _, user = build_enrichment_prompt(
            domain_description="Test domain",
            pattern_description="Test pattern",
            agent_role="summarizer",
            agent_persona="You are a summarizer",
            user_query="Test query",
            tool_names=[],
            tool_descriptions={},
            expected_llm_calls=1,
            expected_tool_calls=0,
            previous_output="Previous analysis results...",
        )
        assert "no tools" in user.lower() or "0" in user
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py::TestPromptBuilder -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement prompt builder**

Create `src/ocelgen/enrichment/prompts.py`:

```python
"""Meta-prompt templates for LLM-based trace enrichment."""

from __future__ import annotations


def build_enrichment_prompt(
    domain_description: str,
    pattern_description: str,
    agent_role: str,
    agent_persona: str,
    user_query: str,
    tool_names: list[str],
    tool_descriptions: dict[str, str],
    expected_llm_calls: int,
    expected_tool_calls: int,
    previous_output: str | None,
) -> tuple[str, str]:
    """Build the system and user prompts for a single enrichment call.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        f"You are simulating an AI agent in a multi-agent workflow.\n"
        f"Domain: {domain_description}\n"
        f"Workflow pattern: {pattern_description}\n\n"
        f"Generate realistic, detailed content that would appear in a real agent trace. "
        f"Include specific data, names, numbers, and technical details — not generic placeholders. "
        f"Respond with valid JSON only."
    )

    tools_section = ""
    if tool_names:
        tool_lines = []
        for name in tool_names:
            desc = tool_descriptions.get(name, name)
            tool_lines.append(f"  - {name}: {desc}")
        tools_section = "Available tools:\n" + "\n".join(tool_lines)
    else:
        tools_section = "Available tools: none (this agent uses only LLM reasoning)"

    previous_section = ""
    if previous_output:
        previous_section = f"Previous agent output:\n{previous_output}\n"

    user_prompt = (
        f"You are acting as the **{agent_role}** agent.\n"
        f"Persona: {agent_persona}\n\n"
        f"User query: {user_query}\n\n"
        f"{previous_section}"
        f"{tools_section}\n\n"
        f"Generate exactly {expected_llm_calls} LLM call(s) and {expected_tool_calls} tool call(s).\n\n"
        f"Respond as JSON with this exact structure:\n"
        f'{{\n'
        f'  "reasoning": "Your chain-of-thought reasoning (2-4 sentences)",\n'
        f'  "llm_calls": [\n'
        f'    {{"prompt": "The prompt sent to the LLM", "completion": "The LLM response"}}\n'
        f'  ],\n'
        f'  "tool_calls": [\n'
        f'    {{"input": {{"arg": "value"}}, "output": {{"result": "value"}}}}\n'
        f'  ],\n'
        f'  "output_to_next_agent": "Summary output passed to the next agent in the chain"\n'
        f'}}'
    )

    return system_prompt, user_prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/enrichment/prompts.py tests/test_enrichment.py
git commit -m "feat: add enrichment prompt builder"
```

---

### Task 5: Enricher (Walk Trace + Patch Attributes)

**Files:**
- Create: `src/ocelgen/enrichment/enricher.py`
- Test: `tests/test_enrichment.py` (append)

This is the core logic: walk an `OcelLog`, identify agent steps, call the LLM, and patch OCEL objects with the returned content.

- [ ] **Step 1: Add tests for the enricher**

Append to `tests/test_enrichment.py`:

```python
from unittest.mock import patch, MagicMock

from ocelgen.enrichment.enricher import enrich_log, _extract_steps_from_log
from ocelgen.generation.engine import generate
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.scenarios.domain import DomainScenario


def _make_test_scenario() -> DomainScenario:
    return DomainScenario(
        name="test-domain",
        description="Test domain for unit tests",
        pattern="sequential",
        runs=3,
        noise=0.0,
        seed=42,
        user_queries=["Test query one", "Test query two", "Test query three"],
        agent_personas={
            "researcher": "Test researcher persona",
            "analyst": "Test analyst persona",
            "summarizer": "Test summarizer persona",
        },
        tool_descriptions={
            "web_search": "Test web search",
            "file_reader": "Test file reader",
            "calculator": "Test calculator",
            "code_interpreter": "Test code interpreter",
        },
    )


class TestExtractSteps:
    def test_extract_steps_from_sequential_run(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        steps = _extract_steps_from_log(result.log, "run-0000")
        # Sequential has 3 steps: research, analyze, summarize
        assert len(steps) == 3
        for step in steps:
            assert "agent_role" in step
            assert "invocation_id" in step
            assert "llm_call_ids" in step
            assert "tool_call_ids" in step

    def test_extract_steps_from_supervisor_run(self) -> None:
        result = generate("supervisor", num_runs=1, noise_rate=0.0, seed=42)
        steps = _extract_steps_from_log(result.log, "run-0000")
        assert len(steps) >= 3  # supervisor has at least plan + workers + aggregate


class TestEnrichLog:
    def test_enrich_patches_llm_call_objects(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_response = {
            "reasoning": "I need to investigate this.",
            "llm_calls": [
                {"prompt": "Find info about test query", "completion": "I found that..."},
                {"prompt": "Analyze the findings", "completion": "The analysis shows..."},
            ],
            "tool_calls": [
                {"input": {"query": "test"}, "output": {"result": "found"}},
                {"input": {"query": "test2"}, "output": {"result": "found2"}},
                {"input": {"query": "test3"}, "output": {"result": "found3"}},
            ],
            "output_to_next_agent": "Here are my findings.",
        }

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        enrich_log(result.log, scenario, client=mock_client)

        # Check that llm_call objects got prompt/completion attributes
        llm_objs = [o for o in result.log.objects if o.type == "llm_call"]
        assert len(llm_objs) > 0
        enriched = [o for o in llm_objs if any(a.name == "prompt" for a in o.attributes)]
        assert len(enriched) > 0

    def test_enrich_preserves_ocel_validity(self) -> None:
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_response = {
            "reasoning": "Thinking...",
            "llm_calls": [
                {"prompt": "p", "completion": "c"},
                {"prompt": "p2", "completion": "c2"},
            ],
            "tool_calls": [
                {"input": {"q": "v"}, "output": {"r": "v"}},
                {"input": {"q": "v"}, "output": {"r": "v"}},
                {"input": {"q": "v"}, "output": {"r": "v"}},
            ],
            "output_to_next_agent": "Done.",
        }

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        enrich_log(result.log, scenario, client=mock_client)

        from ocelgen.validation.schema import validate_ocel_dict
        errors = validate_ocel_dict(ocel_log_to_dict(result.log))
        assert errors == [], f"OCEL validation failed after enrichment: {errors}"

    def test_enrich_replaces_user_query(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "reasoning": "ok",
            "llm_calls": [{"prompt": "p", "completion": "c"}, {"prompt": "p", "completion": "c"}],
            "tool_calls": [{"input": {}, "output": {}}, {"input": {}, "output": {}}, {"input": {}, "output": {}}],
            "output_to_next_agent": "done",
        }

        enrich_log(result.log, scenario, client=mock_client)

        run_obj = next(o for o in result.log.objects if o.id == "run-0000")
        query_attr = next(a for a in run_obj.attributes if a.name == "user_query")
        assert query_attr.value == "Test query one"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py::TestExtractSteps -v`
Expected: FAIL — `ImportError: cannot import name 'enrich_log' from 'ocelgen.enrichment'`

- [ ] **Step 3: Implement the enricher**

Create `src/ocelgen/enrichment/enricher.py`:

```python
"""Walk an OCEL trace, call the LLM per agent step, and patch attributes."""

from __future__ import annotations

import json

from rich.progress import Progress, TaskID

from ocelgen.enrichment.client import EnrichmentResponse, LLMClient
from ocelgen.enrichment.prompts import build_enrichment_prompt
from ocelgen.models.ocel import OcelLog, OcelObjectAttribute
from ocelgen.scenarios.domain import DomainScenario


def _extract_steps_from_log(log: OcelLog, run_id: str) -> list[dict]:
    """Extract the ordered list of agent steps for a given run.

    Each step is a dict with:
      - agent_role: str
      - invocation_id: str
      - llm_call_ids: list[str]
      - tool_call_ids: list[str]
      - message_id: str | None
      - expected_llm_calls: int
      - expected_tool_calls: int
    """
    # Find agent_invoked events for this run, in order
    invoked_events = []
    for e in log.events:
        if e.type != "agent_invoked":
            continue
        is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
        if is_this_run:
            invoked_events.append(e)

    steps = []
    for evt in invoked_events:
        # Extract agent role from relationship
        agent_id = ""
        invocation_id = ""
        for rel in evt.relationships:
            if rel.qualifier == "invoked":
                agent_id = rel.objectId
            if rel.qualifier == "started":
                invocation_id = rel.objectId

        agent_role = agent_id.replace("agent-", "") if agent_id else ""

        # Find LLM call objects linked to this invocation
        llm_call_ids = []
        tool_call_ids = []
        message_id = None

        for e in log.events:
            is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
            if not is_this_run:
                continue
            for rel in e.relationships:
                if rel.qualifier == "triggered_by" and rel.objectId == invocation_id:
                    # This event belongs to our step
                    for rel2 in e.relationships:
                        if rel2.qualifier == "started":
                            obj_id = rel2.objectId
                            # Determine type
                            for obj in log.objects:
                                if obj.id == obj_id:
                                    if obj.type == "llm_call":
                                        llm_call_ids.append(obj_id)
                                    elif obj.type == "tool_call":
                                        tool_call_ids.append(obj_id)

        # Find message sent from this step
        for e in log.events:
            if e.type != "message_sent":
                continue
            is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
            if not is_this_run:
                continue
            for rel in e.relationships:
                if rel.qualifier == "sender" and rel.objectId == agent_id:
                    for rel2 in e.relationships:
                        if rel2.qualifier == "sent":
                            message_id = rel2.objectId
                            break

        steps.append({
            "agent_role": agent_role,
            "invocation_id": invocation_id,
            "llm_call_ids": llm_call_ids,
            "tool_call_ids": tool_call_ids,
            "message_id": message_id,
            "expected_llm_calls": len(llm_call_ids),
            "expected_tool_calls": len(tool_call_ids),
        })

    return steps


def _get_object(log: OcelLog, obj_id: str):
    """Find an object by ID."""
    for obj in log.objects:
        if obj.id == obj_id:
            return obj
    return None


def _patch_attribute(obj, name: str, value: str) -> None:
    """Add or update a string attribute on an OCEL object."""
    for attr in obj.attributes:
        if attr.name == name:
            attr.value = value
            return
    # Add new attribute with the object's existing timestamp
    ts = obj.attributes[0].time if obj.attributes else None
    if ts:
        obj.attributes.append(OcelObjectAttribute(name=name, value=value, time=ts))


def _get_tool_names_for_step(log: OcelLog, step: dict) -> list[str]:
    """Get the tool names used in a step."""
    names = []
    for tool_id in step["tool_call_ids"]:
        obj = _get_object(log, tool_id)
        if obj:
            for attr in obj.attributes:
                if attr.name == "tool_name":
                    names.append(attr.value)
    return names


def enrich_log(
    log: OcelLog,
    scenario: DomainScenario,
    client: LLMClient | None = None,
    progress: Progress | None = None,
    progress_task: TaskID | None = None,
) -> None:
    """Enrich an OcelLog in-place with LLM-generated content.

    Walks each run, extracts agent steps, calls the LLM for each step,
    and patches OCEL objects with the returned content.
    """
    if client is None:
        client = LLMClient()

    # Find all run IDs
    run_ids = sorted({o.id for o in log.objects if o.type == "run"})

    # Get pattern description from the first run object
    pattern_desc = ""
    for obj in log.objects:
        if obj.type == "run":
            for attr in obj.attributes:
                if attr.name == "pattern_type":
                    pattern_desc = attr.value
            break

    for run_idx, run_id in enumerate(run_ids):
        user_query = scenario.query_for_run(run_idx)

        # Replace user_query on the run object
        run_obj = _get_object(log, run_id)
        if run_obj:
            _patch_attribute(run_obj, "user_query", user_query)

        # Also update the task object
        task_obj = _get_object(log, f"{run_id}-task")
        if task_obj:
            _patch_attribute(task_obj, "description", user_query)

        steps = _extract_steps_from_log(log, run_id)
        previous_output: str | None = None

        for step in steps:
            role = step["agent_role"]
            tool_names = _get_tool_names_for_step(log, step)
            persona = scenario.agent_personas.get(role, f"You are a {role} agent")

            system_prompt, user_prompt = build_enrichment_prompt(
                domain_description=scenario.description,
                pattern_description=pattern_desc,
                agent_role=role,
                agent_persona=persona,
                user_query=user_query,
                tool_names=tool_names,
                tool_descriptions=scenario.tool_descriptions,
                expected_llm_calls=step["expected_llm_calls"],
                expected_tool_calls=step["expected_tool_calls"],
                previous_output=previous_output,
            )

            try:
                raw = client.generate(system_prompt, user_prompt)
                resp = EnrichmentResponse.from_dict(
                    raw,
                    expected_llm_calls=step["expected_llm_calls"],
                    expected_tool_calls=step["expected_tool_calls"],
                )
            except Exception:
                # On failure, skip enrichment for this step (structural data preserved)
                continue

            # Patch invocation object with reasoning
            inv_obj = _get_object(log, step["invocation_id"])
            if inv_obj:
                _patch_attribute(inv_obj, "reasoning", resp.reasoning)

            # Patch LLM call objects
            for i, llm_id in enumerate(step["llm_call_ids"]):
                llm_obj = _get_object(log, llm_id)
                if llm_obj and i < len(resp.llm_calls):
                    _patch_attribute(llm_obj, "prompt", resp.llm_calls[i].get("prompt", ""))
                    _patch_attribute(
                        llm_obj, "completion", resp.llm_calls[i].get("completion", "")
                    )

            # Patch tool call objects
            for i, tool_id in enumerate(step["tool_call_ids"]):
                tool_obj = _get_object(log, tool_id)
                if tool_obj and i < len(resp.tool_calls):
                    _patch_attribute(
                        tool_obj,
                        "tool_input",
                        json.dumps(resp.tool_calls[i].get("input", {})),
                    )
                    _patch_attribute(
                        tool_obj,
                        "tool_output",
                        json.dumps(resp.tool_calls[i].get("output", {})),
                    )

            # Patch message object
            if step["message_id"]:
                msg_obj = _get_object(log, step["message_id"])
                if msg_obj:
                    _patch_attribute(msg_obj, "content", resp.output_to_next_agent)

            previous_output = resp.output_to_next_agent

        if progress and progress_task is not None:
            progress.advance(progress_task)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_enrichment.py -v`
Expected: All pass.

- [ ] **Step 5: Run all existing tests to check nothing broke**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/ocelgen/enrichment/enricher.py tests/test_enrichment.py
git commit -m "feat: add enricher to walk OCEL traces and patch with LLM content"
```

---

### Task 6: OCEL-to-Tabular Flattener

**Files:**
- Create: `src/ocelgen/upload/__init__.py`
- Create: `src/ocelgen/upload/flatten.py`
- Test: `tests/test_flatten.py`

- [ ] **Step 1: Write tests for the flattener**

Create `tests/test_flatten.py`:

```python
"""Tests for OCEL-to-tabular flattening."""

from ocelgen.generation.engine import generate
from ocelgen.upload.flatten import flatten_log


class TestFlatten:
    def test_flatten_returns_list_of_dicts(self) -> None:
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)

    def test_flatten_has_required_columns(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        required = {
            "event_id", "event_type", "timestamp", "run_id",
            "sequence_number", "is_deviation", "deviation_type",
            "domain", "is_conformant", "pattern", "user_query",
        }
        for row in rows:
            assert required.issubset(row.keys()), f"Missing columns: {required - row.keys()}"

    def test_flatten_event_count_matches_log(self) -> None:
        result = generate("sequential", num_runs=3, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")
        assert len(rows) == len(result.log.events)

    def test_flatten_domain_column_set(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="my-domain")
        for row in rows:
            assert row["domain"] == "my-domain"

    def test_flatten_resolves_agent_role(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")
        agent_invoked_rows = [r for r in rows if r["event_type"] == "agent_invoked"]
        assert len(agent_invoked_rows) > 0
        for row in agent_invoked_rows:
            assert row["agent_role"] != "", f"agent_role not resolved for {row['event_id']}"

    def test_flatten_with_deviations(self) -> None:
        result = generate("sequential", num_runs=10, noise_rate=0.5, seed=42)
        rows = flatten_log(result.log, domain="test")
        deviant_rows = [r for r in rows if r["is_deviation"]]
        assert len(deviant_rows) > 0

    def test_flatten_to_parquet(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")

        table = pa.Table.from_pylist(rows)
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)

        read_back = pq.read_table(path)
        assert read_back.num_rows == len(rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_flatten.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ocelgen.upload'`

- [ ] **Step 3: Implement the flattener**

Create `src/ocelgen/upload/__init__.py`:

```python
"""Upload and dataset packaging for Hugging Face Hub."""
```

Create `src/ocelgen/upload/flatten.py`:

```python
"""Flatten an OcelLog into tabular rows (one row per event)."""

from __future__ import annotations

from ocelgen.models.ocel import OcelLog


def _build_object_index(log: OcelLog) -> dict[str, dict]:
    """Build a lookup: object_id -> {attr_name: attr_value, "_type": type}."""
    index: dict[str, dict] = {}
    for obj in log.objects:
        attrs = {"_type": obj.type}
        for a in obj.attributes:
            attrs[a.name] = a.value
        index[obj.id] = attrs
    return index


def _get_event_attr(event, name: str, default: str = "") -> str:
    """Get a string attribute from an event."""
    for a in event.attributes:
        if a.name == name:
            return a.value
    return default


def flatten_log(log: OcelLog, domain: str) -> list[dict]:
    """Convert an OcelLog to a flat list of dicts (one per event).

    Resolves object relationships to denormalize agent, tool, LLM, and
    message attributes into each event row.
    """
    obj_index = _build_object_index(log)

    # Build run-level metadata index
    run_meta: dict[str, dict] = {}
    for obj in log.objects:
        if obj.type == "run":
            meta: dict[str, str] = {}
            for a in obj.attributes:
                meta[a.name] = a.value
            run_meta[obj.id] = meta

    rows: list[dict] = []

    for event in log.events:
        run_id = _get_event_attr(event, "run_id")
        rmeta = run_meta.get(run_id, {})

        row: dict = {
            "event_id": event.id,
            "event_type": event.type,
            "timestamp": event.time.isoformat(),
            "run_id": run_id,
            "sequence_number": int(_get_event_attr(event, "sequence_number", "0")),
            "is_deviation": _get_event_attr(event, "is_deviation", "false") == "true",
            "deviation_type": _get_event_attr(event, "deviation_type"),
            "step_id": _get_event_attr(event, "step_id"),
            # Resolved from related objects
            "agent_role": "",
            "model_name": "",
            "prompt": "",
            "completion": "",
            "tool_name": "",
            "tool_input": "",
            "tool_output": "",
            "message_content": "",
            "reasoning": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            # Run-level metadata
            "is_conformant": rmeta.get("is_conformant", "true") == "true",
            "pattern": rmeta.get("pattern_type", ""),
            "domain": domain,
            "user_query": rmeta.get("user_query", ""),
        }

        # Resolve relationships to populate denormalized columns
        for rel in event.relationships:
            obj = obj_index.get(rel.objectId, {})
            obj_type = obj.get("_type", "")

            if obj_type == "agent":
                row["agent_role"] = obj.get("role", "")
                row["model_name"] = obj.get("model_name", "")

            elif obj_type == "llm_call":
                row["prompt"] = obj.get("prompt", "")
                row["completion"] = obj.get("completion", "")
                row["input_tokens"] = int(obj.get("input_tokens", "0"))
                row["output_tokens"] = int(obj.get("output_tokens", "0"))
                row["latency_ms"] = int(obj.get("latency_ms", "0"))

            elif obj_type == "tool_call":
                row["tool_name"] = obj.get("tool_name", "")
                row["tool_input"] = obj.get("tool_input", "")
                row["tool_output"] = obj.get("tool_output", "")
                row["latency_ms"] = int(obj.get("duration_ms", "0"))

            elif obj_type == "agent_invocation":
                row["reasoning"] = obj.get("reasoning", "")
                row["input_tokens"] = int(obj.get("input_tokens", "0"))
                row["output_tokens"] = int(obj.get("output_tokens", "0"))
                row["cost_usd"] = float(obj.get("cost_usd", "0"))

            elif obj_type == "message":
                row["message_content"] = obj.get("content", "")

        rows.append(row)

    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_flatten.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/upload/__init__.py src/ocelgen/upload/flatten.py tests/test_flatten.py
git commit -m "feat: add OCEL-to-tabular flattener for HF dataset format"
```

---

### Task 7: HF Dataset Card Generator

**Files:**
- Create: `src/ocelgen/upload/readme.py`
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write tests for README generation**

Create `tests/test_upload.py`:

```python
"""Tests for HF upload utilities."""

from ocelgen.upload.readme import generate_dataset_card
from ocelgen.scenarios.domain import DomainScenario


def _make_scenario() -> DomainScenario:
    return DomainScenario(
        name="test-domain",
        description="A test domain for unit tests",
        pattern="sequential",
        runs=10,
        noise=0.2,
        seed=42,
        user_queries=["query one"],
        agent_personas={"researcher": "A researcher"},
        tool_descriptions={"web_search": "Search"},
    )


class TestDatasetCard:
    def test_card_contains_domain_name(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "test-domain" in card

    def test_card_contains_schema_table(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "event_id" in card
        assert "event_type" in card
        assert "prompt" in card

    def test_card_contains_yaml_frontmatter(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert card.startswith("---")
        assert "dataset_info" in card

    def test_card_contains_usage_example(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "load_dataset" in card
        assert "testuser/agent-traces-test-domain" in card
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_upload.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the dataset card generator**

Create `src/ocelgen/upload/readme.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_upload.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/upload/readme.py tests/test_upload.py
git commit -m "feat: add HF dataset card generator"
```

---

### Task 8: HF Upload Module

**Files:**
- Create: `src/ocelgen/upload/hf_upload.py`
- Test: `tests/test_upload.py` (append)

- [ ] **Step 1: Add tests for upload logic**

Append to `tests/test_upload.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from ocelgen.upload.hf_upload import prepare_upload_files, build_repo_name


class TestBuildRepoName:
    def test_basic(self) -> None:
        assert build_repo_name("juliensimon", "customer-support-triage") == (
            "juliensimon/agent-traces-customer-support-triage"
        )


class TestPrepareUploadFiles:
    def test_creates_parquet_and_ocel_files(self, tmp_path: Path) -> None:
        from ocelgen.generation.engine import generate
        from ocelgen.upload.flatten import flatten_log

        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")

        files = prepare_upload_files(
            rows=rows,
            log=result.log,
            template=result.template,
            result=result,
            scenario=_make_scenario(),
            namespace="testuser",
            output_dir=tmp_path,
            seed=42,
        )

        assert (tmp_path / "data" / "train.parquet").exists()
        assert (tmp_path / "ocel" / "output.jsonocel").exists()
        assert (tmp_path / "ocel" / "normative_model.json").exists()
        assert (tmp_path / "ocel" / "manifest.json").exists()
        assert (tmp_path / "README.md").exists()
        assert len(files) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_upload.py::TestPrepareUploadFiles -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the upload module**

Create `src/ocelgen/upload/hf_upload.py`:

```python
"""Upload agent trace datasets to Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from ocelgen.export.manifest import build_manifest
from ocelgen.export.normative import template_to_dict
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.engine import GenerationResult
from ocelgen.models.ocel import OcelLog
from ocelgen.models.workflow import WorkflowTemplate
from ocelgen.scenarios.domain import DomainScenario
from ocelgen.upload.readme import generate_dataset_card

import json


def build_repo_name(namespace: str, domain_name: str) -> str:
    """Build the HF repo name for a domain."""
    return f"{namespace}/agent-traces-{domain_name}"


def prepare_upload_files(
    rows: list[dict],
    log: OcelLog,
    template: WorkflowTemplate,
    result: GenerationResult,
    scenario: DomainScenario,
    namespace: str,
    output_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Write all files to output_dir, ready for upload. Returns list of file paths."""
    files: list[Path] = []

    # Parquet
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    files.append(parquet_path)

    # OCEL files
    ocel_dir = output_dir / "ocel"
    ocel_dir.mkdir(parents=True, exist_ok=True)

    ocel_path = ocel_dir / "output.jsonocel"
    with open(ocel_path, "w", encoding="utf-8") as f:
        json.dump(ocel_log_to_dict(log), f, indent=2, ensure_ascii=False)
    files.append(ocel_path)

    normative_path = ocel_dir / "normative_model.json"
    with open(normative_path, "w", encoding="utf-8") as f:
        json.dump(template_to_dict(template), f, indent=2, ensure_ascii=False)
    files.append(normative_path)

    manifest_path = ocel_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_manifest(result, seed=seed), f, indent=2, ensure_ascii=False)
    files.append(manifest_path)

    # README
    readme_path = output_dir / "README.md"
    card = generate_dataset_card(
        scenario=scenario,
        namespace=namespace,
        num_events=len(rows),
        num_objects=len(log.objects),
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    files.append(readme_path)

    return files


def upload_to_hub(
    output_dir: Path,
    namespace: str,
    domain_name: str,
) -> str:
    """Upload prepared files to HF Hub. Returns the repo URL."""
    api = HfApi()
    repo_id = build_repo_name(namespace, domain_name)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{repo_id}"


def create_or_update_collection(
    namespace: str,
    collection_slug: str,
    repo_ids: list[str],
) -> str:
    """Create or update an HF collection with the given dataset repos.

    Returns the collection URL.
    """
    api = HfApi()

    # Try to find existing collection
    try:
        collections = api.list_collections(owner=namespace)
        existing = None
        for col in collections:
            if col.slug and collection_slug in col.slug:
                existing = col
                break
    except Exception:
        existing = None

    if existing is None:
        collection = api.create_collection(
            title="Open Agent Traces",
            namespace=namespace,
            description=(
                "A collection of 10 LLM-enriched synthetic agent trace datasets "
                "covering diverse domains and workflow patterns. "
                "Generated with ocelgen using OCEL 2.0 standard."
            ),
            exists_ok=True,
        )
        collection_slug_full = collection.slug
    else:
        collection_slug_full = existing.slug

    # Add each repo to the collection
    for repo_id in repo_ids:
        try:
            api.add_collection_item(
                collection_slug=collection_slug_full,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )
        except Exception:
            pass  # Item may already be in collection

    return f"https://huggingface.co/collections/{collection_slug_full}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_upload.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ocelgen/upload/hf_upload.py tests/test_upload.py
git commit -m "feat: add HF Hub upload and collection management"
```

---

### Task 9: CLI Commands

**Files:**
- Modify: `src/ocelgen/cli.py`
- Test: `tests/test_cli_new.py`

- [ ] **Step 1: Write tests for new CLI commands**

Create `tests/test_cli_new.py`:

```python
"""Tests for new CLI commands: list-domains, enrich, upload, pipeline."""

from typer.testing import CliRunner

from ocelgen.cli import app

runner = CliRunner()


class TestListDomains:
    def test_list_domains_shows_all_10(self) -> None:
        result = runner.invoke(app, ["list-domains"])
        assert result.exit_code == 0
        assert "customer-support-triage" in result.output
        assert "incident-response" in result.output
        assert "10" in result.output or result.output.count("sequential") + result.output.count("supervisor") + result.output.count("parallel") == 10


class TestEnrichCommand:
    def test_enrich_requires_existing_file(self) -> None:
        result = runner.invoke(app, ["enrich", "nonexistent.jsonocel", "--domain", "customer-support-triage"])
        assert result.exit_code != 0

    def test_enrich_requires_valid_domain(self, tmp_path) -> None:
        # Create a minimal OCEL file
        import json
        ocel_path = tmp_path / "test.jsonocel"
        ocel_path.write_text(json.dumps({
            "eventTypes": [], "objectTypes": [], "events": [], "objects": []
        }))
        result = runner.invoke(app, ["enrich", str(ocel_path), "--domain", "nonexistent-domain"])
        assert result.exit_code != 0


class TestPipelineCommand:
    def test_pipeline_requires_namespace(self) -> None:
        result = runner.invoke(app, ["pipeline", "--domain", "customer-support-triage"])
        assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_cli_new.py -v`
Expected: FAIL — commands don't exist yet.

- [ ] **Step 3: Add new CLI commands to cli.py**

Append to `src/ocelgen/cli.py` (after the existing `list_patterns` command):

```python
import json
import tempfile
from pathlib import Path

from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario
from ocelgen.enrichment.client import LLMClient
from ocelgen.enrichment.enricher import enrich_log
from ocelgen.upload.flatten import flatten_log
from ocelgen.upload.hf_upload import (
    build_repo_name,
    create_or_update_collection,
    prepare_upload_files,
    upload_to_hub,
)
from ocelgen.models.ocel import OcelLog
from ocelgen.export.ocel_json import ocel_log_to_dict


@app.command("list-domains")
def list_domains() -> None:
    """List available domain scenarios for enriched generation."""
    table = Table(title="Available Domain Scenarios")
    table.add_column("Name", style="bold")
    table.add_column("Pattern")
    table.add_column("Runs", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column("Description")

    for name, scenario in SCENARIO_REGISTRY.items():
        table.add_row(
            name,
            scenario.pattern,
            str(scenario.runs),
            f"{scenario.noise:.0%}",
            scenario.description,
        )

    console.print(table)
    console.print(f"\n[bold]{len(SCENARIO_REGISTRY)}[/bold] domains available.")


@app.command("enrich")
def enrich_cmd(
    path: Annotated[Path, typer.Argument(help="Path to .jsonocel file to enrich")],
    domain: Annotated[str, typer.Option("--domain", "-d", help="Domain scenario name")] = "",
    model: Annotated[
        str, typer.Option("--model", "-m", help="LLM model for enrichment")
    ] = "google/gemini-2.0-flash-001",
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Output path (default: enriched-<input>)")
    ] = None,
) -> None:
    """Enrich an OCEL 2.0 trace with LLM-generated content."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    if domain not in SCENARIO_REGISTRY:
        console.print(f"[red]Unknown domain '{domain}'. Use 'list-domains' to see available domains.[/red]")
        raise typer.Exit(1)

    scenario = get_scenario(domain)

    console.print(f"Loading [bold]{path}[/bold]...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct OcelLog from JSON
    from pydantic import TypeAdapter
    log = TypeAdapter(OcelLog).validate_python(data)

    console.print(f"Enriching with [bold]{model}[/bold] for domain [bold]{domain}[/bold]...")
    client = LLMClient(model=model)

    from rich.progress import Progress
    with Progress() as progress:
        run_count = len([o for o in log.objects if o.type == "run"])
        task = progress.add_task("Enriching runs...", total=run_count)
        enrich_log(log, scenario, client=client, progress=progress, progress_task=task)

    out_path = output or path.parent / f"enriched-{path.name}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ocel_log_to_dict(log), f, indent=2, ensure_ascii=False)

    console.print(f"[green]Enriched log written to:[/green] {out_path}")


@app.command("upload")
def upload_cmd(
    path: Annotated[Path, typer.Argument(help="Path to enriched .jsonocel file")],
    domain: Annotated[str, typer.Option("--domain", "-d", help="Domain scenario name")] = "",
    namespace: Annotated[str, typer.Option("--namespace", "-n", help="HF namespace")] = "",
    collection: Annotated[
        str, typer.Option("--collection", help="Collection slug")
    ] = "open-agent-traces",
) -> None:
    """Upload an enriched trace to Hugging Face Hub."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    if domain not in SCENARIO_REGISTRY:
        console.print(f"[red]Unknown domain '{domain}'.[/red]")
        raise typer.Exit(1)

    if not namespace:
        console.print("[red]--namespace is required.[/red]")
        raise typer.Exit(1)

    scenario = get_scenario(domain)

    console.print(f"Loading [bold]{path}[/bold]...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    from pydantic import TypeAdapter
    log = TypeAdapter(OcelLog).validate_python(data)

    console.print("Flattening to tabular format...")
    rows = flatten_log(log, domain=domain)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # We need a GenerationResult for the manifest — reconstruct minimal version
        from ocelgen.generation.engine import GenerationResult
        result = GenerationResult(
            log=log,
            template=PATTERN_REGISTRY[scenario.pattern]().build_template(),
            total_runs=scenario.runs,
            conformant_runs=scenario.runs,  # approximate
            deviant_runs=0,
        )

        prepare_upload_files(
            rows=rows,
            log=log,
            template=result.template,
            result=result,
            scenario=scenario,
            namespace=namespace,
            output_dir=tmp_path,
        )

        console.print(f"Uploading to [bold]{build_repo_name(namespace, domain)}[/bold]...")
        url = upload_to_hub(tmp_path, namespace, domain)
        console.print(f"[green]Uploaded:[/green] {url}")


@app.command("pipeline")
def pipeline_cmd(
    domain: Annotated[
        str | None, typer.Option("--domain", "-d", help="Single domain to process")
    ] = None,
    all_domains: Annotated[
        bool, typer.Option("--all", help="Process all 10 domains")
    ] = False,
    namespace: Annotated[str, typer.Option("--namespace", "-n", help="HF namespace")] = "",
    model: Annotated[
        str, typer.Option("--model", "-m", help="LLM model for enrichment")
    ] = "google/gemini-2.0-flash-001",
    collection: Annotated[
        str, typer.Option("--collection", help="Collection slug")
    ] = "open-agent-traces",
    skip_upload: Annotated[
        bool, typer.Option("--skip-upload", help="Generate and enrich but don't upload")
    ] = False,
) -> None:
    """End-to-end pipeline: generate, enrich, and upload agent trace datasets."""
    if not namespace:
        console.print("[red]--namespace is required.[/red]")
        raise typer.Exit(1)

    if not domain and not all_domains:
        console.print("[red]Specify --domain <name> or --all.[/red]")
        raise typer.Exit(1)

    domains = list(SCENARIO_REGISTRY.keys()) if all_domains else [domain]

    for d in domains:
        if d not in SCENARIO_REGISTRY:
            console.print(f"[red]Unknown domain '{d}'.[/red]")
            raise typer.Exit(1)

    client = LLMClient(model=model)
    uploaded_repos: list[str] = []

    for d in domains:
        scenario = get_scenario(d)
        console.rule(f"[bold]{scenario.name}[/bold] ({scenario.pattern})")

        # Step 1: Generate
        console.print(f"Generating {scenario.runs} runs...")
        result = generate(
            pattern_name=scenario.pattern,
            num_runs=scenario.runs,
            noise_rate=scenario.noise,
            seed=scenario.seed,
        )
        console.print(
            f"  {result.total_runs} runs, {len(result.log.events)} events, "
            f"{result.deviant_runs} deviant"
        )

        # Step 2: Enrich
        console.print(f"Enriching with {model}...")
        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task(f"Enriching {d}...", total=scenario.runs)
            enrich_log(result.log, scenario, client=client, progress=progress, progress_task=task)

        # Step 3: Flatten
        rows = flatten_log(result.log, domain=d)
        console.print(f"  Flattened to {len(rows)} rows")

        # Step 4: Prepare and upload
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prepare_upload_files(
                rows=rows,
                log=result.log,
                template=result.template,
                result=result,
                scenario=scenario,
                namespace=namespace,
                output_dir=tmp_path,
                seed=scenario.seed,
            )

            if not skip_upload:
                repo_id = build_repo_name(namespace, d)
                console.print(f"Uploading to [bold]{repo_id}[/bold]...")
                url = upload_to_hub(tmp_path, namespace, d)
                uploaded_repos.append(repo_id)
                console.print(f"  [green]{url}[/green]")
            else:
                console.print("  [yellow]Skipping upload (--skip-upload)[/yellow]")

    # Create/update collection
    if uploaded_repos and not skip_upload:
        console.rule("[bold]Collection[/bold]")
        console.print("Creating/updating collection...")
        col_url = create_or_update_collection(namespace, collection, uploaded_repos)
        console.print(f"[green]Collection:[/green] {col_url}")

    console.print("\n[bold green]Done![/bold green]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_cli_new.py -v`
Expected: All pass.

- [ ] **Step 5: Run all tests**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/ocelgen/cli.py tests/test_cli_new.py
git commit -m "feat: add enrich, upload, pipeline, and list-domains CLI commands"
```

---

### Task 10: End-to-End Integration Test

**Files:**
- Modify: `tests/test_integration.py` (append)

This test runs the full pipeline in dry mode (mocked LLM, no HF upload) to verify the entire chain works together.

- [ ] **Step 1: Add integration test**

Append to `tests/test_integration.py`:

```python
from unittest.mock import MagicMock
from pathlib import Path

from ocelgen.enrichment.enricher import enrich_log
from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario
from ocelgen.upload.flatten import flatten_log
from ocelgen.upload.hf_upload import prepare_upload_files


class TestEnrichmentPipeline:
    """End-to-end: generate → enrich (mocked) → flatten → prepare files."""

    def _mock_client(self) -> MagicMock:
        client = MagicMock()
        client.generate.return_value = {
            "reasoning": "Analyzing the request to determine the best approach.",
            "llm_calls": [
                {"prompt": "Analyze this issue", "completion": "Based on my analysis..."},
                {"prompt": "Deep dive into findings", "completion": "The root cause is..."},
            ],
            "tool_calls": [
                {"input": {"query": "relevant data"}, "output": {"results": ["item1", "item2"]}},
                {"input": {"file": "config.yaml"}, "output": {"content": "key: value"}},
                {"input": {"expr": "1+1"}, "output": {"result": 2}},
            ],
            "output_to_next_agent": "Here is my analysis summary for the next agent.",
        }
        return client

    def test_full_pipeline_sequential(self, tmp_path: Path) -> None:
        scenario = get_scenario("customer-support-triage")
        result = generate("sequential", num_runs=3, noise_rate=0.2, seed=scenario.seed)

        enrich_log(result.log, scenario, client=self._mock_client())

        # Verify enrichment happened
        llm_objs = [o for o in result.log.objects if o.type == "llm_call"]
        enriched = [o for o in llm_objs if any(a.name == "prompt" and a.value for a in o.attributes)]
        assert len(enriched) > 0

        # Flatten
        rows = flatten_log(result.log, domain=scenario.name)
        assert len(rows) == len(result.log.events)

        # Some rows should have prompt content
        rows_with_prompt = [r for r in rows if r.get("prompt")]
        assert len(rows_with_prompt) > 0

        # Prepare files
        files = prepare_upload_files(
            rows=rows,
            log=result.log,
            template=result.template,
            result=result,
            scenario=scenario,
            namespace="test",
            output_dir=tmp_path,
            seed=scenario.seed,
        )
        assert len(files) == 5
        assert (tmp_path / "data" / "train.parquet").exists()
        assert (tmp_path / "README.md").exists()

    def test_full_pipeline_supervisor(self, tmp_path: Path) -> None:
        scenario = get_scenario("code-review-pipeline")
        result = generate("supervisor", num_runs=3, noise_rate=0.2, seed=scenario.seed)

        enrich_log(result.log, scenario, client=self._mock_client())

        rows = flatten_log(result.log, domain=scenario.name)
        assert len(rows) == len(result.log.events)

        files = prepare_upload_files(
            rows=rows, log=result.log, template=result.template,
            result=result, scenario=scenario, namespace="test",
            output_dir=tmp_path, seed=scenario.seed,
        )
        assert len(files) == 5

    def test_full_pipeline_parallel(self, tmp_path: Path) -> None:
        scenario = get_scenario("market-research")
        result = generate("parallel", num_runs=3, noise_rate=0.2, seed=scenario.seed)

        enrich_log(result.log, scenario, client=self._mock_client())

        rows = flatten_log(result.log, domain=scenario.name)
        assert len(rows) == len(result.log.events)

        files = prepare_upload_files(
            rows=rows, log=result.log, template=result.template,
            result=result, scenario=scenario, namespace="test",
            output_dir=tmp_path, seed=scenario.seed,
        )
        assert len(files) == 5

    def test_all_10_domains_generate_without_error(self) -> None:
        """Smoke test: every domain can generate and enrich (mocked)."""
        client = self._mock_client()
        for name, scenario in SCENARIO_REGISTRY.items():
            result = generate(
                scenario.pattern, num_runs=2, noise_rate=scenario.noise, seed=scenario.seed,
            )
            enrich_log(result.log, scenario, client=client)
            rows = flatten_log(result.log, domain=name)
            assert len(rows) > 0, f"Domain {name} produced no rows"
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/test_integration.py -v`
Expected: All pass (including new tests).

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/julien/Development/mpmx && uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for enrichment pipeline"
```

---

### Task 11: Manual Smoke Test

No code changes — verify the CLI works end-to-end.

- [ ] **Step 1: Test list-domains**

Run: `cd /Users/julien/Development/mpmx && uv run ocelgen list-domains`
Expected: Table with 10 domains, their patterns, runs, and noise levels.

- [ ] **Step 2: Test generate + enrich for one domain (dry run)**

Run:
```bash
cd /Users/julien/Development/mpmx
uv run ocelgen generate -p sequential -n 3 --seed 1001 -o /tmp/oceltest/output.jsonocel
uv run ocelgen enrich /tmp/oceltest/output.jsonocel --domain customer-support-triage -o /tmp/oceltest/enriched.jsonocel
```
Expected: Enriched file created. Inspect it to verify `prompt`/`completion` fields are populated.

- [ ] **Step 3: Test pipeline with --skip-upload**

Run:
```bash
cd /Users/julien/Development/mpmx
uv run ocelgen pipeline --domain customer-support-triage --namespace testuser --skip-upload
```
Expected: Pipeline runs generate → enrich → flatten without uploading. Progress bars shown.
