# TODO

What's shipped, what's next, and what's on the radar.

## Shipped (v0.2.0)

- [x] 3 workflow patterns (sequential, supervisor, parallel)
- [x] 10 built-in enterprise domains with LLM-enriched content
- [x] 10 deviation types with ground-truth labels
- [x] Custom domains via YAML configuration
- [x] Any OpenAI-compatible LLM backend (cloud + local via `--base-url`)
- [x] 5 semantic validators (schema, integrity, temporal, type attributes, conformance)
- [x] PM4Py compatibility (validated across all 10 domains)
- [x] CLI: generate, enrich, validate, pipeline, upload, list-patterns, list-domains
- [x] Python API for programmatic generation and validation
- [x] Pre-built dataset on Hugging Face (17,000+ events)
- [x] 265 tests, 98% coverage, pre-commit hooks (ruff, mypy, pytest)
- [x] Release pipeline: CI + PyPI + TestPyPI + GitHub Releases (SHA-pinned, OIDC)

## Next (v0.3.0)

### Export formats
- [ ] OpenTelemetry span export — emit traces as OTel spans for ingestion by Jaeger, Datadog, etc.
- [ ] LangSmith-compatible export — generate traces that load directly in LangSmith's trace viewer
- [ ] CSV/JSON-lines export — flat file formats for simpler downstream tooling

### Generation improvements
- [ ] Streaming generation — yield events as they're generated instead of batch-and-return
- [ ] Conditional branching — patterns with if/else routing based on agent output
- [ ] Multi-turn conversations — agent steps that involve back-and-forth with the user
- [ ] Configurable agent/tool inventories per pattern (not just per domain)

### Enrichment improvements
- [ ] Batch enrichment — parallel LLM calls for faster enrichment of large logs
- [ ] Caching — skip re-enrichment of unchanged runs
- [ ] Cost tracking — report total LLM spend at end of enrichment
- [ ] Structured output validation — verify LLM responses match expected JSON schema before patching

### Validation
- [ ] `ocelgen validate --deep` CLI flag — run all 5 semantic validators from the CLI, not just JSON schema
- [ ] Validation report as JSON — machine-readable output for CI integration

## Future ideas

### Framework-specific traces
- [ ] LangGraph state machine traces with checkpointing events
- [ ] CrewAI crew/task/agent hierarchy traces
- [ ] AutoGen conversation pattern traces
- [ ] Smolagents tool-use traces

### Advanced deviation types
- [ ] Hallucination injection — agent produces content contradicting its tool outputs
- [ ] Cascading failures — one agent's error propagates through downstream agents
- [ ] Latency spikes — realistic tail-latency patterns (p99 outliers)
- [ ] Token budget exceeded — agent truncates output mid-sentence

### Evaluation
- [ ] LLM-as-judge scoring — rate enriched trace quality with a judge model
- [ ] Human evaluation harness — side-by-side comparison of traces from different models
- [ ] Conformance checking benchmarks — standard test suite for evaluating process mining algorithms

### Scale
- [ ] Large-scale generation (10K+ runs) with memory-efficient streaming
- [ ] Multi-dataset generation — generate all 10 domains in a single parallelized pipeline run
- [ ] Incremental dataset updates — add runs to an existing HF dataset without regenerating

## Contributing

Issues and PRs welcome at [github.com/juliensimon/ocel-generator](https://github.com/juliensimon/ocel-generator).
