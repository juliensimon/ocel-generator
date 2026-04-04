Real agent traces are scarce. If you're building observability, evaluation, or debugging tools for multi-agent systems, you know the pain — production traces are proprietary, and toy examples don't cut it.

I built open-agent-traces to fix this. It generates structurally valid, semantically rich execution traces that look and feel like real multi-agent workflows:

- 10 enterprise domains (customer support, code review, incident response, financial analysis...)
- 3 workflow patterns (sequential, supervisor/worker, parallel fan-out)
- LLM-enriched content — real prompts, completions, tool calls, agent reasoning
- Labeled anomalies for training detectors (wrong tools, skipped steps, timeouts)
- OCEL 2.0 standard — works with PM4Py, Celonis, and other process mining tools

pip install open-agent-traces

Pre-built dataset on Hugging Face: https://huggingface.co/datasets/juliensimon/open-agent-traces
Code: https://github.com/juliensimon/ocel-generator

MIT licensed. Contributions welcome.

#AI #agents #opensource #processmining
