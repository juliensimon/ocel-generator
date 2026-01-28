"""Generate realistic LLM and tool call attributes.

Token counts, costs, and latencies are modeled to approximate real-world
LangChain agent runs. Values are drawn from distributions calibrated to
typical GPT-4o and Claude usage patterns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ocelgen.models.langchain import LLMModel

# Cost per 1K tokens (input/output) by model
_COST_PER_1K: dict[str, tuple[float, float]] = {
    LLMModel.GPT4O.value: (0.0025, 0.01),
    LLMModel.GPT4O_MINI.value: (0.00015, 0.0006),
    LLMModel.CLAUDE_SONNET.value: (0.003, 0.015),
    LLMModel.CLAUDE_HAIKU.value: (0.00025, 0.00125),
}


@dataclass
class LLMCallAttributes:
    """Attributes for a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float


@dataclass
class ToolCallAttributes:
    """Attributes for a single tool execution."""

    tool_name: str
    tool_kind: str
    duration_ms: int
    status: str  # "success" or "error"


def generate_llm_attributes(rng: random.Random, model: LLMModel) -> LLMCallAttributes:
    """Generate realistic LLM call attributes."""
    input_tokens = int(rng.gauss(800, 300))
    input_tokens = max(50, input_tokens)
    output_tokens = int(rng.gauss(400, 200))
    output_tokens = max(20, output_tokens)

    # Latency correlates with output tokens
    base_latency = rng.gauss(500, 200)
    token_latency = output_tokens * rng.uniform(1.5, 3.0)
    latency_ms = max(100, int(base_latency + token_latency))

    input_cost, output_cost = _COST_PER_1K.get(
        model.value, (0.003, 0.015)
    )
    cost = (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost

    return LLMCallAttributes(
        model=model.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=round(cost, 6),
    )


def generate_tool_attributes(
    rng: random.Random, tool_name: str, tool_kind: str
) -> ToolCallAttributes:
    """Generate realistic tool call attributes."""
    duration_ms = max(50, int(rng.lognormvariate(5.5, 1.0)))
    return ToolCallAttributes(
        tool_name=tool_name,
        tool_kind=tool_kind,
        duration_ms=duration_ms,
        status="success",
    )
