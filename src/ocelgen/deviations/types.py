"""Deviation type definitions and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeviationType(str, Enum):
    """All supported deviation types for conformance checking."""

    SKIPPED_ACTIVITY = "skipped_activity"
    INSERTED_ACTIVITY = "inserted_activity"
    WRONG_RESOURCE = "wrong_resource"
    SWAPPED_ORDER = "swapped_order"
    WRONG_TOOL = "wrong_tool"
    REPEATED_ACTIVITY = "repeated_activity"
    TIMEOUT = "timeout"
    WRONG_ROUTING = "wrong_routing"
    MISSING_HANDOFF = "missing_handoff"
    EXTRA_LLM_CALL = "extra_llm_call"


# Default weights for random deviation selection
DEFAULT_WEIGHTS: dict[DeviationType, float] = {
    DeviationType.SKIPPED_ACTIVITY: 1.5,
    DeviationType.INSERTED_ACTIVITY: 1.0,
    DeviationType.WRONG_RESOURCE: 1.0,
    DeviationType.SWAPPED_ORDER: 1.0,
    DeviationType.WRONG_TOOL: 1.2,
    DeviationType.REPEATED_ACTIVITY: 0.8,
    DeviationType.TIMEOUT: 0.7,
    DeviationType.WRONG_ROUTING: 0.8,
    DeviationType.MISSING_HANDOFF: 0.9,
    DeviationType.EXTRA_LLM_CALL: 1.0,
}


@dataclass
class DeviationConfig:
    """Configuration for deviation injection."""

    noise_rate: float = 0.2  # Fraction of runs that get deviations
    max_deviations_per_run: int = 3
    weights: dict[DeviationType, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class DeviationSpec:
    """Record of a single injected deviation (for the manifest)."""

    run_id: str
    deviation_type: DeviationType
    step_id: str | None = None
    description: str = ""
