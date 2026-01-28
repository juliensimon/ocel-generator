"""Maps DeviationType → Strategy class for lookup."""

from __future__ import annotations

from ocelgen.deviations.strategies import (
    DeviationStrategy,
    ExtraLLMCallStrategy,
    InsertedActivityStrategy,
    MissingHandoffStrategy,
    RepeatedActivityStrategy,
    SkippedActivityStrategy,
    SwappedOrderStrategy,
    TimeoutStrategy,
    WrongResourceStrategy,
    WrongRoutingStrategy,
    WrongToolStrategy,
)
from ocelgen.deviations.types import DeviationType

STRATEGY_REGISTRY: dict[DeviationType, type[DeviationStrategy]] = {
    DeviationType.SKIPPED_ACTIVITY: SkippedActivityStrategy,
    DeviationType.INSERTED_ACTIVITY: InsertedActivityStrategy,
    DeviationType.WRONG_RESOURCE: WrongResourceStrategy,
    DeviationType.SWAPPED_ORDER: SwappedOrderStrategy,
    DeviationType.WRONG_TOOL: WrongToolStrategy,
    DeviationType.REPEATED_ACTIVITY: RepeatedActivityStrategy,
    DeviationType.TIMEOUT: TimeoutStrategy,
    DeviationType.WRONG_ROUTING: WrongRoutingStrategy,
    DeviationType.MISSING_HANDOFF: MissingHandoffStrategy,
    DeviationType.EXTRA_LLM_CALL: ExtraLLMCallStrategy,
}


def get_strategy(deviation_type: DeviationType) -> DeviationStrategy:
    """Get an instance of the strategy for the given deviation type."""
    cls = STRATEGY_REGISTRY.get(deviation_type)
    if cls is None:
        raise ValueError(f"No strategy registered for {deviation_type}")
    return cls()
