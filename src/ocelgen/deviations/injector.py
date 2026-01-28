"""Orchestrates deviation injection into conformant runs.

The injector decides which runs get deviations (based on noise_rate),
picks deviation types using weighted random selection, and applies
the corresponding strategies.
"""

from __future__ import annotations

import random

from ocelgen.deviations.registry import get_strategy
from ocelgen.deviations.types import DeviationConfig, DeviationSpec, DeviationType
from ocelgen.models.ocel import OcelLog
from ocelgen.models.workflow import WorkflowTemplate


class DeviationInjector:
    """Orchestrates deviation injection across multiple runs."""

    def __init__(self, config: DeviationConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def should_inject(self) -> bool:
        """Decide if a run should receive deviations."""
        return self.rng.random() < self.config.noise_rate

    def select_deviations(self) -> list[DeviationType]:
        """Pick which deviation types to inject for a single run."""
        num = self.rng.randint(1, self.config.max_deviations_per_run)
        types = list(self.config.weights.keys())
        weights = list(self.config.weights.values())
        return self.rng.choices(types, weights=weights, k=num)

    def inject(
        self,
        log: OcelLog,
        run_id: str,
        template: WorkflowTemplate,
    ) -> list[DeviationSpec]:
        """Inject deviations into a single run within the log.

        Returns the list of deviations that were actually applied.
        """
        if not self.should_inject():
            return []

        selected = self.select_deviations()
        specs: list[DeviationSpec] = []

        for dev_type in selected:
            strategy = get_strategy(dev_type)
            spec = strategy.apply(log, run_id, template, self.rng)
            if spec is not None:
                specs.append(spec)

        return specs
