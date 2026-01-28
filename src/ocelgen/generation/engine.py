"""Top-level generation orchestrator.

Coordinates run simulation, deviation injection, and output assembly
for batch generation of OCEL 2.0 event logs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from faker import Faker

from ocelgen.deviations.injector import DeviationInjector
from ocelgen.deviations.types import DeviationConfig, DeviationSpec
from ocelgen.generation.run_simulator import RunSimulator
from ocelgen.models.ocel import OcelLog
from ocelgen.models.workflow import WorkflowTemplate
from ocelgen.patterns.base import BasePattern
from ocelgen.patterns.parallel import ParallelPattern
from ocelgen.patterns.sequential import SequentialPattern
from ocelgen.patterns.supervisor import SupervisorPattern

PATTERN_REGISTRY: dict[str, type[BasePattern]] = {
    "sequential": SequentialPattern,
    "supervisor": SupervisorPattern,
    "parallel": ParallelPattern,
}


@dataclass
class GenerationResult:
    """Result of a batch generation run."""

    log: OcelLog
    template: WorkflowTemplate
    deviations: list[DeviationSpec] = field(default_factory=list)
    total_runs: int = 0
    conformant_runs: int = 0
    deviant_runs: int = 0


def generate(
    pattern_name: str,
    num_runs: int,
    noise_rate: float = 0.2,
    max_deviations_per_run: int = 3,
    seed: int | None = None,
) -> GenerationResult:
    """Generate a batch of OCEL 2.0 event log runs.

    Args:
        pattern_name: One of "sequential", "supervisor", "parallel".
        num_runs: Number of runs to generate.
        noise_rate: Fraction of runs to inject with deviations (0.0–1.0).
        max_deviations_per_run: Max deviations per deviant run.
        seed: Random seed for reproducibility.

    Returns:
        GenerationResult containing the merged log, template, and deviation records.
    """
    if pattern_name not in PATTERN_REGISTRY:
        raise ValueError(
            f"Unknown pattern '{pattern_name}'. Available: {list(PATTERN_REGISTRY.keys())}"
        )

    # Initialize RNGs
    master_seed = seed if seed is not None else random.randint(0, 2**31)
    rng = random.Random(master_seed)
    Faker.seed(master_seed)

    pattern = PATTERN_REGISTRY[pattern_name]()
    template = pattern.build_template()

    deviation_config = DeviationConfig(
        noise_rate=noise_rate,
        max_deviations_per_run=max_deviations_per_run,
    )
    injector = DeviationInjector(deviation_config, rng)

    merged_log = OcelLog()
    all_deviations: list[DeviationSpec] = []
    conformant_count = 0
    deviant_count = 0

    # Stagger run start times across a simulated day
    base_time = datetime(2025, 1, 15, 8, 0, 0, tzinfo=UTC)

    for i in range(num_runs):
        # Each run starts slightly after the previous one
        run_base = base_time + timedelta(
            seconds=i * rng.uniform(5, 30),
        )
        run_rng = random.Random(rng.randint(0, 2**31))
        run_faker = Faker()
        run_faker.seed_instance(rng.randint(0, 2**31))

        sim = RunSimulator(
            template=template,
            run_index=i,
            rng=run_rng,
            faker=run_faker,
            base_time=run_base,
        )
        run_log = sim.simulate()

        # Inject deviations
        specs = injector.inject(run_log, sim.run_id, template)
        if specs:
            all_deviations.extend(specs)
            deviant_count += 1
        else:
            conformant_count += 1

        merged_log.merge(run_log)

    return GenerationResult(
        log=merged_log,
        template=template,
        deviations=all_deviations,
        total_runs=num_runs,
        conformant_runs=conformant_count,
        deviant_runs=deviant_count,
    )
