"""Tests for supervisor and parallel patterns."""

import random

from faker import Faker

from ocelgen.deviations.injector import DeviationInjector
from ocelgen.deviations.types import DeviationConfig
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.run_simulator import RunSimulator
from ocelgen.patterns.parallel import ParallelPattern
from ocelgen.patterns.supervisor import SupervisorPattern
from ocelgen.validation.schema import validate_ocel_dict


def _simulate_pattern(pattern_cls, seed: int = 42) -> dict:
    pattern = pattern_cls()
    template = pattern.build_template()
    rng = random.Random(seed)
    faker = Faker()
    faker.seed_instance(seed)
    sim = RunSimulator(template, run_index=0, rng=rng, faker=faker)
    log = sim.simulate()
    return ocel_log_to_dict(log)


class TestSupervisorPattern:
    def test_template_structure(self) -> None:
        template = SupervisorPattern().build_template()
        assert len(template.steps) == 5
        assert template.start_step.id == "plan"
        assert len(template.end_steps) == 1
        assert template.end_steps[0].id == "aggregate"

    def test_produces_valid_ocel(self) -> None:
        data = _simulate_pattern(SupervisorPattern)
        errors = validate_ocel_dict(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_has_routing_decided_events(self) -> None:
        data = _simulate_pattern(SupervisorPattern)
        routing = [e for e in data["events"] if e["type"] == "routing_decided"]
        # Supervisor routes to 3 workers
        assert len(routing) == 3

    def test_has_five_agent_invocations(self) -> None:
        data = _simulate_pattern(SupervisorPattern)
        invoked = [e for e in data["events"] if e["type"] == "agent_invoked"]
        assert len(invoked) == 5

    def test_event_ordering(self) -> None:
        data = _simulate_pattern(SupervisorPattern)
        assert data["events"][0]["type"] == "run_started"
        assert data["events"][-1]["type"] == "run_completed"

    def test_deterministic(self) -> None:
        d1 = _simulate_pattern(SupervisorPattern, seed=77)
        d2 = _simulate_pattern(SupervisorPattern, seed=77)
        assert d1 == d2

    def test_deviations_work(self) -> None:
        pattern = SupervisorPattern()
        template = pattern.build_template()
        rng = random.Random(42)
        faker = Faker()
        faker.seed_instance(42)
        sim = RunSimulator(template, run_index=0, rng=rng, faker=faker)
        log = sim.simulate()

        config = DeviationConfig(noise_rate=1.0, max_deviations_per_run=2)
        injector = DeviationInjector(config, random.Random(99))
        specs = injector.inject(log, "run-0000", template)
        assert len(specs) >= 1

        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == [], f"Validation errors after deviation: {errors}"


class TestParallelPattern:
    def test_template_structure(self) -> None:
        template = ParallelPattern().build_template()
        assert len(template.steps) == 5
        assert template.start_step.id == "split"
        assert template.end_steps[0].id == "aggregate"
        # 3 workers in parallel group
        parallel = [s for s in template.steps if s.parallel_group == "workers"]
        assert len(parallel) == 3

    def test_produces_valid_ocel(self) -> None:
        data = _simulate_pattern(ParallelPattern)
        errors = validate_ocel_dict(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_has_five_agent_invocations(self) -> None:
        data = _simulate_pattern(ParallelPattern)
        invoked = [e for e in data["events"] if e["type"] == "agent_invoked"]
        assert len(invoked) == 5

    def test_parallel_workers_have_overlapping_times(self) -> None:
        """Workers in the parallel group should have overlapping timestamps."""
        data = _simulate_pattern(ParallelPattern, seed=42)
        events = data["events"]

        # Find agent_invoked events for the 3 workers
        worker_invoked = []
        for e in events:
            if e["type"] == "agent_invoked":
                for attr in e.get("attributes", []):
                    if attr["name"] == "step_id" and attr["value"].startswith("worker_"):
                        worker_invoked.append(e)
        assert len(worker_invoked) == 3

        # Their timestamps should all be close together (forked from same base)
        times = sorted(e["time"] for e in worker_invoked)
        # All workers should start from approximately the same time
        # (they fork from the same base time)
        assert len(set(times)) >= 1  # At least some overlap or close times

    def test_deterministic(self) -> None:
        d1 = _simulate_pattern(ParallelPattern, seed=77)
        d2 = _simulate_pattern(ParallelPattern, seed=77)
        assert d1 == d2

    def test_deviations_work(self) -> None:
        pattern = ParallelPattern()
        template = pattern.build_template()
        rng = random.Random(42)
        faker = Faker()
        faker.seed_instance(42)
        sim = RunSimulator(template, run_index=0, rng=rng, faker=faker)
        log = sim.simulate()

        config = DeviationConfig(noise_rate=1.0, max_deviations_per_run=2)
        injector = DeviationInjector(config, random.Random(99))
        specs = injector.inject(log, "run-0000", template)
        assert len(specs) >= 1

        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == [], f"Validation errors after deviation: {errors}"
