"""Tests for the deviation injection system."""

import random

from faker import Faker

from ocelgen.deviations.injector import DeviationInjector
from ocelgen.deviations.registry import STRATEGY_REGISTRY, get_strategy
from ocelgen.deviations.strategies import (
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
from ocelgen.deviations.types import DeviationConfig, DeviationType
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.run_simulator import RunSimulator
from ocelgen.patterns.sequential import SequentialPattern
from ocelgen.validation.schema import validate_ocel_dict


def _make_conformant_log(seed: int = 42):
    """Generate a conformant sequential run."""
    template = SequentialPattern().build_template()
    rng = random.Random(seed)
    faker = Faker()
    faker.seed_instance(seed)
    sim = RunSimulator(template, run_index=0, rng=rng, faker=faker)
    log = sim.simulate()
    return log, template


class TestDeviationRegistry:
    def test_all_types_registered(self) -> None:
        for dt in DeviationType:
            assert dt in STRATEGY_REGISTRY, f"{dt} not registered"

    def test_get_strategy_returns_instance(self) -> None:
        for dt in DeviationType:
            strategy = get_strategy(dt)
            assert strategy.deviation_type == dt


class TestSkippedActivity:
    def test_removes_events(self) -> None:
        log, template = _make_conformant_log()
        orig_count = len(log.events)
        rng = random.Random(99)
        strategy = SkippedActivityStrategy()
        spec = strategy.apply(log, "run-0000", template, rng)
        assert spec is not None
        assert spec.deviation_type == DeviationType.SKIPPED_ACTIVITY
        assert len(log.events) < orig_count

    def test_marks_run_nonconformant(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        SkippedActivityStrategy().apply(log, "run-0000", template, rng)
        for obj in log.objects:
            if obj.id == "run-0000":
                for attr in obj.attributes:
                    if attr.name == "is_conformant":
                        assert attr.value == "false"

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        SkippedActivityStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == [], f"Validation errors: {errors}"


class TestInsertedActivity:
    def test_adds_events(self) -> None:
        log, template = _make_conformant_log()
        orig_count = len(log.events)
        rng = random.Random(99)
        spec = InsertedActivityStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        assert len(log.events) > orig_count

    def test_inserted_events_marked(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        InsertedActivityStrategy().apply(log, "run-0000", template, rng)
        deviant_events = [
            e for e in log.events
            if any(a.name == "is_deviation" and a.value == "true" for a in e.attributes)
        ]
        assert len(deviant_events) >= 2  # invoked + completed

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        InsertedActivityStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestWrongResource:
    def test_changes_agent(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        spec = WrongResourceStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        assert "instead of" in spec.description

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        WrongResourceStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestSwappedOrder:
    def test_swaps_events(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        spec = SwappedOrderStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        assert spec.deviation_type == DeviationType.SWAPPED_ORDER

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        SwappedOrderStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestWrongTool:
    def test_changes_tool(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        spec = WrongToolStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        WrongToolStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestRepeatedActivity:
    def test_adds_retry(self) -> None:
        log, template = _make_conformant_log()
        orig_count = len(log.events)
        rng = random.Random(99)
        spec = RepeatedActivityStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        assert len(log.events) > orig_count
        retry_events = [e for e in log.events if e.type == "retry_started"]
        assert len(retry_events) >= 1

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        RepeatedActivityStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestTimeout:
    def test_adds_error_event(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        spec = TimeoutStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        error_events = [e for e in log.events if e.type == "error_occurred"]
        assert len(error_events) >= 1

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        TimeoutStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestWrongRouting:
    def test_adds_routing_event(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        spec = WrongRoutingStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        routing_events = [e for e in log.events if e.type == "routing_decided"]
        assert len(routing_events) >= 1

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        WrongRoutingStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestMissingHandoff:
    def test_removes_message(self) -> None:
        log, template = _make_conformant_log()
        orig_msgs = len([e for e in log.events if e.type == "message_sent"])
        rng = random.Random(99)
        spec = MissingHandoffStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        new_msgs = len([e for e in log.events if e.type == "message_sent"])
        assert new_msgs < orig_msgs

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        MissingHandoffStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestExtraLLMCall:
    def test_adds_llm_events(self) -> None:
        log, template = _make_conformant_log()
        orig_count = len(log.events)
        rng = random.Random(99)
        spec = ExtraLLMCallStrategy().apply(log, "run-0000", template, rng)
        assert spec is not None
        assert len(log.events) > orig_count

    def test_still_valid_ocel(self) -> None:
        log, template = _make_conformant_log()
        rng = random.Random(99)
        ExtraLLMCallStrategy().apply(log, "run-0000", template, rng)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == []


class TestDeviationInjector:
    def test_respects_noise_rate_zero(self) -> None:
        config = DeviationConfig(noise_rate=0.0)
        rng = random.Random(42)
        injector = DeviationInjector(config, rng)
        log, template = _make_conformant_log()
        specs = injector.inject(log, "run-0000", template)
        assert specs == []

    def test_always_injects_at_rate_one(self) -> None:
        config = DeviationConfig(noise_rate=1.0, max_deviations_per_run=1)
        rng = random.Random(42)
        injector = DeviationInjector(config, rng)
        log, template = _make_conformant_log()
        specs = injector.inject(log, "run-0000", template)
        assert len(specs) >= 1

    def test_result_still_valid_ocel(self) -> None:
        config = DeviationConfig(noise_rate=1.0, max_deviations_per_run=2)
        rng = random.Random(42)
        injector = DeviationInjector(config, rng)
        log, template = _make_conformant_log()
        injector.inject(log, "run-0000", template)
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        assert errors == [], f"Validation errors: {errors}"
