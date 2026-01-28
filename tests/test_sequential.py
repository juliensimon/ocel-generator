"""Tests for domain models, sequential pattern, and run simulation."""

import random

from faker import Faker

from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.run_simulator import RunSimulator
from ocelgen.patterns.sequential import SequentialPattern
from ocelgen.validation.schema import validate_ocel_dict


class TestWorkflowTemplate:
    def test_sequential_template_structure(self) -> None:
        pattern = SequentialPattern()
        template = pattern.build_template()
        assert template.name == "sequential"
        assert len(template.steps) == 3
        assert len(template.edges) == 2

    def test_start_and_end_steps(self) -> None:
        template = SequentialPattern().build_template()
        assert template.start_step.id == "research"
        assert len(template.end_steps) == 1
        assert template.end_steps[0].id == "summarize"

    def test_topological_order(self) -> None:
        template = SequentialPattern().build_template()
        order = template.topological_order()
        ids = [s.id for s in order]
        assert ids == ["research", "analyze", "summarize"]

    def test_successors(self) -> None:
        template = SequentialPattern().build_template()
        succs = template.successors("research")
        assert len(succs) == 1
        assert succs[0].id == "analyze"

    def test_predecessors(self) -> None:
        template = SequentialPattern().build_template()
        preds = template.predecessors("summarize")
        assert len(preds) == 1
        assert preds[0].id == "analyze"


class TestRunSimulator:
    def _simulate_run(self, seed: int = 42) -> dict:
        pattern = SequentialPattern()
        template = pattern.build_template()
        rng = random.Random(seed)
        faker = Faker()
        faker.seed_instance(seed)
        sim = RunSimulator(template, run_index=0, rng=rng, faker=faker)
        log = sim.simulate()
        return ocel_log_to_dict(log)

    def test_produces_valid_ocel(self) -> None:
        data = self._simulate_run()
        errors = validate_ocel_dict(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_has_all_event_types(self) -> None:
        data = self._simulate_run()
        event_type_names = {et["name"] for et in data["eventTypes"]}
        assert "run_started" in event_type_names
        assert "agent_invoked" in event_type_names
        assert "llm_request_sent" in event_type_names
        assert "llm_response_received" in event_type_names
        assert "tool_called" in event_type_names
        assert "tool_returned" in event_type_names
        assert "message_sent" in event_type_names
        assert "agent_completed" in event_type_names
        assert "run_completed" in event_type_names

    def test_has_all_object_types(self) -> None:
        data = self._simulate_run()
        obj_type_names = {ot["name"] for ot in data["objectTypes"]}
        assert "run" in obj_type_names
        assert "agent" in obj_type_names
        assert "agent_invocation" in obj_type_names
        assert "llm_call" in obj_type_names
        assert "tool_call" in obj_type_names
        assert "message" in obj_type_names
        assert "task" in obj_type_names

    def test_event_ordering(self) -> None:
        data = self._simulate_run()
        events = data["events"]
        # First event is run_started, last is run_completed
        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_completed"
        # Events are in chronological order
        times = [e["time"] for e in events]
        assert times == sorted(times)

    def test_sequence_numbers_monotonic(self) -> None:
        data = self._simulate_run()
        events = data["events"]
        seq_nums = []
        for e in events:
            for attr in e.get("attributes", []):
                if attr["name"] == "sequence_number":
                    seq_nums.append(int(attr["value"]))
        assert seq_nums == sorted(seq_nums)
        assert len(set(seq_nums)) == len(seq_nums)  # All unique

    def test_three_agent_invocations(self) -> None:
        """Sequential pattern should invoke exactly 3 agents."""
        data = self._simulate_run()
        invoked_events = [e for e in data["events"] if e["type"] == "agent_invoked"]
        completed_events = [e for e in data["events"] if e["type"] == "agent_completed"]
        assert len(invoked_events) == 3
        assert len(completed_events) == 3

    def test_inter_agent_messages(self) -> None:
        """Should have 2 messages: research→analyze, analyze→summarize."""
        data = self._simulate_run()
        msg_events = [e for e in data["events"] if e["type"] == "message_sent"]
        assert len(msg_events) == 2

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical output."""
        data1 = self._simulate_run(seed=123)
        data2 = self._simulate_run(seed=123)
        assert data1 == data2

    def test_different_seeds_differ(self) -> None:
        data1 = self._simulate_run(seed=1)
        data2 = self._simulate_run(seed=2)
        # Events should differ (different timestamps, token counts, etc.)
        assert data1 != data2

    def test_all_events_have_deviation_attrs(self) -> None:
        """Every event should carry is_deviation and deviation_type."""
        data = self._simulate_run()
        for event in data["events"]:
            attr_names = {a["name"] for a in event.get("attributes", [])}
            assert "is_deviation" in attr_names, f"Event {event['id']} missing is_deviation"
            assert "deviation_type" in attr_names, f"Event {event['id']} missing deviation_type"

    def test_conformant_run_has_no_deviations(self) -> None:
        data = self._simulate_run()
        for event in data["events"]:
            for attr in event.get("attributes", []):
                if attr["name"] == "is_deviation":
                    assert attr["value"] == "false", (
                        f"Event {event['id']} marked as deviation in conformant run"
                    )
