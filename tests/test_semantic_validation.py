"""Semantic validation tests for OCEL 2.0 logs.

These tests validate properties that go beyond JSON schema compliance:
referential integrity, temporal ordering, and workflow conformance.

They also serve as **code samples** showing how to work with generated
logs programmatically — inspecting events, objects, relationships,
and validating domain-specific invariants.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ocelgen.generation.engine import generate
from ocelgen.models.ocel import (
    OcelEvent,
    OcelEventAttribute,
    OcelObject,
    OcelObjectAttribute,
    OcelRelationship,
)
from ocelgen.validation.integrity import (
    validate_referential_integrity,
    validate_type_attributes,
)
from ocelgen.validation.temporal import validate_temporal_order

# ---------------------------------------------------------------------------
# Helpers: generate logs for different patterns/configs
# ---------------------------------------------------------------------------


def _generate(pattern: str = "sequential", runs: int = 10, noise: float = 0.2, seed: int = 42):
    """Shorthand to generate a log. Also a code sample for the README."""
    return generate(
        pattern_name=pattern,
        num_runs=runs,
        noise_rate=noise,
        seed=seed,
    )


# ===================================================================
# 1. Referential integrity
# ===================================================================


class TestReferentialIntegrity:
    """Every objectId in a relationship must resolve to an existing object."""

    # --- Code sample: validate a generated log ---

    def test_sequential_clean(self) -> None:
        """Generate a clean sequential log and verify all references resolve."""
        result = _generate("sequential", runs=20, noise=0.0)
        errors = validate_referential_integrity(result.log)
        assert errors == [], "Dangling references found:\n" + "\n".join(errors)

    def test_supervisor_clean(self) -> None:
        result = _generate("supervisor", runs=20, noise=0.0)
        errors = validate_referential_integrity(result.log)
        assert errors == []

    def test_parallel_clean(self) -> None:
        result = _generate("parallel", runs=20, noise=0.0)
        errors = validate_referential_integrity(result.log)
        assert errors == []

    def test_sequential_with_deviations(self) -> None:
        """Deviation injection must not break referential integrity."""
        result = _generate("sequential", runs=50, noise=1.0, seed=123)
        errors = validate_referential_integrity(result.log)
        assert errors == [], "Deviations broke integrity:\n" + "\n".join(errors)

    def test_supervisor_with_deviations(self) -> None:
        result = _generate("supervisor", runs=30, noise=1.0, seed=456)
        errors = validate_referential_integrity(result.log)
        assert errors == []

    def test_parallel_with_deviations(self) -> None:
        result = _generate("parallel", runs=30, noise=1.0, seed=789)
        errors = validate_referential_integrity(result.log)
        assert errors == []

    # --- Code sample: detect a dangling reference ---

    def test_detects_dangling_event_relationship(self) -> None:
        """Manually inject a dangling reference and verify detection."""
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log

        # Inject an event that references a non-existent object
        log.events.append(
            OcelEvent(
                id="fake-evt-001",
                type="run_started",
                time=datetime(2025, 1, 1, tzinfo=UTC),
                attributes=[OcelEventAttribute(name="run_id", value="run-9999")],
                relationships=[
                    OcelRelationship(objectId="ghost-object-999", qualifier="started"),
                ],
            )
        )

        errors = validate_referential_integrity(log)
        assert any("ghost-object-999" in e for e in errors)

    def test_detects_duplicate_event_ids(self) -> None:
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log
        # Duplicate the first event
        log.events.append(log.events[0].model_copy())
        errors = validate_referential_integrity(log)
        assert any("Duplicate event ID" in e for e in errors)

    def test_detects_undeclared_event_type(self) -> None:
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log
        log.events.append(
            OcelEvent(
                id="bad-type-evt",
                type="totally_made_up_event",
                time=datetime(2025, 1, 1, tzinfo=UTC),
                attributes=[],
            )
        )
        errors = validate_referential_integrity(log)
        assert any("undeclared type" in e for e in errors)

    def test_detects_undeclared_object_type(self) -> None:
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log
        log.objects.append(
            OcelObject(
                id="bad-type-obj",
                type="nonexistent_type",
                attributes=[],
            )
        )
        errors = validate_referential_integrity(log)
        assert any("undeclared type" in e for e in errors)


class TestTypeAttributes:
    """Every attribute on an instance must be declared in its type schema."""

    def test_all_event_attributes_declared(self) -> None:
        """All event attributes should match their eventType declarations."""
        result = _generate("sequential", runs=10, noise=0.5)
        errors = validate_type_attributes(result.log)
        assert errors == [], "Undeclared attributes:\n" + "\n".join(errors)

    def test_all_object_attributes_declared(self) -> None:
        result = _generate("supervisor", runs=10, noise=0.5)
        errors = validate_type_attributes(result.log)
        assert errors == []

    def test_detects_undeclared_event_attribute(self) -> None:
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log
        # Add a bogus attribute to the first event
        log.events[0].attributes.append(OcelEventAttribute(name="secret_field", value="oops"))
        errors = validate_type_attributes(log)
        assert any("secret_field" in e for e in errors)

    def test_detects_undeclared_object_attribute(self) -> None:
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log
        log.objects[0].attributes.append(
            OcelObjectAttribute(
                name="mystery_attr",
                value="oops",
                time=datetime(2025, 1, 1, tzinfo=UTC),
            )
        )
        errors = validate_type_attributes(log)
        assert any("mystery_attr" in e for e in errors)


# ===================================================================
# 2. Temporal ordering
# ===================================================================


class TestTemporalOrder:
    """Events within a run must respect chronological and causal ordering."""

    def test_sequential_ordering(self) -> None:
        """Clean sequential runs have correct temporal order."""
        result = _generate("sequential", runs=20, noise=0.0)
        errors = validate_temporal_order(result.log)
        assert errors == [], "Temporal order violations:\n" + "\n".join(errors)

    def test_supervisor_ordering(self) -> None:
        result = _generate("supervisor", runs=20, noise=0.0)
        errors = validate_temporal_order(result.log)
        assert errors == []

    def test_parallel_ordering(self) -> None:
        """Parallel workers interleave sequence numbers by design.
        Temporal checks still verify run boundaries and causal pairs."""
        result = _generate("parallel", runs=20, noise=0.0)
        errors = validate_temporal_order(result.log)
        # Filter out sequence monotonicity errors — parallel workers
        # interleave their sequences by design
        causal_errors = [e for e in errors if "sequence" not in e]
        assert causal_errors == [], "Temporal violations:\n" + "\n".join(causal_errors)

    def test_run_starts_with_run_started(self) -> None:
        """Code sample: extract events for a specific run."""
        result = _generate("sequential", runs=5, noise=0.0)

        # How to get events for a specific run
        run_events = [
            e
            for e in result.log.events
            if any(a.name == "run_id" and a.value == "run-0000" for a in e.attributes)
        ]
        run_events.sort(key=lambda e: e.time)

        assert run_events[0].type == "run_started"
        assert run_events[-1].type == "run_completed"

    def test_llm_request_before_response(self) -> None:
        """Code sample: trace an LLM call's lifecycle through events."""
        result = _generate("sequential", runs=1, noise=0.0)

        # Find all llm_call objects
        llm_calls = [o for o in result.log.objects if o.type == "llm_call"]
        assert len(llm_calls) > 0

        for llm_obj in llm_calls:
            # Find the request and response events for this LLM call
            request_time = None
            response_time = None

            for event in result.log.events:
                for rel in event.relationships:
                    if rel.objectId == llm_obj.id:
                        if rel.qualifier == "started":
                            request_time = event.time
                        elif rel.qualifier == "completed":
                            response_time = event.time

            if request_time and response_time:
                assert request_time <= response_time, (
                    f"LLM call '{llm_obj.id}': response at {response_time} "
                    f"before request at {request_time}"
                )

    def test_sequence_numbers_are_monotonic(self) -> None:
        """Code sample: extract sequence numbers from a run."""
        result = _generate("sequential", runs=3, noise=0.0)

        for run_idx in range(3):
            run_id = f"run-{run_idx:04d}"
            run_events = sorted(
                [
                    e
                    for e in result.log.events
                    if any(a.name == "run_id" and a.value == run_id for a in e.attributes)
                ],
                key=lambda e: e.time,
            )

            seq_numbers = []
            for e in run_events:
                for a in e.attributes:
                    if a.name == "sequence_number":
                        seq_numbers.append(int(a.value))

            for i in range(1, len(seq_numbers)):
                assert seq_numbers[i] >= seq_numbers[i - 1]

    def test_detects_completed_before_started(self) -> None:
        """Inject a causal violation: swap started/completed timestamps."""
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log

        # Find a pair of started/completed events for the same object
        for event in log.events:
            if event.type == "llm_response_received":
                # This has qualifier "completed" — swap its time with the request
                for req in log.events:
                    if req.type == "llm_request_sent":
                        # Check they share the same object
                        req_obj = {
                            r.objectId for r in req.relationships if r.qualifier == "started"
                        }
                        comp_obj = {
                            r.objectId for r in event.relationships if r.qualifier == "completed"
                        }
                        if req_obj & comp_obj:
                            # Swap times so completed is before started
                            event.time, req.time = req.time, event.time
                            errors = validate_temporal_order(log)
                            causal_errors = [
                                e for e in errors if "completed" in e and "before" in e
                            ]
                            assert len(causal_errors) > 0, "Should detect completed before started"
                            return
        # If we get here, the test data didn't have the expected structure
        assert False, "Could not find a started/completed pair to swap"

    def test_detects_orphaned_events(self) -> None:
        """Events without run_id should be reported."""
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log

        # Add an event with no run_id
        log.events.append(
            OcelEvent(
                id="orphan-evt",
                type="run_started",
                time=datetime(2025, 1, 1, tzinfo=UTC),
                attributes=[],  # no run_id
            )
        )
        errors = validate_temporal_order(log)
        assert any("no 'run_id'" in e for e in errors)

    def test_deviations_may_alter_ordering(self) -> None:
        """Deviation strategies (e.g., SwappedOrder) intentionally reorder
        events. We verify that conformant runs still pass causal checks,
        while deviant runs are allowed to violate them."""
        result = _generate("sequential", runs=20, noise=1.0)

        # Identify which runs are deviant
        deviant_run_ids = {spec.run_id for spec in result.deviations}

        errors = validate_temporal_order(result.log)
        # Filter to only conformant-run causal violations
        conformant_causal_errors = [
            e
            for e in errors
            if ("completed" in e and "before" in e) and not any(rid in e for rid in deviant_run_ids)
        ]
        assert conformant_causal_errors == [], (
            "Causal violations in conformant runs:\n" + "\n".join(conformant_causal_errors)
        )


# ===================================================================
# 3. Workflow conformance
# ===================================================================

from ocelgen.validation.conformance import validate_workflow_conformance


class TestWorkflowConformance:
    """Conformant runs must follow the normative workflow template."""

    def test_conformant_runs_follow_template(self) -> None:
        """All runs generated with noise=0 should be conformant."""
        result = _generate("sequential", runs=20, noise=0.0)
        errors = validate_workflow_conformance(result.log, result.template)
        assert errors == [], "Conformance violations:\n" + "\n".join(errors)

    def test_supervisor_conformant(self) -> None:
        result = _generate("supervisor", runs=20, noise=0.0)
        errors = validate_workflow_conformance(result.log, result.template)
        assert errors == []

    def test_parallel_conformant(self) -> None:
        result = _generate("parallel", runs=20, noise=0.0)
        errors = validate_workflow_conformance(result.log, result.template)
        assert errors == []

    def test_agent_roles_match_template_steps(self) -> None:
        """Code sample: map agent invocations back to template steps."""
        result = _generate("sequential", runs=1, noise=0.0)
        template = result.template

        # Get the expected agent roles from the template
        expected_roles = [step.agent_role.value for step in template.topological_order()]

        # Extract agent invocations for run-0000
        invoked_events = sorted(
            [
                e
                for e in result.log.events
                if e.type == "agent_invoked"
                and any(a.name == "run_id" and a.value == "run-0000" for a in e.attributes)
            ],
            key=lambda e: e.time,
        )

        # Resolve agent roles from relationships
        actual_roles = []
        for event in invoked_events:
            for rel in event.relationships:
                if rel.qualifier == "invoked":
                    # agent id is "agent-<role>"
                    role = rel.objectId.replace("agent-", "")
                    actual_roles.append(role)

        assert actual_roles == expected_roles

    def test_every_template_step_has_events(self) -> None:
        """Code sample: verify every workflow step produced events."""
        result = _generate("sequential", runs=1, noise=0.0)
        template = result.template

        for step in template.steps:
            # Find agent_invoked events for this step
            step_events = [
                e
                for e in result.log.events
                if e.type == "agent_invoked"
                and any(a.name == "step_id" and a.value == step.id for a in e.attributes)
                and any(a.name == "run_id" and a.value == "run-0000" for a in e.attributes)
            ]
            assert len(step_events) == 1, (
                f"Step '{step.id}' ({step.agent_role.value}) expected 1 "
                f"agent_invoked event, got {len(step_events)}"
            )

    # --- Negative tests: detect violations ---

    def test_detects_wrong_step_order(self) -> None:
        """Inject wrong agent ordering into a conformant run."""
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log

        # Swap the agent relationships on the first two agent_invoked events
        invoked = [e for e in log.events if e.type == "agent_invoked"]
        if len(invoked) >= 2:
            # Swap the "invoked" relationship objectIds
            for rel in invoked[0].relationships:
                if rel.qualifier == "invoked":
                    old_id = rel.objectId
                    break
            for rel in invoked[1].relationships:
                if rel.qualifier == "invoked":
                    rel.objectId, old_id = old_id, rel.objectId
            for rel in invoked[0].relationships:
                if rel.qualifier == "invoked":
                    rel.objectId = old_id

        errors = validate_workflow_conformance(log, result.template)
        assert any("expected role" in e for e in errors), (
            f"Should detect wrong step order, got: {errors}"
        )

    def test_detects_missing_step(self) -> None:
        """Remove an agent_invoked event from a conformant run."""
        result = _generate("sequential", runs=1, noise=0.0)
        log = result.log

        # Remove the last agent_invoked event for run-0000
        invoked = [
            e
            for e in log.events
            if e.type == "agent_invoked"
            and any(a.name == "run_id" and a.value == "run-0000" for a in e.attributes)
        ]
        if invoked:
            log.events.remove(invoked[-1])

        errors = validate_workflow_conformance(log, result.template)
        assert any("expected" in e and "steps" in e for e in errors), (
            f"Should detect missing step, got: {errors}"
        )

    def test_deviant_runs_are_correctly_flagged(self) -> None:
        """Code sample: cross-check manifest ground truth vs. event attributes."""
        result = _generate("sequential", runs=50, noise=0.5, seed=42)

        # Build ground truth from deviations
        deviant_run_ids = {spec.run_id for spec in result.deviations}

        for run_obj in result.log.objects:
            if run_obj.type != "run":
                continue
            is_conformant = any(
                a.name == "is_conformant" and a.value == "true" for a in run_obj.attributes
            )
            if run_obj.id in deviant_run_ids:
                assert not is_conformant, (
                    f"Run '{run_obj.id}' has deviations but is marked conformant"
                )
            else:
                assert is_conformant, (
                    f"Run '{run_obj.id}' has no deviations but is marked non-conformant"
                )


# ===================================================================
# 4. PM4Py round-trip
# ===================================================================

from pathlib import Path

import pytest

from ocelgen.export.ocel_json import write_ocel_json

try:
    import pm4py
except ImportError:
    pm4py = None  # type: ignore[assignment]


@pytest.mark.skipif(pm4py is None, reason="pm4py not installed")
class TestPM4PyRoundTrip:
    """Verify logs can be loaded and queried by pm4py — the reference OCEL 2.0 library.

    These tests also serve as **code samples** showing how to use generated
    traces with pm4py for process mining analysis.
    """

    def _write_and_load(self, result, tmp_path: Path):
        """Helper: write log to disk and load with pm4py."""
        ocel_path = tmp_path / "test.jsonocel"
        write_ocel_json(result.log, ocel_path)
        return pm4py.read.read_ocel2_json(str(ocel_path))

    def test_sequential_loads(self, tmp_path: Path) -> None:
        """Code sample: generate a log and load it in pm4py."""
        result = _generate("sequential", runs=10, noise=0.2)
        ocel = self._write_and_load(result, tmp_path)

        # pm4py returns DataFrames for events and objects
        assert len(ocel.events) > 0
        assert len(ocel.objects) > 0

    def test_supervisor_loads(self, tmp_path: Path) -> None:
        result = _generate("supervisor", runs=10, noise=0.2)
        ocel = self._write_and_load(result, tmp_path)
        assert len(ocel.events) > 0

    def test_parallel_loads(self, tmp_path: Path) -> None:
        result = _generate("parallel", runs=10, noise=0.2)
        ocel = self._write_and_load(result, tmp_path)
        assert len(ocel.events) > 0

    def test_event_types_preserved(self, tmp_path: Path) -> None:
        """Code sample: inspect event types via pm4py.

        Note: pm4py uses 'ocel:activity' for event types (not 'ocel:type',
        which is reserved for object types).
        """
        result = _generate("sequential", runs=5, noise=0.0)
        ocel = self._write_and_load(result, tmp_path)

        # pm4py uses 'ocel:activity' for event types
        event_types = set(ocel.events["ocel:activity"].unique())
        assert "run_started" in event_types
        assert "run_completed" in event_types
        assert "agent_invoked" in event_types
        assert "llm_request_sent" in event_types

    def test_object_types_preserved(self, tmp_path: Path) -> None:
        """Code sample: inspect object types via pm4py."""
        result = _generate("sequential", runs=5, noise=0.0)
        ocel = self._write_and_load(result, tmp_path)

        object_types = set(ocel.objects["ocel:type"].unique())
        assert "run" in object_types
        assert "agent" in object_types
        assert "llm_call" in object_types

    def test_event_count_matches(self, tmp_path: Path) -> None:
        """Verify pm4py reads the same number of events we wrote."""
        result = _generate("sequential", runs=10, noise=0.0)
        ocel = self._write_and_load(result, tmp_path)
        assert len(ocel.events) == len(result.log.events)

    def test_object_count_matches(self, tmp_path: Path) -> None:
        result = _generate("sequential", runs=10, noise=0.0)
        ocel = self._write_and_load(result, tmp_path)
        assert len(ocel.objects) == len(result.log.objects)

    def test_large_log_with_deviations(self, tmp_path: Path) -> None:
        """Stress test: 100 runs with full noise still loads correctly."""
        result = _generate("sequential", runs=100, noise=1.0, seed=99)
        ocel = self._write_and_load(result, tmp_path)
        assert len(ocel.events) == len(result.log.events)

    def test_relationships_loaded(self, tmp_path: Path) -> None:
        """Code sample: access event-to-object relationships in pm4py."""
        result = _generate("sequential", runs=3, noise=0.0)
        ocel = self._write_and_load(result, tmp_path)

        # pm4py stores relations in the relations DataFrame
        assert hasattr(ocel, "relations")
        if ocel.relations is not None and len(ocel.relations) > 0:
            assert "ocel:eid" in ocel.relations.columns
            assert "ocel:oid" in ocel.relations.columns
