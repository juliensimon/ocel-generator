"""Temporal ordering validation for OCEL 2.0 agent trace logs.

Checks causal ordering constraints specific to agent traces:
- Events within each run are chronologically ordered
- Paired events respect causal order (request before response)
- run_started is the first event and run_completed is the last
"""

from __future__ import annotations

from datetime import datetime

from ocelgen.models.ocel import OcelEvent, OcelLog


def validate_temporal_order(log: OcelLog) -> list[str]:
    """Validate chronological and causal ordering of events within each run.

    Returns a list of error messages (empty if all ordering is valid).
    """
    errors: list[str] = []

    # Group events by run_id
    events_by_run: dict[str, list[OcelEvent]] = {}
    orphaned_count = 0
    for event in log.events:
        run_id = ""
        for attr in event.attributes:
            if attr.name == "run_id":
                run_id = attr.value
                break
        if run_id:
            events_by_run.setdefault(run_id, []).append(event)
        else:
            orphaned_count += 1

    if orphaned_count:
        errors.append(f"{orphaned_count} event(s) have no 'run_id' attribute")

    for run_id, events in events_by_run.items():
        # Sort by time for chronological checks
        sorted_events = sorted(events, key=lambda e: e.time)

        # Check run_started is first
        if sorted_events and sorted_events[0].type != "run_started":
            errors.append(
                f"Run '{run_id}': first event is '{sorted_events[0].type}', expected 'run_started'"
            )

        # Check run_completed is last
        if sorted_events and sorted_events[-1].type != "run_completed":
            errors.append(
                f"Run '{run_id}': last event is '{sorted_events[-1].type}', "
                f"expected 'run_completed'"
            )

        # Check sequence_number monotonicity
        seq_numbers: list[int] = []
        for event in sorted_events:
            for attr in event.attributes:
                if attr.name == "sequence_number":
                    try:
                        seq_numbers.append(int(attr.value))
                    except ValueError:
                        errors.append(
                            f"Run '{run_id}': event '{event.id}' has "
                            f"non-integer sequence_number '{attr.value}'"
                        )
        for i in range(1, len(seq_numbers)):
            if seq_numbers[i] < seq_numbers[i - 1]:
                errors.append(
                    f"Run '{run_id}': sequence numbers not monotonic "
                    f"(seq {seq_numbers[i]} follows {seq_numbers[i - 1]})"
                )

        # Check causal pairs: "started" qualifier must precede "completed"
        # for the same object (single start/complete cycle per object assumed)
        _check_causal_pairs(run_id, sorted_events, errors)

    return errors


def _check_causal_pairs(run_id: str, events: list[OcelEvent], errors: list[str]) -> None:
    """Check that 'started' events precede 'completed' events for each object."""
    # Map: objectId -> {qualifier: [(event_id, time)]}
    object_events: dict[str, dict[str, list[tuple[str, datetime]]]] = {}

    for event in events:
        for rel in event.relationships:
            key = rel.objectId
            if key not in object_events:
                object_events[key] = {}
            qual = rel.qualifier
            if qual not in object_events[key]:
                object_events[key][qual] = []
            object_events[key][qual].append((event.id, event.time))

    for obj_id, quals in object_events.items():
        started = quals.get("started", [])
        completed = quals.get("completed", [])
        if started and completed:
            earliest_start = min(t for _, t in started)
            earliest_complete = min(t for _, t in completed)
            if earliest_complete < earliest_start:
                errors.append(
                    f"Run '{run_id}': object '{obj_id}' was completed before it was started"
                )
