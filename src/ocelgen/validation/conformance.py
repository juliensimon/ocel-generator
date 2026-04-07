"""Workflow conformance validation for OCEL 2.0 agent trace logs.

Checks that conformant runs (is_conformant=true) actually follow the
normative workflow template:
- Agent invocations appear in the order defined by the template
- Parallel groups are treated as unordered sets (any permutation is valid)
- Every template step is represented in the run
- Agent roles match the expected roles for each step
"""

from __future__ import annotations

from ocelgen.models.ocel import OcelLog
from ocelgen.models.workflow import WorkflowTemplate


def _build_expected_segments(template: WorkflowTemplate) -> list[list[str]]:
    """Build expected role segments from the template.

    Sequential steps become single-element segments. Parallel groups
    become multi-element segments where any permutation is valid.

    Returns a list of segments, where each segment is a list of expected roles.
    """
    ordered = template.topological_order()
    segments: list[list[str]] = []
    i = 0
    while i < len(ordered):
        step = ordered[i]
        if step.parallel_group:
            # Collect all steps in this parallel group
            group_roles: list[str] = []
            group_id = step.parallel_group
            while i < len(ordered) and ordered[i].parallel_group == group_id:
                group_roles.append(ordered[i].agent_role.value)
                i += 1
            segments.append(group_roles)
        else:
            segments.append([step.agent_role.value])
            i += 1
    return segments


def _build_expected_step_segments(template: WorkflowTemplate) -> list[list[str]]:
    """Same as _build_expected_segments but for step IDs."""
    ordered = template.topological_order()
    segments: list[list[str]] = []
    i = 0
    while i < len(ordered):
        step = ordered[i]
        if step.parallel_group:
            group_ids: list[str] = []
            group_id = step.parallel_group
            while i < len(ordered) and ordered[i].parallel_group == group_id:
                group_ids.append(ordered[i].id)
                i += 1
            segments.append(group_ids)
        else:
            segments.append([step.id])
            i += 1
    return segments


def validate_workflow_conformance(log: OcelLog, template: WorkflowTemplate) -> list[str]:
    """Validate that conformant runs follow the workflow template.

    Only checks runs marked as is_conformant=true. Deviant runs are
    expected to diverge and are not validated here.

    Parallel groups are treated as unordered sets — any permutation
    of worker roles within a group is valid.

    Returns a list of error messages (empty if all conformant runs match).
    """
    errors: list[str] = []

    expected_steps = template.topological_order()
    role_segments = _build_expected_segments(template)
    step_segments = _build_expected_step_segments(template)

    # Flatten expected counts
    expected_count = sum(len(seg) for seg in role_segments)

    # Identify conformant runs
    conformant_run_ids: set[str] = set()
    for obj in log.objects:
        if obj.type != "run":
            continue
        is_conformant = any(a.name == "is_conformant" and a.value == "true" for a in obj.attributes)
        if is_conformant:
            conformant_run_ids.add(obj.id)

    for run_id in sorted(conformant_run_ids):
        # Extract agent_invoked events for this run, sorted by time
        invoked_events = sorted(
            [
                e
                for e in log.events
                if e.type == "agent_invoked"
                and any(a.name == "run_id" and a.value == run_id for a in e.attributes)
            ],
            key=lambda e: e.time,
        )

        # Resolve agent roles from object attributes (not ID parsing)
        obj_index = {o.id: o for o in log.objects}
        actual_roles: list[str] = []
        actual_step_ids: list[str] = []
        for event in invoked_events:
            role = ""
            step_id = ""
            for rel in event.relationships:
                if rel.qualifier == "invoked":
                    agent_obj = obj_index.get(rel.objectId)
                    if agent_obj:
                        for obj_attr in agent_obj.attributes:
                            if obj_attr.name == "role":
                                role = obj_attr.value
                                break
                    if not role:
                        errors.append(
                            f"Run '{run_id}': agent_invoked event '{event.id}' "
                            f"references object '{rel.objectId}' with no 'role' attribute"
                        )
            for evt_attr in event.attributes:
                if evt_attr.name == "step_id":
                    step_id = evt_attr.value
            actual_roles.append(role)
            actual_step_ids.append(step_id)

        # Check step count matches
        if len(actual_roles) != expected_count:
            errors.append(
                f"Run '{run_id}': expected {expected_count} steps, got {len(actual_roles)}"
            )
            continue

        # Check role segments (parallel groups are unordered)
        pos = 0
        for seg_idx, segment in enumerate(role_segments):
            actual_chunk = actual_roles[pos : pos + len(segment)]
            if len(segment) == 1:
                # Sequential step — exact match required
                if actual_chunk != segment:
                    errors.append(
                        f"Run '{run_id}' step {pos}: expected role "
                        f"'{segment[0]}', got '{actual_chunk[0] if actual_chunk else '?'}'"
                    )
            else:
                # Parallel group — any permutation is valid
                if sorted(actual_chunk) != sorted(segment):
                    errors.append(
                        f"Run '{run_id}' parallel group at step {pos}: "
                        f"expected roles {sorted(segment)}, got {sorted(actual_chunk)}"
                    )
            pos += len(segment)

        # Check step ID segments
        pos = 0
        for seg_idx, segment in enumerate(step_segments):
            actual_chunk = actual_step_ids[pos : pos + len(segment)]
            if len(segment) == 1:
                if actual_chunk != segment:
                    errors.append(
                        f"Run '{run_id}' step {pos}: expected step_id "
                        f"'{segment[0]}', got '{actual_chunk[0] if actual_chunk else '?'}'"
                    )
            else:
                if sorted(actual_chunk) != sorted(segment):
                    errors.append(
                        f"Run '{run_id}' parallel group at step {pos}: "
                        f"expected step_ids {sorted(segment)}, got {sorted(actual_chunk)}"
                    )
            pos += len(segment)

        # Check that agent_completed events exist for each step
        completed_step_ids = set()
        for event in log.events:
            if event.type != "agent_completed":
                continue
            is_this_run = any(a.name == "run_id" and a.value == run_id for a in event.attributes)
            if is_this_run:
                for ea in event.attributes:
                    if ea.name == "step_id":
                        completed_step_ids.add(ea.value)

        for step in expected_steps:
            if step.id not in completed_step_ids:
                errors.append(f"Run '{run_id}': step '{step.id}' was invoked but never completed")

    return errors
