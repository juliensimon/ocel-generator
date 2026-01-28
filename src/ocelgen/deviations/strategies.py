"""Concrete deviation strategies — one class per DeviationType.

Each strategy mutates an OcelLog in-place to inject a specific kind of
process deviation. The strategy marks affected events with is_deviation=true
and sets the deviation_type attribute.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import timedelta

from ocelgen.deviations.types import DeviationSpec, DeviationType
from ocelgen.generation.attributes import generate_llm_attributes
from ocelgen.models.langchain import AgentRole, LLMModel, ToolKind
from ocelgen.models.ocel import (
    OcelEvent,
    OcelEventAttribute,
    OcelLog,
    OcelObject,
    OcelObjectAttribute,
    OcelRelationship,
)
from ocelgen.models.workflow import WorkflowTemplate


def _mark_event_as_deviation(event: OcelEvent, dev_type: DeviationType) -> None:
    """Set is_deviation=true and deviation_type on an event."""
    for attr in event.attributes:
        if attr.name == "is_deviation":
            attr.value = "true"
        if attr.name == "deviation_type":
            attr.value = dev_type.value


def _mark_run_nonconformant(log: OcelLog, run_id: str) -> None:
    """Mark a run object as non-conformant."""
    for obj in log.objects:
        if obj.id == run_id:
            for attr in obj.attributes:
                if attr.name == "is_conformant":
                    attr.value = "false"


def _find_events_by_type(log: OcelLog, run_id: str, event_type: str) -> list[OcelEvent]:
    """Find all events of a given type belonging to a run."""
    results = []
    for e in log.events:
        if e.type != event_type:
            continue
        for attr in e.attributes:
            if attr.name == "run_id" and attr.value == run_id:
                results.append(e)
                break
    return results


def _find_events_for_run(log: OcelLog, run_id: str) -> list[OcelEvent]:
    """Find all events belonging to a run, in order."""
    results = []
    for e in log.events:
        for attr in e.attributes:
            if attr.name == "run_id" and attr.value == run_id:
                results.append(e)
                break
    return results


class DeviationStrategy(ABC):
    """Base class for deviation injection strategies."""

    @property
    @abstractmethod
    def deviation_type(self) -> DeviationType: ...

    @abstractmethod
    def apply(
        self,
        log: OcelLog,
        run_id: str,
        template: WorkflowTemplate,
        rng: random.Random,
    ) -> DeviationSpec | None:
        """Apply the deviation to the log. Returns a spec if applied, None if not applicable."""


class SkippedActivityStrategy(DeviationStrategy):
    """Remove all events for a non-start, non-end step."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.SKIPPED_ACTIVITY

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        # Pick a skippable step (not start or end)
        candidates = [s for s in template.steps if not s.is_start and not s.is_end]
        if not candidates:
            # If only 2 steps, allow skipping non-start
            candidates = [s for s in template.steps if not s.is_start]
        if not candidates:
            return None

        step = rng.choice(candidates)
        step_id = step.id
        invocation_id = f"{run_id}-inv-{step_id}"

        # Remove events related to this invocation or step
        events_to_remove = set()
        for i, e in enumerate(log.events):
            # Check if event has step_id attribute matching
            for attr in e.attributes:
                if attr.name == "step_id" and attr.value == step_id:
                    for a2 in e.attributes:
                        if a2.name == "run_id" and a2.value == run_id:
                            events_to_remove.add(i)
                            break
            # Check if event references the invocation
            for rel in e.relationships:
                if rel.objectId == invocation_id:
                    events_to_remove.add(i)

        # Also remove message events to/from this step
        msg_id_prefix = f"{run_id}-msg-{step_id}-"
        msg_id_suffix = f"-to-{step_id}"
        for i, e in enumerate(log.events):
            for rel in e.relationships:
                if rel.objectId.startswith(msg_id_prefix) or rel.objectId.endswith(msg_id_suffix):
                    events_to_remove.add(i)

        if not events_to_remove:
            return None

        # Mark remaining run events near the gap as deviations (the run_completed)
        run_completed = _find_events_by_type(log, run_id, "run_completed")
        for e in run_completed:
            _mark_event_as_deviation(e, self.deviation_type)

        # Remove events (iterate in reverse to preserve indices)
        for i in sorted(events_to_remove, reverse=True):
            log.events.pop(i)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            step_id=step_id,
            description=f"Skipped step '{step_id}'",
        )


class InsertedActivityStrategy(DeviationStrategy):
    """Insert an unexpected agent invocation into the run."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.INSERTED_ACTIVITY

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        run_events = _find_events_for_run(log, run_id)
        if len(run_events) < 3:
            return None

        # Find a good insertion point (between agent_completed and next agent_invoked)
        insert_idx = None
        for i, e in enumerate(log.events):
            if e in run_events and e.type == "agent_completed":
                insert_idx = log.events.index(e) + 1
                break
        if insert_idx is None:
            return None

        ref_event = log.events[insert_idx - 1]
        base_time = ref_event.time + timedelta(milliseconds=rng.randint(50, 200))

        # Create a spurious extra step
        extra_inv_id = f"{run_id}-inv-extra-{rng.randint(1000,9999)}"
        extra_agent_id = "agent-reviewer"  # an agent not in the template

        # Ensure agent object exists
        log.add_object(OcelObject(
            id=extra_agent_id,
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value="reviewer", time=base_time),
                OcelObjectAttribute(
                    name="model_name",
                    value=LLMModel.GPT4O_MINI.value,
                    time=base_time,
                ),
            ],
        ))
        log.add_object(OcelObject(
            id=extra_inv_id,
            type="agent_invocation",
            attributes=[
                OcelObjectAttribute(name="status", value="completed", time=base_time),
                OcelObjectAttribute(name="input_tokens", value="200", time=base_time),
                OcelObjectAttribute(name="output_tokens", value="100", time=base_time),
                OcelObjectAttribute(name="cost_usd", value="0.001", time=base_time),
            ],
        ))

        # Find max sequence number for this run
        max_seq = 0
        for e in run_events:
            for attr in e.attributes:
                if attr.name == "sequence_number":
                    max_seq = max(max_seq, int(attr.value))

        def _dev_attrs(seq: int) -> list[OcelEventAttribute]:
            return [
                OcelEventAttribute(name="run_id", value=run_id),
                OcelEventAttribute(name="sequence_number", value=str(seq)),
                OcelEventAttribute(name="is_deviation", value="true"),
                OcelEventAttribute(name="deviation_type", value=self.deviation_type.value),
                OcelEventAttribute(name="step_id", value="extra"),
            ]

        evt_id_base = f"{run_id}-dev-ins-{rng.randint(1000,9999)}"
        invoked_evt = OcelEvent(
            id=f"{evt_id_base}-invoked",
            type="agent_invoked",
            time=base_time,
            attributes=_dev_attrs(max_seq + 1),
            relationships=[
                OcelRelationship(objectId=run_id, qualifier="part_of"),
                OcelRelationship(objectId=extra_agent_id, qualifier="invoked"),
                OcelRelationship(objectId=extra_inv_id, qualifier="started"),
            ],
        )
        completed_evt = OcelEvent(
            id=f"{evt_id_base}-completed",
            type="agent_completed",
            time=base_time + timedelta(milliseconds=rng.randint(100, 500)),
            attributes=_dev_attrs(max_seq + 2),
            relationships=[
                OcelRelationship(objectId=run_id, qualifier="part_of"),
                OcelRelationship(objectId=extra_inv_id, qualifier="completed"),
            ],
        )

        log.events.insert(insert_idx, invoked_evt)
        log.events.insert(insert_idx + 1, completed_evt)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description="Inserted unexpected agent invocation (reviewer)",
        )


class WrongResourceStrategy(DeviationStrategy):
    """Swap the agent that handles a step to a wrong one."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.WRONG_RESOURCE

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        invoked_events = _find_events_by_type(log, run_id, "agent_invoked")
        if not invoked_events:
            return None

        target_event = rng.choice(invoked_events)
        # Find which agent was invoked
        original_agent = None
        for rel in target_event.relationships:
            if rel.qualifier == "invoked":
                original_agent = rel.objectId
                break
        if not original_agent:
            return None

        # Pick a different agent role
        all_roles = list(AgentRole)
        original_role = original_agent.replace("agent-", "")
        other_roles = [r for r in all_roles if r.value != original_role]
        new_role = rng.choice(other_roles)
        new_agent_id = f"agent-{new_role.value}"

        # Ensure new agent object exists
        log.add_object(OcelObject(
            id=new_agent_id,
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value=new_role.value, time=target_event.time),
                OcelObjectAttribute(
                    name="model_name", value=LLMModel.GPT4O.value, time=target_event.time
                ),
            ],
        ))

        # Swap the relationship
        for rel in target_event.relationships:
            if rel.qualifier == "invoked":
                rel.objectId = new_agent_id
        _mark_event_as_deviation(target_event, self.deviation_type)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description=f"Step handled by '{new_role.value}' instead of '{original_role}'",
        )


class SwappedOrderStrategy(DeviationStrategy):
    """Swap the timestamps of two consecutive agent invocations."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.SWAPPED_ORDER

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        invoked_events = _find_events_by_type(log, run_id, "agent_invoked")
        if len(invoked_events) < 2:
            return None

        # Pick two consecutive invocations to swap
        idx = rng.randint(0, len(invoked_events) - 2)
        e1 = invoked_events[idx]
        e2 = invoked_events[idx + 1]

        # Swap their timestamps
        e1.time, e2.time = e2.time, e1.time

        # Also swap in the main events list to maintain position consistency
        i1 = log.events.index(e1)
        i2 = log.events.index(e2)
        log.events[i1], log.events[i2] = log.events[i2], log.events[i1]

        _mark_event_as_deviation(e1, self.deviation_type)
        _mark_event_as_deviation(e2, self.deviation_type)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description=f"Swapped order of steps at positions {idx} and {idx+1}",
        )


class WrongToolStrategy(DeviationStrategy):
    """Replace a tool call with a different, unexpected tool."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.WRONG_TOOL

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        tool_events = _find_events_by_type(log, run_id, "tool_called")
        if not tool_events:
            return None

        target = rng.choice(tool_events)
        # Find the tool_call object
        for rel in target.relationships:
            if rel.qualifier == "started":
                tool_obj_id = rel.objectId
                for obj in log.objects:
                    if obj.id == tool_obj_id:
                        # Change to a wrong tool
                        wrong_tool = rng.choice([
                            ToolKind.DATABASE_QUERY, ToolKind.API_CALL, ToolKind.VECTOR_SEARCH
                        ])
                        for attr in obj.attributes:
                            if attr.name == "tool_name":
                                attr.value = wrong_tool.value
                            if attr.name == "tool_kind":
                                attr.value = wrong_tool.value
                        break
                break

        _mark_event_as_deviation(target, self.deviation_type)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description="Agent used wrong tool",
        )


class RepeatedActivityStrategy(DeviationStrategy):
    """Duplicate an agent invocation (simulating a retry)."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.REPEATED_ACTIVITY

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        completed_events = _find_events_by_type(log, run_id, "agent_completed")
        if not completed_events:
            return None

        target = rng.choice(completed_events)
        target_idx = log.events.index(target)

        # Find the step_id
        step_id = ""
        for attr in target.attributes:
            if attr.name == "step_id":
                step_id = attr.value

        retry_inv_id = f"{run_id}-inv-{step_id}-retry"
        base_time = target.time + timedelta(milliseconds=rng.randint(50, 200))

        max_seq = 0
        for e in _find_events_for_run(log, run_id):
            for attr in e.attributes:
                if attr.name == "sequence_number":
                    max_seq = max(max_seq, int(attr.value))

        # Create retry_started event
        retry_evt = OcelEvent(
            id=f"{run_id}-dev-retry-{rng.randint(1000,9999)}",
            type="retry_started",
            time=base_time,
            attributes=[
                OcelEventAttribute(name="run_id", value=run_id),
                OcelEventAttribute(name="sequence_number", value=str(max_seq + 1)),
                OcelEventAttribute(name="is_deviation", value="true"),
                OcelEventAttribute(name="deviation_type", value=self.deviation_type.value),
            ],
            relationships=[
                OcelRelationship(objectId=run_id, qualifier="part_of"),
                OcelRelationship(objectId=retry_inv_id, qualifier="retried"),
            ],
        )
        log.events.insert(target_idx + 1, retry_evt)

        # Create the retry invocation object
        log.add_object(OcelObject(
            id=retry_inv_id,
            type="agent_invocation",
            attributes=[
                OcelObjectAttribute(name="status", value="completed", time=base_time),
                OcelObjectAttribute(name="input_tokens", value="300", time=base_time),
                OcelObjectAttribute(name="output_tokens", value="150", time=base_time),
                OcelObjectAttribute(name="cost_usd", value="0.002", time=base_time),
            ],
        ))

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            step_id=step_id,
            description=f"Step '{step_id}' was retried",
        )


class TimeoutStrategy(DeviationStrategy):
    """Make a step time out and fail."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.TIMEOUT

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        invoked_events = _find_events_by_type(log, run_id, "agent_invoked")
        if not invoked_events:
            return None

        target = rng.choice(invoked_events)
        target_idx = log.events.index(target)

        step_id = ""
        for attr in target.attributes:
            if attr.name == "step_id":
                step_id = attr.value

        # Find the invocation id
        invocation_id = ""
        for rel in target.relationships:
            if rel.qualifier == "started":
                invocation_id = rel.objectId
                break

        # Add an error_occurred event right after the invoked event
        base_time = target.time + timedelta(seconds=rng.randint(30, 120))

        max_seq = 0
        for e in _find_events_for_run(log, run_id):
            for attr in e.attributes:
                if attr.name == "sequence_number":
                    max_seq = max(max_seq, int(attr.value))

        error_evt = OcelEvent(
            id=f"{run_id}-dev-timeout-{rng.randint(1000,9999)}",
            type="error_occurred",
            time=base_time,
            attributes=[
                OcelEventAttribute(name="run_id", value=run_id),
                OcelEventAttribute(name="error_message", value="Step timed out after 60s"),
                OcelEventAttribute(name="sequence_number", value=str(max_seq + 1)),
                OcelEventAttribute(name="is_deviation", value="true"),
                OcelEventAttribute(name="deviation_type", value=self.deviation_type.value),
            ],
            relationships=[
                OcelRelationship(objectId=run_id, qualifier="part_of"),
                OcelRelationship(objectId=invocation_id, qualifier="failed") if invocation_id
                else OcelRelationship(objectId=run_id, qualifier="part_of"),
            ],
        )
        log.events.insert(target_idx + 1, error_evt)

        # Update invocation status to failed
        if invocation_id:
            for obj in log.objects:
                if obj.id == invocation_id:
                    for obj_attr in obj.attributes:
                        if obj_attr.name == "status":
                            obj_attr.value = "failed"

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            step_id=step_id,
            description=f"Step '{step_id}' timed out",
        )


class WrongRoutingStrategy(DeviationStrategy):
    """Insert a routing_decided event that selects the wrong agent."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.WRONG_ROUTING

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        invoked_events = _find_events_by_type(log, run_id, "agent_invoked")
        if not invoked_events:
            return None

        target = rng.choice(invoked_events)
        target_idx = log.events.index(target)

        # Pick a wrong agent for the routing decision
        all_roles = list(AgentRole)
        wrong_role = rng.choice(all_roles)
        wrong_agent_id = f"agent-{wrong_role.value}"

        log.add_object(OcelObject(
            id=wrong_agent_id,
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value=wrong_role.value, time=target.time),
                OcelObjectAttribute(
                    name="model_name", value=LLMModel.GPT4O.value, time=target.time
                ),
            ],
        ))

        max_seq = 0
        for e in _find_events_for_run(log, run_id):
            for attr in e.attributes:
                if attr.name == "sequence_number":
                    max_seq = max(max_seq, int(attr.value))

        routing_evt = OcelEvent(
            id=f"{run_id}-dev-routing-{rng.randint(1000,9999)}",
            type="routing_decided",
            time=target.time - timedelta(milliseconds=rng.randint(10, 50)),
            attributes=[
                OcelEventAttribute(name="run_id", value=run_id),
                OcelEventAttribute(name="sequence_number", value=str(max_seq + 1)),
                OcelEventAttribute(name="is_deviation", value="true"),
                OcelEventAttribute(name="deviation_type", value=self.deviation_type.value),
            ],
            relationships=[
                OcelRelationship(objectId=run_id, qualifier="part_of"),
                OcelRelationship(objectId=wrong_agent_id, qualifier="selected"),
            ],
        )
        log.events.insert(target_idx, routing_evt)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description=f"Wrong routing decision: selected '{wrong_role.value}'",
        )


class MissingHandoffStrategy(DeviationStrategy):
    """Remove an inter-agent message_sent event."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.MISSING_HANDOFF

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        msg_events = _find_events_by_type(log, run_id, "message_sent")
        if not msg_events:
            return None

        target = rng.choice(msg_events)
        target_idx = log.events.index(target)

        # Mark the surrounding events
        if target_idx > 0:
            _mark_event_as_deviation(log.events[target_idx - 1], self.deviation_type)
        if target_idx < len(log.events) - 1:
            _mark_event_as_deviation(log.events[target_idx + 1], self.deviation_type)

        # Remove the message event
        log.events.pop(target_idx)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description="Inter-agent handoff message removed",
        )


class ExtraLLMCallStrategy(DeviationStrategy):
    """Insert an unnecessary extra LLM call into a step."""

    @property
    def deviation_type(self) -> DeviationType:
        return DeviationType.EXTRA_LLM_CALL

    def apply(
        self, log: OcelLog, run_id: str, template: WorkflowTemplate, rng: random.Random
    ) -> DeviationSpec | None:
        llm_events = _find_events_by_type(log, run_id, "llm_response_received")
        if not llm_events:
            return None

        target = rng.choice(llm_events)
        target_idx = log.events.index(target)

        # Find the invocation this belongs to
        invocation_id = ""
        for rel in target.relationships:
            if rel.qualifier == "triggered_by":
                invocation_id = rel.objectId
                break

        base_time = target.time + timedelta(milliseconds=rng.randint(20, 100))
        llm_attrs = generate_llm_attributes(rng, LLMModel.GPT4O_MINI)

        extra_llm_id = f"{run_id}-dev-llm-{rng.randint(1000,9999)}"
        log.add_object(OcelObject(
            id=extra_llm_id,
            type="llm_call",
            attributes=[
                OcelObjectAttribute(name="model", value=llm_attrs.model, time=base_time),
                OcelObjectAttribute(
                    name="input_tokens", value=str(llm_attrs.input_tokens), time=base_time
                ),
                OcelObjectAttribute(
                    name="output_tokens", value=str(llm_attrs.output_tokens), time=base_time
                ),
                OcelObjectAttribute(
                    name="latency_ms", value=str(llm_attrs.latency_ms), time=base_time
                ),
            ],
        ))

        max_seq = 0
        for e in _find_events_for_run(log, run_id):
            for attr in e.attributes:
                if attr.name == "sequence_number":
                    max_seq = max(max_seq, int(attr.value))

        def _dev_attrs(seq: int) -> list[OcelEventAttribute]:
            return [
                OcelEventAttribute(name="run_id", value=run_id),
                OcelEventAttribute(name="sequence_number", value=str(seq)),
                OcelEventAttribute(name="is_deviation", value="true"),
                OcelEventAttribute(name="deviation_type", value=self.deviation_type.value),
            ]

        req_evt = OcelEvent(
            id=f"{extra_llm_id}-req",
            type="llm_request_sent",
            time=base_time,
            attributes=_dev_attrs(max_seq + 1),
            relationships=[
                OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                OcelRelationship(objectId=extra_llm_id, qualifier="started"),
            ],
        )
        resp_evt = OcelEvent(
            id=f"{extra_llm_id}-resp",
            type="llm_response_received",
            time=base_time + timedelta(milliseconds=llm_attrs.latency_ms),
            attributes=_dev_attrs(max_seq + 2),
            relationships=[
                OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                OcelRelationship(objectId=extra_llm_id, qualifier="completed"),
            ],
        )

        log.events.insert(target_idx + 1, req_evt)
        log.events.insert(target_idx + 2, resp_evt)

        _mark_run_nonconformant(log, run_id)

        return DeviationSpec(
            run_id=run_id,
            deviation_type=self.deviation_type,
            description="Extra unnecessary LLM call inserted",
        )
