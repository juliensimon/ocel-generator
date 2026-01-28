"""Simulate a single conformant run from a workflow template.

The simulator walks the workflow template in topological order, generating
OCEL events and objects for each step. Each step produces:
  - agent_invoked / agent_completed events
  - llm_request_sent / llm_response_received events (per LLM call)
  - tool_called / tool_returned events (per tool call)
  - message_sent events (for inter-agent handoffs)
  - Corresponding OCEL objects (agent_invocation, llm_call, tool_call, message)
"""

from __future__ import annotations

import random
from datetime import datetime

from faker import Faker

from ocelgen.generation.attributes import (
    generate_llm_attributes,
    generate_tool_attributes,
)
from ocelgen.generation.timestamp import TimestampGenerator
from ocelgen.models.langchain import AgentRole
from ocelgen.models.ocel import (
    OcelAttributeDefinition,
    OcelEvent,
    OcelEventAttribute,
    OcelLog,
    OcelObject,
    OcelObjectAttribute,
    OcelRelationship,
)
from ocelgen.models.workflow import WorkflowStep, WorkflowTemplate

# All event-level attribute definitions shared across runs
EVENT_ATTR_DEFS: dict[str, list[OcelAttributeDefinition]] = {
    "run_started": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "agent_invoked": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="step_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "llm_request_sent": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "llm_response_received": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "tool_called": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "tool_returned": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "message_sent": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "routing_decided": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "agent_completed": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="step_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "error_occurred": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="error_message", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "retry_started": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
    "run_completed": [
        OcelAttributeDefinition(name="run_id", type="string"),
        OcelAttributeDefinition(name="sequence_number", type="integer"),
        OcelAttributeDefinition(name="is_deviation", type="boolean"),
        OcelAttributeDefinition(name="deviation_type", type="string"),
    ],
}

# Object type attribute definitions
OBJECT_ATTR_DEFS: dict[str, list[OcelAttributeDefinition]] = {
    "run": [
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="pattern_type", type="string"),
        OcelAttributeDefinition(name="is_conformant", type="boolean"),
        OcelAttributeDefinition(name="user_query", type="string"),
    ],
    "agent": [
        OcelAttributeDefinition(name="role", type="string"),
        OcelAttributeDefinition(name="model_name", type="string"),
    ],
    "agent_invocation": [
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="input_tokens", type="integer"),
        OcelAttributeDefinition(name="output_tokens", type="integer"),
        OcelAttributeDefinition(name="cost_usd", type="float"),
    ],
    "tool_call": [
        OcelAttributeDefinition(name="tool_name", type="string"),
        OcelAttributeDefinition(name="tool_kind", type="string"),
        OcelAttributeDefinition(name="status", type="string"),
        OcelAttributeDefinition(name="duration_ms", type="integer"),
    ],
    "llm_call": [
        OcelAttributeDefinition(name="model", type="string"),
        OcelAttributeDefinition(name="input_tokens", type="integer"),
        OcelAttributeDefinition(name="output_tokens", type="integer"),
        OcelAttributeDefinition(name="latency_ms", type="integer"),
    ],
    "message": [
        OcelAttributeDefinition(name="role", type="string"),
        OcelAttributeDefinition(name="content_length", type="integer"),
    ],
    "task": [
        OcelAttributeDefinition(name="description", type="string"),
        OcelAttributeDefinition(name="status", type="string"),
    ],
}


def _base_event_attrs(
    run_id: str, seq: int, is_deviation: bool = False, deviation_type: str = ""
) -> list[OcelEventAttribute]:
    """Common attributes present on every event."""
    return [
        OcelEventAttribute(name="run_id", value=run_id),
        OcelEventAttribute(name="sequence_number", value=str(seq)),
        OcelEventAttribute(name="is_deviation", value=str(is_deviation).lower()),
        OcelEventAttribute(name="deviation_type", value=deviation_type),
    ]


class RunSimulator:
    """Simulates a single conformant run from a workflow template."""

    def __init__(
        self,
        template: WorkflowTemplate,
        run_index: int,
        rng: random.Random,
        faker: Faker,
        base_time: datetime | None = None,
    ) -> None:
        self.template = template
        self.run_id = f"run-{run_index:04d}"
        self.rng = rng
        self.faker = faker
        self.ts = TimestampGenerator(rng, base_time)
        self._seq = 0
        self._event_counter = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"{self.run_id}-evt-{self._event_counter:04d}"

    def simulate(self) -> OcelLog:
        """Generate a complete conformant run as an OcelLog."""
        log = OcelLog()

        # Register all type schemas
        for evt_type, attrs in EVENT_ATTR_DEFS.items():
            log.add_event_type(evt_type, attrs)
        for obj_type, attrs in OBJECT_ATTR_DEFS.items():
            log.add_object_type(obj_type, attrs)

        creation_time = self.ts.current
        user_query = self.faker.sentence(nb_words=8)

        # Create run object
        run_obj = OcelObject(
            id=self.run_id,
            type="run",
            attributes=[
                OcelObjectAttribute(name="status", value="completed", time=creation_time),
                OcelObjectAttribute(
                    name="pattern_type", value=self.template.name, time=creation_time
                ),
                OcelObjectAttribute(name="is_conformant", value="true", time=creation_time),
                OcelObjectAttribute(name="user_query", value=user_query, time=creation_time),
            ],
        )
        log.add_object(run_obj)

        # Create task object
        task_id = f"{self.run_id}-task"
        task_obj = OcelObject(
            id=task_id,
            type="task",
            attributes=[
                OcelObjectAttribute(name="description", value=user_query, time=creation_time),
                OcelObjectAttribute(name="status", value="completed", time=creation_time),
            ],
        )
        log.add_object(task_obj)

        # run_started event
        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="run_started",
            time=self.ts.advance(mean_ms=10),
            attributes=_base_event_attrs(self.run_id, self._next_seq()),
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="started"),
                OcelRelationship(objectId=task_id, qualifier="root_task"),
            ],
        ))

        # Walk steps, handling parallel groups and supervisor routing
        ordered_steps = self.template.topological_order()
        prev_step: WorkflowStep | None = None
        i = 0
        while i < len(ordered_steps):
            step = ordered_steps[i]

            # Check if this step is part of a parallel group
            if step.parallel_group:
                # Collect all steps in this parallel group
                group_steps = []
                group = step.parallel_group
                while i < len(ordered_steps) and ordered_steps[i].parallel_group == group:
                    group_steps.append(ordered_steps[i])
                    i += 1
                self._simulate_parallel_group(log, group_steps, prev_step)
                prev_step = group_steps[-1]
                continue

            # Check if this step is a supervisor routing to workers
            successors = self.template.successors(step.id)
            is_supervisor_step = (
                step.agent_role == AgentRole.SUPERVISOR
                and len(successors) > 1
            )

            self._simulate_step(log, step, prev_step)

            if is_supervisor_step:
                # Emit routing_decided events for each worker
                for succ in successors:
                    self._simulate_routing(log, step, succ)

            prev_step = step
            i += 1

        # run_completed event
        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="run_completed",
            time=self.ts.advance(mean_ms=10),
            attributes=_base_event_attrs(self.run_id, self._next_seq()),
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="completed"),
            ],
        ))

        return log

    def _simulate_step(
        self,
        log: OcelLog,
        step: WorkflowStep,
        prev_step: WorkflowStep | None,
    ) -> None:
        """Simulate a single workflow step, generating all its events and objects."""
        step_time = self.ts.current

        # Create agent object (shared across runs — same id for same role)
        agent_id = f"agent-{step.agent_role.value}"
        agent_obj = OcelObject(
            id=agent_id,
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value=step.agent_role.value, time=step_time),
                OcelObjectAttribute(name="model_name", value=step.model.value, time=step_time),
            ],
        )
        log.add_object(agent_obj)

        # Create agent_invocation object
        invocation_id = f"{self.run_id}-inv-{step.id}"
        total_input = 0
        total_output = 0
        total_cost = 0.0

        # agent_invoked event
        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="agent_invoked",
            time=self.ts.advance(mean_ms=50),
            attributes=_base_event_attrs(self.run_id, self._next_seq()) + [
                OcelEventAttribute(name="step_id", value=step.id),
            ],
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="part_of"),
                OcelRelationship(objectId=agent_id, qualifier="invoked"),
                OcelRelationship(objectId=invocation_id, qualifier="started"),
            ],
        ))

        # Generate inter-agent message if there was a previous step
        if prev_step is not None:
            self._simulate_message(log, prev_step, step)

        # LLM calls
        for i in range(step.expected_llm_calls):
            attrs = generate_llm_attributes(self.rng, step.model)
            total_input += attrs.input_tokens
            total_output += attrs.output_tokens
            total_cost += attrs.cost_usd

            llm_call_id = f"{invocation_id}-llm-{i}"
            llm_obj = OcelObject(
                id=llm_call_id,
                type="llm_call",
                attributes=[
                    OcelObjectAttribute(name="model", value=attrs.model, time=self.ts.current),
                    OcelObjectAttribute(
                        name="input_tokens", value=str(attrs.input_tokens), time=self.ts.current
                    ),
                    OcelObjectAttribute(
                        name="output_tokens", value=str(attrs.output_tokens), time=self.ts.current
                    ),
                    OcelObjectAttribute(
                        name="latency_ms", value=str(attrs.latency_ms), time=self.ts.current
                    ),
                ],
            )
            log.add_object(llm_obj)

            # llm_request_sent
            log.add_event(OcelEvent(
                id=self._next_event_id(),
                type="llm_request_sent",
                time=self.ts.advance(mean_ms=20),
                attributes=_base_event_attrs(self.run_id, self._next_seq()),
                relationships=[
                    OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                    OcelRelationship(objectId=llm_call_id, qualifier="started"),
                ],
            ))

            # llm_response_received
            log.add_event(OcelEvent(
                id=self._next_event_id(),
                type="llm_response_received",
                time=self.ts.advance(mean_ms=float(attrs.latency_ms)),
                attributes=_base_event_attrs(self.run_id, self._next_seq()),
                relationships=[
                    OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                    OcelRelationship(objectId=llm_call_id, qualifier="completed"),
                ],
            ))

        # Tool calls
        for i in range(step.expected_tool_calls):
            tool = self.rng.choice(step.tools) if step.tools else None
            if tool is None:
                continue
            tool_attrs = generate_tool_attributes(self.rng, tool.value, tool.value)
            tool_call_id = f"{invocation_id}-tool-{i}"

            tool_obj = OcelObject(
                id=tool_call_id,
                type="tool_call",
                attributes=[
                    OcelObjectAttribute(
                        name="tool_name", value=tool_attrs.tool_name, time=self.ts.current
                    ),
                    OcelObjectAttribute(
                        name="tool_kind", value=tool_attrs.tool_kind, time=self.ts.current
                    ),
                    OcelObjectAttribute(
                        name="status", value=tool_attrs.status, time=self.ts.current
                    ),
                    OcelObjectAttribute(
                        name="duration_ms", value=str(tool_attrs.duration_ms), time=self.ts.current
                    ),
                ],
            )
            log.add_object(tool_obj)

            # tool_called
            log.add_event(OcelEvent(
                id=self._next_event_id(),
                type="tool_called",
                time=self.ts.advance(mean_ms=20),
                attributes=_base_event_attrs(self.run_id, self._next_seq()),
                relationships=[
                    OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                    OcelRelationship(objectId=tool_call_id, qualifier="started"),
                ],
            ))

            # tool_returned
            log.add_event(OcelEvent(
                id=self._next_event_id(),
                type="tool_returned",
                time=self.ts.advance(mean_ms=float(tool_attrs.duration_ms)),
                attributes=_base_event_attrs(self.run_id, self._next_seq()),
                relationships=[
                    OcelRelationship(objectId=invocation_id, qualifier="triggered_by"),
                    OcelRelationship(objectId=tool_call_id, qualifier="completed"),
                ],
            ))

        # Create agent_invocation object with accumulated stats
        inv_obj = OcelObject(
            id=invocation_id,
            type="agent_invocation",
            attributes=[
                OcelObjectAttribute(name="status", value="completed", time=self.ts.current),
                OcelObjectAttribute(
                    name="input_tokens", value=str(total_input), time=self.ts.current
                ),
                OcelObjectAttribute(
                    name="output_tokens", value=str(total_output), time=self.ts.current
                ),
                OcelObjectAttribute(
                    name="cost_usd", value=str(round(total_cost, 6)), time=self.ts.current
                ),
            ],
        )
        log.add_object(inv_obj)

        # agent_completed event
        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="agent_completed",
            time=self.ts.advance(mean_ms=20),
            attributes=_base_event_attrs(self.run_id, self._next_seq()) + [
                OcelEventAttribute(name="step_id", value=step.id),
            ],
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="part_of"),
                OcelRelationship(objectId=invocation_id, qualifier="completed"),
            ],
        ))

    def _simulate_parallel_group(
        self,
        log: OcelLog,
        steps: list[WorkflowStep],
        prev_step: WorkflowStep | None,
    ) -> None:
        """Simulate steps that execute concurrently with overlapping timestamps.

        Each parallel worker gets its own TimestampGenerator forked from the
        current time, so their events interleave temporally.
        """
        from ocelgen.generation.timestamp import TimestampGenerator

        fork_time = self.ts.current
        all_events_with_time: list[tuple[datetime, OcelEvent]] = []

        for step in steps:
            # Fork a separate timestamp generator for each parallel worker
            worker_ts = TimestampGenerator(
                random.Random(self.rng.randint(0, 2**31)),
                base_time=fork_time,
            )
            saved_ts = self.ts
            self.ts = worker_ts

            # Simulate the step (events added to log directly)
            events_before = len(log.events)
            self._simulate_step(log, step, prev_step)
            new_events = log.events[events_before:]

            # Collect for sorting
            for evt in new_events:
                all_events_with_time.append((evt.time, evt))

            self.ts = saved_ts

        # Advance main clock past all parallel work
        latest = max(t for t, _ in all_events_with_time)
        if latest > self.ts.current:
            diff_ms = (latest - self.ts.current).total_seconds() * 1000
            self.ts.advance_fixed(diff_ms + self.rng.uniform(10, 50))

    def _simulate_routing(
        self,
        log: OcelLog,
        supervisor_step: WorkflowStep,
        target_step: WorkflowStep,
    ) -> None:
        """Emit a routing_decided event from supervisor to target worker."""
        target_agent_id = f"agent-{target_step.agent_role.value}"
        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="routing_decided",
            time=self.ts.advance(mean_ms=20),
            attributes=_base_event_attrs(self.run_id, self._next_seq()),
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="part_of"),
                OcelRelationship(objectId=target_agent_id, qualifier="selected"),
            ],
        ))

    def _simulate_message(
        self,
        log: OcelLog,
        sender_step: WorkflowStep,
        receiver_step: WorkflowStep,
    ) -> None:
        """Generate a message_sent event between two agents."""
        msg_id = f"{self.run_id}-msg-{sender_step.id}-to-{receiver_step.id}"
        content_length = self.rng.randint(100, 2000)
        msg_obj = OcelObject(
            id=msg_id,
            type="message",
            attributes=[
                OcelObjectAttribute(name="role", value="handoff", time=self.ts.current),
                OcelObjectAttribute(
                    name="content_length", value=str(content_length), time=self.ts.current
                ),
            ],
        )
        log.add_object(msg_obj)

        sender_agent_id = f"agent-{sender_step.agent_role.value}"
        receiver_agent_id = f"agent-{receiver_step.agent_role.value}"

        log.add_event(OcelEvent(
            id=self._next_event_id(),
            type="message_sent",
            time=self.ts.advance(mean_ms=30),
            attributes=_base_event_attrs(self.run_id, self._next_seq()),
            relationships=[
                OcelRelationship(objectId=self.run_id, qualifier="part_of"),
                OcelRelationship(objectId=msg_id, qualifier="sent"),
                OcelRelationship(objectId=sender_agent_id, qualifier="sender"),
                OcelRelationship(objectId=receiver_agent_id, qualifier="receiver"),
            ],
        ))
