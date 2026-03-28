"""Walk an OCEL trace, call the LLM per agent step, and patch attributes."""

from __future__ import annotations

import json

from rich.progress import Progress, TaskID

from ocelgen.enrichment.client import EnrichmentResponse, LLMClient
from ocelgen.enrichment.prompts import build_enrichment_prompt
from ocelgen.models.ocel import OcelLog, OcelObjectAttribute
from ocelgen.scenarios.domain import DomainScenario


def _extract_steps_from_log(log: OcelLog, run_id: str) -> list[dict]:
    """Extract the ordered list of agent steps for a given run.

    Each step is a dict with:
      - agent_role: str
      - invocation_id: str
      - llm_call_ids: list[str]
      - tool_call_ids: list[str]
      - message_id: str | None
      - expected_llm_calls: int
      - expected_tool_calls: int
    """
    invoked_events = []
    for e in log.events:
        if e.type != "agent_invoked":
            continue
        is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
        if is_this_run:
            invoked_events.append(e)

    steps = []
    for evt in invoked_events:
        agent_id = ""
        invocation_id = ""
        for rel in evt.relationships:
            if rel.qualifier == "invoked":
                agent_id = rel.objectId
            if rel.qualifier == "started":
                invocation_id = rel.objectId

        agent_role = agent_id.replace("agent-", "") if agent_id else ""

        llm_call_ids = []
        tool_call_ids = []
        message_id = None

        for e in log.events:
            is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
            if not is_this_run:
                continue
            for rel in e.relationships:
                if rel.qualifier == "triggered_by" and rel.objectId == invocation_id:
                    for rel2 in e.relationships:
                        if rel2.qualifier == "started":
                            obj_id = rel2.objectId
                            for obj in log.objects:
                                if obj.id == obj_id:
                                    if obj.type == "llm_call":
                                        llm_call_ids.append(obj_id)
                                    elif obj.type == "tool_call":
                                        tool_call_ids.append(obj_id)

        for e in log.events:
            if e.type != "message_sent":
                continue
            is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
            if not is_this_run:
                continue
            for rel in e.relationships:
                if rel.qualifier == "sender" and rel.objectId == agent_id:
                    for rel2 in e.relationships:
                        if rel2.qualifier == "sent":
                            message_id = rel2.objectId
                            break

        steps.append({
            "agent_role": agent_role,
            "invocation_id": invocation_id,
            "llm_call_ids": llm_call_ids,
            "tool_call_ids": tool_call_ids,
            "message_id": message_id,
            "expected_llm_calls": len(llm_call_ids),
            "expected_tool_calls": len(tool_call_ids),
        })

    return steps


def _get_object(log: OcelLog, obj_id: str):
    for obj in log.objects:
        if obj.id == obj_id:
            return obj
    return None


def _patch_attribute(obj, name: str, value: str) -> None:
    for attr in obj.attributes:
        if attr.name == name:
            attr.value = value
            return
    ts = obj.attributes[0].time if obj.attributes else None
    if ts:
        obj.attributes.append(OcelObjectAttribute(name=name, value=value, time=ts))


def _get_tool_names_for_step(log: OcelLog, step: dict) -> list[str]:
    names = []
    for tool_id in step["tool_call_ids"]:
        obj = _get_object(log, tool_id)
        if obj:
            for attr in obj.attributes:
                if attr.name == "tool_name":
                    names.append(attr.value)
    return names


def enrich_log(
    log: OcelLog,
    scenario: DomainScenario,
    client: LLMClient | None = None,
    progress: Progress | None = None,
    progress_task: TaskID | None = None,
) -> None:
    """Enrich an OcelLog in-place with LLM-generated content."""
    if client is None:
        client = LLMClient()

    run_ids = sorted({o.id for o in log.objects if o.type == "run"})

    pattern_desc = ""
    for obj in log.objects:
        if obj.type == "run":
            for attr in obj.attributes:
                if attr.name == "pattern_type":
                    pattern_desc = attr.value
            break

    for run_idx, run_id in enumerate(run_ids):
        user_query = scenario.query_for_run(run_idx)

        run_obj = _get_object(log, run_id)
        if run_obj:
            _patch_attribute(run_obj, "user_query", user_query)

        task_obj = _get_object(log, f"{run_id}-task")
        if task_obj:
            _patch_attribute(task_obj, "description", user_query)

        steps = _extract_steps_from_log(log, run_id)
        previous_output: str | None = None

        for step in steps:
            role = step["agent_role"]
            tool_names = _get_tool_names_for_step(log, step)
            persona = scenario.agent_personas.get(role, f"You are a {role} agent")

            system_prompt, user_prompt = build_enrichment_prompt(
                domain_description=scenario.description,
                pattern_description=pattern_desc,
                agent_role=role,
                agent_persona=persona,
                user_query=user_query,
                tool_names=tool_names,
                tool_descriptions=scenario.tool_descriptions,
                expected_llm_calls=step["expected_llm_calls"],
                expected_tool_calls=step["expected_tool_calls"],
                previous_output=previous_output,
            )

            try:
                raw = client.generate(system_prompt, user_prompt)
                resp = EnrichmentResponse.from_dict(
                    raw,
                    expected_llm_calls=step["expected_llm_calls"],
                    expected_tool_calls=step["expected_tool_calls"],
                )
            except Exception:
                continue

            inv_obj = _get_object(log, step["invocation_id"])
            if inv_obj:
                _patch_attribute(inv_obj, "reasoning", resp.reasoning)

            for i, llm_id in enumerate(step["llm_call_ids"]):
                llm_obj = _get_object(log, llm_id)
                if llm_obj and i < len(resp.llm_calls):
                    _patch_attribute(llm_obj, "prompt", resp.llm_calls[i].get("prompt", ""))
                    _patch_attribute(llm_obj, "completion", resp.llm_calls[i].get("completion", ""))

            for i, tool_id in enumerate(step["tool_call_ids"]):
                tool_obj = _get_object(log, tool_id)
                if tool_obj and i < len(resp.tool_calls):
                    _patch_attribute(tool_obj, "tool_input", json.dumps(resp.tool_calls[i].get("input", {})))
                    _patch_attribute(tool_obj, "tool_output", json.dumps(resp.tool_calls[i].get("output", {})))

            if step["message_id"]:
                msg_obj = _get_object(log, step["message_id"])
                if msg_obj:
                    _patch_attribute(msg_obj, "content", resp.output_to_next_agent)

            previous_output = resp.output_to_next_agent

        if progress and progress_task is not None:
            progress.advance(progress_task)
