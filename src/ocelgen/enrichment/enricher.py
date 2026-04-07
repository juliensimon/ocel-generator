"""Walk an OCEL trace, call the LLM per agent step, and patch attributes."""

from __future__ import annotations

import json
import logging
import random
from datetime import timedelta

from rich.progress import Progress, TaskID

from ocelgen.enrichment.client import EnrichmentResponse, LLMClient
from ocelgen.enrichment.prompts import build_enrichment_prompt
from ocelgen.models.ocel import OcelLog, OcelObjectAttribute
from ocelgen.scenarios.domain import DomainScenario

logger = logging.getLogger(__name__)

# Cost per 1K tokens (input, output) by model — mirrors generation/attributes.py
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3.5-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using word-to-token approximation."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def _recalculate_metrics(log: OcelLog, step: dict, resp: EnrichmentResponse) -> None:
    """Recalculate token counts, latency, and cost based on actual enriched content."""
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for i, llm_id in enumerate(step["llm_call_ids"]):
        llm_obj = _get_object(log, llm_id)
        if not llm_obj or i >= len(resp.llm_calls):
            continue

        prompt_text = resp.llm_calls[i].get("prompt", "")
        completion_text = resp.llm_calls[i].get("completion", "")
        input_tokens = _estimate_tokens(prompt_text)
        output_tokens = _estimate_tokens(completion_text)

        # Recalculate latency correlated with output tokens + jitter
        rng = random.Random(hash(llm_id))
        latency_ms = int(output_tokens * 2.5 + rng.uniform(-50, 50))
        latency_ms = max(100, latency_ms)

        _patch_attribute(llm_obj, "input_tokens", str(input_tokens))
        _patch_attribute(llm_obj, "output_tokens", str(output_tokens))
        _patch_attribute(llm_obj, "latency_ms", str(latency_ms))

        # Calculate cost based on model
        model_name = ""
        for attr in llm_obj.attributes:
            if attr.name == "model":
                model_name = attr.value
                break
        input_rate, output_rate = _COST_PER_1K.get(model_name, (0.003, 0.015))
        call_cost = (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate

        total_input += input_tokens
        total_output += output_tokens
        total_cost += call_cost

    # Update cost on agent_invocation object
    inv_obj = _get_object(log, step["invocation_id"])
    if inv_obj:
        _patch_attribute(inv_obj, "input_tokens", str(total_input))
        _patch_attribute(inv_obj, "output_tokens", str(total_output))
        _patch_attribute(inv_obj, "cost_usd", str(round(total_cost, 6)))


def _detect_run_deviations(log: OcelLog, run_id: str) -> list[dict]:
    """Scan events for a run and return a list of deviation descriptors."""
    deviations = []
    for event in log.events:
        is_this_run = any(a.name == "run_id" and a.value == run_id for a in event.attributes)
        if not is_this_run:
            continue
        is_dev = any(a.name == "is_deviation" and a.value == "true" for a in event.attributes)
        if not is_dev:
            continue
        dev_type = ""
        step_id = ""
        for a in event.attributes:
            if a.name == "deviation_type":
                dev_type = a.value
            if a.name == "step_id":
                step_id = a.value
        deviations.append(
            {
                "type": dev_type or event.type,
                "step_id": step_id,
                "event_type": event.type,
            }
        )
    return deviations


def _build_deviation_context(deviations: list[dict], current_step_id: str) -> str | None:
    """Build a deviation context string for the prompt if the current step has deviations."""
    # Find deviations relevant to this step (matching step_id or without step_id)
    relevant = [d for d in deviations if d["step_id"] == current_step_id or not d["step_id"]]
    if not relevant:
        return None

    lines = ["IMPORTANT: This run has deviations from the normal workflow:"]
    for d in relevant:
        dev_type = d["type"]
        if "wrong_tool" in dev_type:
            lines.append(f"- This step used the WRONG TOOL ({dev_type})")
        elif "skip" in dev_type:
            lines.append(f"- A previous step was SKIPPED ({dev_type})")
        elif "extra" in dev_type:
            lines.append(f"- An EXTRA step was added ({dev_type})")
        elif "reorder" in dev_type:
            lines.append(f"- Steps were REORDERED ({dev_type})")
        else:
            lines.append(f"- Deviation detected: {dev_type} (event: {d['event_type']})")
    lines.append(
        "Generate content that reflects these anomalies. Show confusion, errors, or inappropriate behavior."
    )
    return "\n".join(lines)


# Timing ranges per event type (min_s, max_s)
_TIMESTAMP_RANGES: dict[str, tuple[float, float]] = {
    "run_started": (0.0, 0.0),  # base
    "agent_invoked": (0.1, 0.5),
    "llm_request_sent": (0.05, 0.2),
    "llm_response_received": (1.0, 5.0),
    "tool_called": (0.05, 0.2),
    "tool_returned": (0.5, 3.0),
    "message_sent": (0.1, 0.3),
    "routing_decided": (0.1, 0.5),
    "agent_completed": (0.05, 0.2),
    "run_completed": (0.1, 0.5),
}


def _rewrite_timestamps(log: OcelLog, run_id: str) -> None:
    """Rewrite event timestamps for a run to use realistic durations."""
    rng = random.Random(hash(run_id))

    # Collect events for this run in their current order
    run_events = []
    for e in log.events:
        is_this_run = any(a.name == "run_id" and a.value == run_id for a in e.attributes)
        if is_this_run:
            run_events.append(e)

    if not run_events:
        return

    # Sort by current time to maintain ordering
    run_events.sort(key=lambda e: e.time)

    base_time = run_events[0].time

    # Build a map of llm_call objects for token-correlated latency
    llm_output_tokens: dict[str, int] = {}
    for obj in log.objects:
        if obj.type == "llm_call":
            for attr in obj.attributes:
                if attr.name == "output_tokens":
                    try:
                        llm_output_tokens[obj.id] = int(attr.value)
                    except (ValueError, TypeError):
                        pass

    current_time = base_time
    for event in run_events:
        lo, hi = _TIMESTAMP_RANGES.get(event.type, (0.05, 0.3))

        if event.type == "llm_response_received":
            # Correlate with output token count if available
            for rel in event.relationships:
                if rel.qualifier == "completed" and rel.objectId in llm_output_tokens:
                    tokens = llm_output_tokens[rel.objectId]
                    # tokens -> seconds: ~20-50 tokens/sec
                    lo = max(lo, tokens / 50.0)
                    hi = max(hi, tokens / 20.0)
                    break

        delta_s = rng.uniform(lo, hi)
        current_time = current_time + timedelta(seconds=delta_s)
        event.time = current_time


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

        steps.append(
            {
                "agent_role": agent_role,
                "invocation_id": invocation_id,
                "llm_call_ids": llm_call_ids,
                "tool_call_ids": tool_call_ids,
                "message_id": message_id,
                "expected_llm_calls": len(llm_call_ids),
                "expected_tool_calls": len(tool_call_ids),
            }
        )

    return steps


def _get_object(log: OcelLog, obj_id: str):
    for obj in log.objects:
        if obj.id == obj_id:
            return obj
    return None


def _patch_attribute(obj, name: str, value) -> None:  # type: ignore[no-untyped-def]
    """Set an attribute on an object, updating existing or appending new."""
    # Coerce non-string values (LLM may return dicts/lists)
    if not isinstance(value, str):
        value = json.dumps(value) if value is not None else ""
    for attr in obj.attributes:
        if attr.name == name:
            attr.value = value
            return
    # Append new attribute — need a timestamp from an existing attribute
    if not obj.attributes:
        logger.debug(
            "Cannot add attribute '%s' to object '%s': no existing attributes for timestamp",
            name,
            obj.id,
        )
        return
    ts = obj.attributes[0].time
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


def _detect_parallel_groups(steps: list[dict]) -> dict[int, list[int]]:
    """Detect parallel step groups by finding steps that share the same predecessors.

    Returns a mapping from aggregator step index -> list of parallel worker indices.
    This uses a heuristic: if multiple consecutive steps share no dependency chain
    with each other, they are parallel. The next step after the group is the aggregator.
    """
    # Build a simple adjacency model from invocation IDs
    # Heuristic: steps that share a prefix pattern like "run-XXXX-inv-stepN"
    # where the step ID suffix indicates parallel group membership.
    # We look for invocation IDs that contain "parallel" or a shared group marker.
    parallel_groups: dict[int, list[int]] = {}

    # Detect by looking for invocation IDs with parallel group markers
    i = 0
    while i < len(steps):
        # Check if this step's invocation_id contains a parallel group marker
        inv_id = steps[i]["invocation_id"]
        if "-parallel-" in inv_id or "-worker-" in inv_id:
            # Find all consecutive parallel workers
            group_indices = []
            while i < len(steps):
                iid = steps[i]["invocation_id"]
                if "-parallel-" in iid or "-worker-" in iid:
                    group_indices.append(i)
                    i += 1
                else:
                    break
            # The next step after the parallel group is the aggregator
            if i < len(steps) and len(group_indices) > 1:
                parallel_groups[i] = group_indices
        else:
            i += 1

    return parallel_groups


def enrich_log(
    log: OcelLog,
    scenario: DomainScenario,
    client: LLMClient | None = None,
    progress: Progress | None = None,
    progress_task: TaskID | None = None,
) -> None:
    """Enrich an OcelLog in-place with LLM-generated content.

    Returns the number of steps that failed enrichment (0 if all succeeded).
    """
    if client is None:
        client = LLMClient()

    failed_steps = 0

    run_ids = sorted({o.id for o in log.objects if o.type == "run"})

    pattern_desc = ""
    for obj in log.objects:
        if obj.type == "run":
            for attr in obj.attributes:
                if attr.name == "pattern_type":
                    pattern_desc = attr.value
            break

    # Expand queries if we have fewer than needed
    expanded_queries: list[str] | None = None
    if len(scenario.user_queries) < len(run_ids):
        try:
            expanded_queries = client.generate_queries(
                seed_queries=scenario.user_queries,
                domain_description=scenario.description,
                count=len(run_ids),
            )
        except Exception as exc:
            logger.warning(
                "Query expansion failed, falling back to cycling %d seed queries: %s",
                len(scenario.user_queries),
                exc,
            )
            expanded_queries = None

    for run_idx, run_id in enumerate(run_ids):
        # Use expanded queries if available, otherwise fall back to cycling
        if expanded_queries and run_idx < len(expanded_queries):
            user_query = expanded_queries[run_idx]
        else:
            user_query = scenario.query_for_run(run_idx)

        run_obj = _get_object(log, run_id)
        if run_obj:
            _patch_attribute(run_obj, "user_query", user_query)

        task_obj = _get_object(log, f"{run_id}-task")
        if task_obj:
            _patch_attribute(task_obj, "description", user_query)

        steps = _extract_steps_from_log(log, run_id)

        # Improvement 3: Detect deviations for this run
        deviations = _detect_run_deviations(log, run_id)

        # Improvement 5: Track outputs per step for parallel coherence
        step_outputs: dict[str, str] = {}  # agent_role -> output
        previous_output: str | None = None

        # Detect parallel groups for aggregator step handling
        parallel_groups = _detect_parallel_groups(steps)

        for step_idx, step in enumerate(steps):
            role = step["agent_role"]
            tool_names = _get_tool_names_for_step(log, step)
            persona = scenario.agent_personas.get(role, f"You are a {role} agent")

            # Improvement 5: For aggregator steps, concatenate all parallel outputs
            effective_previous_output = previous_output
            if step_idx in parallel_groups:
                worker_indices = parallel_groups[step_idx]
                worker_outputs = []
                for wi in worker_indices:
                    w_role = steps[wi]["agent_role"]
                    w_output = step_outputs.get(w_role)
                    if w_output:
                        worker_outputs.append(f"**{w_role}**: {w_output}")
                if worker_outputs:
                    effective_previous_output = "Results from parallel agents:\n\n" + "\n\n".join(
                        worker_outputs
                    )

            # Improvement 3: Build deviation context for this step
            step_id_for_dev = (
                step.get("invocation_id", "").split("-inv-")[-1]
                if "-inv-" in step.get("invocation_id", "")
                else ""
            )
            deviation_context = _build_deviation_context(deviations, step_id_for_dev)

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
                previous_output=effective_previous_output,
                deviation_context=deviation_context,
            )

            try:
                raw = client.generate(system_prompt, user_prompt)
                resp = EnrichmentResponse.from_dict(
                    raw,
                    expected_llm_calls=step["expected_llm_calls"],
                    expected_tool_calls=step["expected_tool_calls"],
                )
            except Exception as exc:
                failed_steps += 1
                logger.warning(
                    "Enrichment failed for %s step %d (%s): %s",
                    run_id,
                    step_idx,
                    role,
                    exc,
                )
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
                    _patch_attribute(
                        tool_obj, "tool_input", json.dumps(resp.tool_calls[i].get("input", {}))
                    )
                    _patch_attribute(
                        tool_obj, "tool_output", json.dumps(resp.tool_calls[i].get("output", {}))
                    )

            if step["message_id"]:
                msg_obj = _get_object(log, step["message_id"])
                if msg_obj:
                    _patch_attribute(msg_obj, "content", resp.output_to_next_agent)

            # Improvement 1: Recalculate token counts and costs
            _recalculate_metrics(log, step, resp)

            # Track output for parallel coherence (Improvement 5)
            previous_output = resp.output_to_next_agent
            step_outputs[role] = resp.output_to_next_agent

        # Rewrite timestamps after enriching all steps in this run
        _rewrite_timestamps(log, run_id)

        if progress and progress_task is not None:
            progress.advance(progress_task)

    if failed_steps:
        logger.warning("Enrichment completed with %d failed step(s)", failed_steps)

    return failed_steps
