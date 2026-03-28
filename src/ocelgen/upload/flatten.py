"""Flatten an OcelLog into tabular rows (one row per event)."""

from __future__ import annotations

from ocelgen.models.ocel import OcelLog


def _build_object_index(log: OcelLog) -> dict[str, dict]:
    """Build a lookup: object_id -> {attr_name: attr_value, "_type": type}."""
    index: dict[str, dict] = {}
    for obj in log.objects:
        attrs = {"_type": obj.type}
        for a in obj.attributes:
            attrs[a.name] = a.value
        index[obj.id] = attrs
    return index


def _get_event_attr(event, name: str, default: str = "") -> str:
    """Get a string attribute from an event."""
    for a in event.attributes:
        if a.name == name:
            return a.value
    return default


def flatten_log(log: OcelLog, domain: str) -> list[dict]:
    """Convert an OcelLog to a flat list of dicts (one per event).

    Resolves object relationships to denormalize agent, tool, LLM, and
    message attributes into each event row.
    """
    obj_index = _build_object_index(log)

    # Build run-level metadata index
    run_meta: dict[str, dict] = {}
    for obj in log.objects:
        if obj.type == "run":
            meta: dict[str, str] = {}
            for a in obj.attributes:
                meta[a.name] = a.value
            run_meta[obj.id] = meta

    rows: list[dict] = []

    for event in log.events:
        run_id = _get_event_attr(event, "run_id")
        rmeta = run_meta.get(run_id, {})

        row: dict = {
            "event_id": event.id,
            "event_type": event.type,
            "timestamp": event.time.isoformat(),
            "run_id": run_id,
            "sequence_number": int(_get_event_attr(event, "sequence_number", "0")),
            "is_deviation": _get_event_attr(event, "is_deviation", "false") == "true",
            "deviation_type": _get_event_attr(event, "deviation_type"),
            "step_id": _get_event_attr(event, "step_id"),
            # Resolved from related objects
            "agent_role": "",
            "model_name": "",
            "prompt": "",
            "completion": "",
            "tool_name": "",
            "tool_input": "",
            "tool_output": "",
            "message_content": "",
            "reasoning": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            # Run-level metadata
            "is_conformant": rmeta.get("is_conformant", "true") == "true",
            "pattern": rmeta.get("pattern_type", ""),
            "domain": domain,
            "user_query": rmeta.get("user_query", ""),
        }

        # Resolve relationships to populate denormalized columns
        for rel in event.relationships:
            obj = obj_index.get(rel.objectId, {})
            obj_type = obj.get("_type", "")

            if obj_type == "agent":
                row["agent_role"] = obj.get("role", "")
                row["model_name"] = obj.get("model_name", "")

            elif obj_type == "llm_call":
                row["prompt"] = obj.get("prompt", "")
                row["completion"] = obj.get("completion", "")
                row["input_tokens"] = int(obj.get("input_tokens", "0"))
                row["output_tokens"] = int(obj.get("output_tokens", "0"))
                row["latency_ms"] = int(obj.get("latency_ms", "0"))

            elif obj_type == "tool_call":
                row["tool_name"] = obj.get("tool_name", "")
                row["tool_input"] = obj.get("tool_input", "")
                row["tool_output"] = obj.get("tool_output", "")
                row["latency_ms"] = int(obj.get("duration_ms", "0"))

            elif obj_type == "agent_invocation":
                row["reasoning"] = obj.get("reasoning", "")
                row["input_tokens"] = int(obj.get("input_tokens", "0"))
                row["output_tokens"] = int(obj.get("output_tokens", "0"))
                row["cost_usd"] = float(obj.get("cost_usd", "0"))

            elif obj_type == "message":
                row["message_content"] = obj.get("content", "")

        rows.append(row)

    return rows
