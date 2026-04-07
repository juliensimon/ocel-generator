#!/usr/bin/env python3
"""Inspect a single run's events, LLM calls, tool calls, and deviations.

Shows how to walk the OCEL 2.0 object graph for a specific run —
the same data you'd see in an agent observability dashboard.
"""

from ocelgen.generation.engine import generate


def main() -> None:
    result = generate(
        pattern_name="sequential",
        num_runs=10,
        noise_rate=0.3,
        seed=42,
    )
    log = result.log
    run_id = "run-0000"

    # --- Run metadata ---
    run_obj = next((o for o in log.objects if o.id == run_id), None)
    if run_obj is None:
        print(f"Run '{run_id}' not found in log")
        return
    attrs = {a.name: a.value for a in run_obj.attributes}
    print(f"Run: {run_id}")
    print(f"  Pattern:     {attrs.get('pattern_type')}")
    print(f"  Conformant:  {attrs.get('is_conformant')}")
    print(f"  User query:  {attrs.get('user_query')}")
    print()

    # --- Events timeline ---
    run_events = sorted(
        [
            e
            for e in log.events
            if any(a.name == "run_id" and a.value == run_id for a in e.attributes)
        ],
        key=lambda e: e.time,
    )

    # Build object lookup
    obj_index = {o.id: o for o in log.objects}

    print(f"Events ({len(run_events)}):")
    for event in run_events:
        ea = {a.name: a.value for a in event.attributes}
        is_dev = ea.get("is_deviation") == "true"
        dev_marker = f" [DEVIATION: {ea.get('deviation_type')}]" if is_dev else ""

        # Resolve key relationships
        related = ""
        for rel in event.relationships:
            obj = obj_index.get(rel.objectId)
            if obj and obj.type == "agent":
                role = next((a.value for a in obj.attributes if a.name == "role"), "?")
                related = f" agent={role}"
            elif obj and obj.type == "llm_call":
                model = next((a.value for a in obj.attributes if a.name == "model"), "?")
                tokens = next((a.value for a in obj.attributes if a.name == "output_tokens"), "?")
                related = f" model={model} tokens={tokens}"
            elif obj and obj.type == "tool_call":
                tool = next((a.value for a in obj.attributes if a.name == "tool_name"), "?")
                related = f" tool={tool}"

        ts = event.time.strftime("%H:%M:%S.%f")[:-3]
        print(f"  {ts}  {event.type:25s}{related}{dev_marker}")

    # --- Deviation summary ---
    dev_events = [
        e
        for e in run_events
        if any(a.name == "is_deviation" and a.value == "true" for a in e.attributes)
    ]
    if dev_events:
        print(f"\nDeviations in this run ({len(dev_events)}):")
        for e in dev_events:
            ea = {a.name: a.value for a in e.attributes}
            print(f"  {e.type}: {ea.get('deviation_type')}")

    # --- LLM calls ---
    llm_objs = [o for o in log.objects if o.type == "llm_call" and o.id.startswith(run_id)]
    if llm_objs:
        print(f"\nLLM calls ({len(llm_objs)}):")
        for obj in llm_objs:
            oa = {a.name: a.value for a in obj.attributes}
            print(f"  {obj.id}")
            print(f"    Model:   {oa.get('model')}")
            print(f"    Tokens:  {oa.get('input_tokens')} in / {oa.get('output_tokens')} out")
            print(f"    Latency: {oa.get('latency_ms')} ms")

    # --- Tool calls ---
    tool_objs = [o for o in log.objects if o.type == "tool_call" and o.id.startswith(run_id)]
    if tool_objs:
        print(f"\nTool calls ({len(tool_objs)}):")
        for obj in tool_objs:
            oa = {a.name: a.value for a in obj.attributes}
            print(
                f"  {oa.get('tool_name'):20s} status={oa.get('status')}  duration={oa.get('duration_ms')}ms"
            )

    # --- Cost summary ---
    inv_objs = [o for o in log.objects if o.type == "agent_invocation" and o.id.startswith(run_id)]
    total_cost = sum(
        float(next((a.value for a in o.attributes if a.name == "cost_usd"), "0")) for o in inv_objs
    )
    total_input = sum(
        int(next((a.value for a in o.attributes if a.name == "input_tokens"), "0"))
        for o in inv_objs
    )
    total_output = sum(
        int(next((a.value for a in o.attributes if a.name == "output_tokens"), "0"))
        for o in inv_objs
    )
    print(f"\nRun cost: ${total_cost:.6f}  ({total_input} input + {total_output} output tokens)")


if __name__ == "__main__":
    main()
