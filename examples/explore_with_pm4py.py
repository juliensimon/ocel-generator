#!/usr/bin/env python3
"""Load agent traces from Hugging Face and explore them with pm4py.

Shows how to use the published dataset with pm4py — the reference
implementation for OCEL 2.0 process mining.

Requires: uv pip install ocelgen[conformance]
"""

from huggingface_hub import hf_hub_download

REPO_ID = "juliensimon/open-agent-traces"


def main() -> None:
    try:
        import pm4py
    except ImportError:
        print("pm4py not installed. Install with: uv pip install ocelgen[conformance]")
        return

    # Download one domain's OCEL file
    domain = "incident-response"
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"ocel/{domain}/output.jsonocel",
        repo_type="dataset",
    )

    # Load with pm4py
    ocel = pm4py.read.read_ocel2_json(path)

    print(f"Domain: {domain}")
    print(f"Events:  {len(ocel.events)}")
    print(f"Objects: {len(ocel.objects)}")
    print()

    # Event types — pm4py uses 'ocel:activity' (not 'ocel:type')
    print("Event types:")
    for t in sorted(ocel.events["ocel:activity"].unique()):
        count = (ocel.events["ocel:activity"] == t).sum()
        print(f"  {t:30s} {count:5d}")
    print()

    # Object types — pm4py uses 'ocel:type'
    print("Object types:")
    for t in sorted(ocel.objects["ocel:type"].unique()):
        count = (ocel.objects["ocel:type"] == t).sum()
        print(f"  {t:25s} {count:5d}")
    print()

    # Relationships
    if ocel.relations is not None and len(ocel.relations) > 0:
        print(f"Relationships: {len(ocel.relations)}")
        print(f"Columns: {list(ocel.relations.columns)}")
        print()

        # Show qualifier distribution
        if "ocel:qualifier" in ocel.relations.columns:
            print("Relationship qualifiers:")
            for q in sorted(ocel.relations["ocel:qualifier"].unique()):
                count = (ocel.relations["ocel:qualifier"] == q).sum()
                print(f"  {q:25s} {count:5d}")

    # You can also load via the datasets library for tabular access
    print("\n--- Tabular access via datasets library ---")
    from datasets import load_dataset

    ds = load_dataset(REPO_ID, domain)
    train = ds["train"]
    print(f"Rows: {len(train)}")
    print(f"Columns: {train.column_names}")

    # Filter to LLM completions
    llm_events = train.filter(lambda x: x["event_type"] == "llm_response_received")
    print(f"\nLLM completions: {len(llm_events)}")
    if len(llm_events) > 0:
        row = llm_events[0]
        print(f"  Prompt:     {row['prompt'][:80]}...")
        print(f"  Completion: {row['completion'][:80]}...")
        print(f"  Tokens:     {row['input_tokens']} in / {row['output_tokens']} out")


if __name__ == "__main__":
    main()
