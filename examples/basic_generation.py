#!/usr/bin/env python3
"""Basic example: generate OCEL 2.0 event logs programmatically.

This script demonstrates using the ocelgen API directly (without the CLI)
to generate event logs, inspect the results, and write output files.
"""

from pathlib import Path

from ocelgen.export.manifest import build_manifest, write_manifest
from ocelgen.export.normative import write_normative_model
from ocelgen.export.ocel_json import ocel_log_to_dict, write_ocel_json
from ocelgen.generation.engine import generate
from ocelgen.validation.schema import validate_ocel_dict


def main() -> None:
    # Generate 100 sequential runs with 20% noise, deterministic seed
    result = generate(
        pattern_name="sequential",
        num_runs=100,
        noise_rate=0.2,
        seed=42,
    )

    # Inspect the result
    print(f"Pattern:         {result.template.name}")
    print(f"Total runs:      {result.total_runs}")
    print(f"Conformant runs: {result.conformant_runs}")
    print(f"Deviant runs:    {result.deviant_runs}")
    print(f"Total events:    {len(result.log.events)}")
    print(f"Total objects:   {len(result.log.objects)}")
    print(f"Deviations:      {len(result.deviations)}")

    # Validate
    errors = validate_ocel_dict(ocel_log_to_dict(result.log))
    print(f"\nSchema validation: {'PASS' if not errors else 'FAIL'}")
    for err in errors:
        print(f"  - {err}")

    # Write output files
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    write_ocel_json(result.log, output_dir / "sequential.jsonocel")
    write_normative_model(result.template, output_dir / "normative_model.json")
    write_manifest(result, output_dir / "manifest.json", seed=42)

    print(f"\nFiles written to {output_dir}/")

    # Print deviation breakdown
    manifest = build_manifest(result, seed=42)
    print("\nDeviation breakdown:")
    for dev_type, count in sorted(manifest["deviation_summary"].items()):
        print(f"  {dev_type}: {count}")


if __name__ == "__main__":
    main()
