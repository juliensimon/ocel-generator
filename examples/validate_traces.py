#!/usr/bin/env python3
"""Run all semantic validators on generated OCEL 2.0 traces.

Demonstrates four validation layers beyond JSON schema:
  1. Referential integrity — every relationship points to an existing object
  2. Type attributes — every attribute is declared in its type schema
  3. Temporal ordering — events respect causal order within each run
  4. Workflow conformance — conformant runs follow the normative template
"""

from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.generation.engine import generate
from ocelgen.validation import (
    validate_ocel_dict,
    validate_referential_integrity,
    validate_temporal_order,
    validate_type_attributes,
    validate_workflow_conformance,
)


def main() -> None:
    for pattern in ("sequential", "supervisor", "parallel"):
        result = generate(
            pattern_name=pattern,
            num_runs=50,
            noise_rate=0.3,
            seed=42,
        )
        log = result.log
        template = result.template

        print(f"\n{'=' * 60}")
        print(f"Pattern: {pattern}  |  {len(log.events)} events, {len(log.objects)} objects")
        print(f"{'=' * 60}")

        # 1. JSON Schema
        errors = validate_ocel_dict(ocel_log_to_dict(log))
        _report("JSON Schema", errors)

        # 2. Referential integrity
        errors = validate_referential_integrity(log)
        _report("Referential integrity", errors)

        # 3. Type attributes
        errors = validate_type_attributes(log)
        _report("Type attributes", errors)

        # 4. Temporal ordering (conformant runs only)
        errors = validate_temporal_order(log)
        deviant_ids = {s.run_id for s in result.deviations}
        conformant_errors = [e for e in errors if not any(rid in e for rid in deviant_ids)]
        # Parallel patterns interleave sequence numbers by design
        if pattern == "parallel":
            conformant_errors = [e for e in conformant_errors if "sequence" not in e]
        _report("Temporal ordering (conformant runs)", conformant_errors)

        # 5. Workflow conformance
        errors = validate_workflow_conformance(log, template)
        _report("Workflow conformance", errors)


def _report(name: str, errors: list[str]) -> None:
    status = "PASS" if not errors else f"FAIL ({len(errors)} errors)"
    print(f"  {name:40s} {status}")
    for e in errors[:3]:
        print(f"    {e}")
    if len(errors) > 3:
        print(f"    ... and {len(errors) - 3} more")


if __name__ == "__main__":
    main()
