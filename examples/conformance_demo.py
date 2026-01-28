#!/usr/bin/env python3
"""Conformance checking demo using pm4py.

This script generates an OCEL 2.0 event log, writes it to disk,
and loads it with pm4py to verify it parses correctly.

Requires the optional `conformance` extra:
    uv pip install ocelgen[conformance]
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from ocelgen.export.ocel_json import write_ocel_json
from ocelgen.generation.engine import generate


def main() -> None:
    # Generate a small log
    result = generate(
        pattern_name="supervisor",
        num_runs=20,
        noise_rate=0.3,
        seed=42,
    )

    with TemporaryDirectory() as tmpdir:
        ocel_path = Path(tmpdir) / "supervisor.jsonocel"
        write_ocel_json(result.log, ocel_path)

        try:
            import pm4py

            ocel = pm4py.read.read_ocel2_json(str(ocel_path))
            print("pm4py successfully loaded the OCEL 2.0 log!")
            print(f"  Events:  {len(ocel.events)}")
            print(f"  Objects: {len(ocel.objects)}")
            print(f"  Event types:  {ocel.events['ocel:type'].nunique()}")
            print(f"  Object types: {ocel.objects['ocel:type'].nunique()}")
        except ImportError:
            print("pm4py not installed. Install with: uv pip install ocelgen[conformance]")
            print(f"The OCEL file was written to {ocel_path} for manual inspection.")
        except Exception as e:
            print(f"pm4py error: {e}")
            print("This may indicate a format compatibility issue.")


if __name__ == "__main__":
    main()
