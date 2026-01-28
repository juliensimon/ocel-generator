"""Generation manifest with ground truth for conformance checking.

The manifest records exactly which runs are conformant, which have deviations,
and what type of deviation was injected. This serves as the ground truth for
evaluating conformance checking algorithms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ocelgen.deviations.types import DeviationSpec
from ocelgen.generation.engine import GenerationResult


def build_manifest(result: GenerationResult, seed: int | None = None) -> dict[str, Any]:
    """Build a manifest dict from a generation result."""
    # Group deviations by run_id
    deviations_by_run: dict[str, list[dict[str, Any]]] = {}
    for spec in result.deviations:
        if spec.run_id not in deviations_by_run:
            deviations_by_run[spec.run_id] = []
        deviations_by_run[spec.run_id].append({
            "deviation_type": spec.deviation_type.value,
            "step_id": spec.step_id,
            "description": spec.description,
        })

    # Build run summary
    runs: list[dict[str, Any]] = []
    for i in range(result.total_runs):
        run_id = f"run-{i:04d}"
        is_conformant = run_id not in deviations_by_run
        run_entry: dict[str, Any] = {
            "run_id": run_id,
            "is_conformant": is_conformant,
        }
        if not is_conformant:
            run_entry["deviations"] = deviations_by_run[run_id]
        runs.append(run_entry)

    return {
        "generator": "ocelgen",
        "version": "0.1.0",
        "pattern": result.template.name,
        "seed": seed,
        "total_runs": result.total_runs,
        "conformant_runs": result.conformant_runs,
        "deviant_runs": result.deviant_runs,
        "deviation_summary": _deviation_summary(result.deviations),
        "runs": runs,
    }


def _deviation_summary(deviations: list[DeviationSpec]) -> dict[str, int]:
    """Count deviations by type."""
    counts: dict[str, int] = {}
    for spec in deviations:
        key = spec.deviation_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_manifest(result: GenerationResult, path: Path, seed: int | None = None) -> None:
    """Write the generation manifest to a JSON file."""
    data = build_manifest(result, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
