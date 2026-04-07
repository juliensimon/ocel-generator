"""Tests for export write functions and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

from ocelgen.export.manifest import write_manifest
from ocelgen.export.normative import write_normative_model
from ocelgen.generation.engine import generate


class TestWriteNormativeModel:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        out = tmp_path / "sub" / "normative.json"
        write_normative_model(result.template, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "name" in data
        assert "steps" in data


class TestWriteManifest:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        result = generate("sequential", num_runs=3, noise_rate=0.5, seed=42)
        out = tmp_path / "sub" / "manifest.json"
        write_manifest(result, out, seed=42)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["seed"] == 42
        assert data["total_runs"] == 3
        assert "runs" in data
