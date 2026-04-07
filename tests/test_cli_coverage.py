"""Tests for CLI commands: generate, validate, list-patterns, enrich, upload, pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ocelgen.cli import app

runner = CliRunner()


class TestGenerateCommand:
    def test_generate_default(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonocel"
        result = runner.invoke(
            app,
            [
                "generate",
                "-n",
                "3",
                "--seed",
                "42",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()
        assert (tmp_path / "normative_model.json").exists()
        assert (tmp_path / "manifest.json").exists()
        assert "Generation Summary" in result.output

    def test_generate_supervisor_pattern(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonocel"
        result = runner.invoke(
            app,
            [
                "generate",
                "-p",
                "supervisor",
                "-n",
                "2",
                "--seed",
                "1",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert "supervisor" in result.output.lower() or out.exists()

    def test_generate_unknown_pattern_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonocel"
        result = runner.invoke(
            app,
            [
                "generate",
                "-p",
                "nonexistent",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "Unknown pattern" in result.output


class TestValidateCommand:
    def test_validate_valid_file(self, tmp_path: Path) -> None:
        # Generate a valid file first
        out = tmp_path / "test.jsonocel"
        runner.invoke(app, ["generate", "-n", "2", "--seed", "42", "-o", str(out)])
        result = runner.invoke(app, ["validate", str(out)])
        assert result.exit_code == 0
        assert "Valid OCEL 2.0 JSON" in result.output

    def test_validate_missing_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/file.jsonocel"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_validate_invalid_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonocel"
        bad.write_text(json.dumps({"not": "valid ocel"}))
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code != 0
        assert "validation error" in result.output


class TestListPatternsCommand:
    def test_list_patterns_shows_all(self) -> None:
        result = runner.invoke(app, ["list-patterns"])
        assert result.exit_code == 0
        assert "sequential" in result.output
        assert "supervisor" in result.output
        assert "parallel" in result.output


class TestEnrichCommandExtended:
    def test_enrich_missing_file(self) -> None:
        result = runner.invoke(
            app,
            [
                "enrich",
                "/nonexistent.jsonocel",
                "--domain",
                "customer-support-triage",
            ],
        )
        assert result.exit_code != 0

    def test_enrich_invalid_domain(self, tmp_path: Path) -> None:
        ocel = tmp_path / "test.jsonocel"
        ocel.write_text(
            json.dumps(
                {
                    "eventTypes": [],
                    "objectTypes": [],
                    "events": [],
                    "objects": [],
                }
            )
        )
        result = runner.invoke(app, ["enrich", str(ocel), "--domain", "bogus-domain"])
        assert result.exit_code != 0
        assert "Unknown domain" in result.output


class TestUploadCommand:
    def test_upload_missing_file(self) -> None:
        result = runner.invoke(
            app,
            [
                "upload",
                "/nonexistent.jsonocel",
                "--domain",
                "customer-support-triage",
                "--namespace",
                "testuser",
            ],
        )
        assert result.exit_code != 0

    def test_upload_invalid_domain(self, tmp_path: Path) -> None:
        ocel = tmp_path / "test.jsonocel"
        ocel.write_text(
            json.dumps(
                {
                    "eventTypes": [],
                    "objectTypes": [],
                    "events": [],
                    "objects": [],
                }
            )
        )
        result = runner.invoke(
            app,
            [
                "upload",
                str(ocel),
                "--domain",
                "bogus",
                "--namespace",
                "testuser",
            ],
        )
        assert result.exit_code != 0

    def test_upload_missing_namespace(self, tmp_path: Path) -> None:
        ocel = tmp_path / "test.jsonocel"
        ocel.write_text(
            json.dumps(
                {
                    "eventTypes": [],
                    "objectTypes": [],
                    "events": [],
                    "objects": [],
                }
            )
        )
        result = runner.invoke(
            app,
            [
                "upload",
                str(ocel),
                "--domain",
                "customer-support-triage",
            ],
        )
        assert result.exit_code != 0


class TestPipelineCommandExtended:
    def test_pipeline_missing_namespace(self) -> None:
        result = runner.invoke(app, ["pipeline", "--domain", "customer-support-triage"])
        assert result.exit_code != 0
        assert "namespace" in result.output.lower()

    def test_pipeline_no_domain_no_all(self) -> None:
        result = runner.invoke(app, ["pipeline", "--namespace", "testuser"])
        assert result.exit_code != 0
        assert "domain" in result.output.lower() or "all" in result.output.lower()

    def test_pipeline_unknown_domain(self) -> None:
        result = runner.invoke(
            app,
            [
                "pipeline",
                "--namespace",
                "testuser",
                "--domain",
                "bogus-domain",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown domain" in result.output
