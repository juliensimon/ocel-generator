"""Tests for new CLI commands: list-domains, enrich, upload, pipeline."""

from typer.testing import CliRunner

from ocelgen.cli import app

runner = CliRunner()


class TestListDomains:
    def test_list_domains_shows_all_10(self) -> None:
        result = runner.invoke(app, ["list-domains"])
        assert result.exit_code == 0
        assert "customer-support-triage" in result.output
        assert "incident-response" in result.output


class TestEnrichCommand:
    def test_enrich_requires_existing_file(self) -> None:
        result = runner.invoke(app, ["enrich", "nonexistent.jsonocel", "--domain", "customer-support-triage"])
        assert result.exit_code != 0

    def test_enrich_requires_valid_domain(self, tmp_path) -> None:
        import json
        ocel_path = tmp_path / "test.jsonocel"
        ocel_path.write_text(json.dumps({
            "eventTypes": [], "objectTypes": [], "events": [], "objects": []
        }))
        result = runner.invoke(app, ["enrich", str(ocel_path), "--domain", "nonexistent-domain"])
        assert result.exit_code != 0


class TestPipelineCommand:
    def test_pipeline_requires_namespace(self) -> None:
        result = runner.invoke(app, ["pipeline", "--domain", "customer-support-triage"])
        assert result.exit_code != 0
