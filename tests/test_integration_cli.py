"""Integration tests for CLI happy paths: enrich, upload, and pipeline commands.

These tests mock only external services (LLM API, HF Hub) and exercise the
full command flow including file I/O, enrichment, flattening, and upload prep.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ocelgen.cli import app

runner = CliRunner()

# Reusable mock LLM response that satisfies sequential pattern expectations
_MOCK_LLM_RESPONSE = {
    "reasoning": "Investigating the customer issue step by step.",
    "llm_calls": [
        {
            "prompt": "Search for relevant information",
            "completion": "Found documentation about the issue.",
        },
        {"prompt": "Analyze the findings", "completion": "The analysis reveals a clear pattern."},
    ],
    "tool_calls": [
        {"input": {"query": "customer issue"}, "output": {"result": "Relevant data found"}},
        {"input": {"query": "policy lookup"}, "output": {"result": "Policy details retrieved"}},
        {"input": {"file": "orders.csv"}, "output": {"result": "Order history loaded"}},
    ],
    "output_to_next_agent": "Here are my findings for the next step.",
}

_MOCK_QUERIES_RESPONSE = {
    "queries": [
        "How do I get a refund?",
        "My order is delayed",
        "I need to change my shipping address",
    ],
}


def _generate_ocel_file(tmp_path: Path, runs: int = 3) -> Path:
    """Generate a valid OCEL file using the generate command."""
    out = tmp_path / "test.jsonocel"
    result = runner.invoke(
        app,
        [
            "generate",
            "-n",
            str(runs),
            "--seed",
            "42",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0
    return out


def _mock_openai_create(**kwargs):
    """Factory for mock OpenAI completions.create responses."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(_MOCK_LLM_RESPONSE)
    return mock_resp


def _mock_openai_create_queries(**kwargs):
    """Returns queries response for generate_queries calls."""
    content = kwargs.get("messages", [{}])[-1].get("content", "")
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    if "Generate exactly" in content:
        mock_resp.choices[0].message.content = json.dumps(_MOCK_QUERIES_RESPONSE)
    else:
        mock_resp.choices[0].message.content = json.dumps(_MOCK_LLM_RESPONSE)
    return mock_resp


class TestEnrichIntegration:
    """Test the full enrich command with mocked LLM."""

    @patch("ocelgen.enrichment.client.OpenAI")
    def test_enrich_happy_path(self, mock_openai_cls: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        ocel_path = _generate_ocel_file(tmp_path, runs=2)
        out_path = tmp_path / "enriched.jsonocel"

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                app,
                [
                    "enrich",
                    str(ocel_path),
                    "--domain",
                    "customer-support-triage",
                    "--model",
                    "gpt-4o-mini",
                    "-o",
                    str(out_path),
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert out_path.exists()
        assert "Enriched log written to" in result.output

        # Verify the output is valid JSON
        data = json.loads(out_path.read_text())
        assert "events" in data
        assert "objects" in data

    @patch("ocelgen.enrichment.client.OpenAI")
    def test_enrich_with_base_url(self, mock_openai_cls: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        ocel_path = _generate_ocel_file(tmp_path, runs=1)

        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(
                app,
                [
                    "enrich",
                    str(ocel_path),
                    "--domain",
                    "customer-support-triage",
                    "--base-url",
                    "http://localhost:8080/v1",
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"

    @patch("ocelgen.enrichment.client.OpenAI")
    def test_enrich_default_output_path(self, mock_openai_cls: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        ocel_path = _generate_ocel_file(tmp_path, runs=1)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                app,
                [
                    "enrich",
                    str(ocel_path),
                    "--domain",
                    "customer-support-triage",
                ],
            )

        assert result.exit_code == 0
        # Default output should be enriched-<original>
        expected_out = tmp_path / "enriched-test.jsonocel"
        assert expected_out.exists()


class TestUploadIntegration:
    """Test the full upload command with mocked HF Hub."""

    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_upload_happy_path(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.create_repo.return_value = None
        mock_api.upload_folder.return_value = None

        ocel_path = _generate_ocel_file(tmp_path, runs=2)

        result = runner.invoke(
            app,
            [
                "upload",
                str(ocel_path),
                "--domain",
                "customer-support-triage",
                "--namespace",
                "testuser",
            ],
        )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Uploaded" in result.output
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()


class TestPipelineIntegration:
    """Test the full pipeline command (generate → enrich → upload)."""

    @patch("ocelgen.upload.hf_upload.HfApi")
    @patch("ocelgen.enrichment.client.OpenAI")
    def test_pipeline_single_domain_with_upload(
        self,
        mock_openai_cls: MagicMock,
        mock_api_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Mock LLM
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        # Mock HF Hub
        mock_api = mock_api_cls.return_value
        mock_api.create_repo.return_value = None
        mock_api.upload_folder.return_value = None
        mock_collection = MagicMock()
        mock_collection.slug = "testuser/open-agent-traces-abc123"
        mock_api.list_collections.return_value = []
        mock_api.create_collection.return_value = mock_collection
        mock_api.add_collection_item.return_value = None

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                app,
                [
                    "pipeline",
                    "--domain",
                    "customer-support-triage",
                    "--namespace",
                    "testuser",
                    "--model",
                    "gpt-4o-mini",
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Done!" in result.output
        assert "Collection" in result.output
        mock_api.upload_folder.assert_called()

    @patch("ocelgen.enrichment.client.OpenAI")
    def test_pipeline_skip_upload(
        self,
        mock_openai_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                app,
                [
                    "pipeline",
                    "--domain",
                    "customer-support-triage",
                    "--namespace",
                    "testuser",
                    "--skip-upload",
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Skipping upload" in result.output
        assert "Done!" in result.output

    @patch("ocelgen.enrichment.client.OpenAI")
    def test_pipeline_with_base_url(
        self,
        mock_openai_cls: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=_mock_openai_create_queries)
        mock_openai_cls.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(
                app,
                [
                    "pipeline",
                    "--domain",
                    "customer-support-triage",
                    "--namespace",
                    "testuser",
                    "--base-url",
                    "http://localhost:8080/v1",
                    "--skip-upload",
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
