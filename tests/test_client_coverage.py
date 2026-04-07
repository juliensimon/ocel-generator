"""Tests for LLM client generate/retry and generate_queries methods."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ocelgen.enrichment.client import EnrichmentResponse, LLMClient


class TestLLMClientGenerate:
    def _make_client(self) -> LLMClient:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            return LLMClient()

    def test_generate_returns_parsed_json(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"result": "ok"})
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        result = client.generate("system", "user")
        assert result == {"result": "ok"}

    def test_generate_retries_on_failure(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"ok": True})

        client._client.chat.completions.create = MagicMock(
            side_effect=[Exception("fail1"), Exception("fail2"), mock_response],
        )

        result = client.generate("system", "user")
        assert result == {"ok": True}
        assert client._client.chat.completions.create.call_count == 3

    def test_generate_raises_after_max_retries(self) -> None:
        client = self._make_client()
        client._client.chat.completions.create = MagicMock(
            side_effect=Exception("always fails"),
        )

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.generate("system", "user")

    def test_generate_handles_empty_content(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        result = client.generate("system", "user")
        assert result == {}


class TestLLMClientGenerateQueries:
    def _make_client(self) -> LLMClient:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            return LLMClient()

    def test_returns_exact_count(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "queries": ["q1", "q2", "q3", "q4", "q5"],
            }
        )
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        queries = client.generate_queries(["seed1", "seed2"], "test domain", 5)
        assert len(queries) == 5

    def test_pads_with_seeds_if_too_few(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "queries": ["q1"],
            }
        )
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        queries = client.generate_queries(["seed1", "seed2"], "test domain", 3)
        assert len(queries) == 3
        assert queries[0] == "q1"
        # Padded with seeds
        assert queries[1] == "seed1"
        assert queries[2] == "seed2"

    def test_truncates_if_too_many(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "queries": ["q1", "q2", "q3", "q4", "q5"],
            }
        )
        client._client.chat.completions.create = MagicMock(return_value=mock_response)

        queries = client.generate_queries(["seed1"], "test domain", 2)
        assert len(queries) == 2


class TestLLMClientLocalServer:
    def test_local_server_no_api_key_needed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(base_url="http://localhost:8080/v1")
            assert client.model == "google/gemini-2.0-flash-001"

    def test_127_0_0_1_no_api_key_needed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(base_url="http://127.0.0.1:8080/v1")
            assert client.model == "google/gemini-2.0-flash-001"


class TestEnrichmentResponseCoercion:
    def test_non_string_reasoning_coerced(self) -> None:
        raw = {
            "reasoning": {"step": "thinking"},
            "output_to_next_agent": ["list", "output"],
        }
        resp = EnrichmentResponse.from_dict(raw)
        assert isinstance(resp.reasoning, str)
        assert isinstance(resp.output_to_next_agent, str)
