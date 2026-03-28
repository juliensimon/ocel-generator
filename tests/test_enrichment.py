"""Tests for enrichment client and enricher (mocked LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ocelgen.enrichment.client import LLMClient, EnrichmentResponse


class TestLLMClient:
    def test_client_creation_with_defaults(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            assert client.model == "google/gemini-2.0-flash-001"

    def test_client_custom_model(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient(model="openai/gpt-4o-mini")
            assert client.model == "openai/gpt-4o-mini"

    def test_client_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                LLMClient()


class TestEnrichmentResponse:
    def test_parse_valid_response(self) -> None:
        raw = {
            "reasoning": "I need to search the knowledge base first.",
            "llm_calls": [
                {"prompt": "Search for refund policy", "completion": "The refund policy states..."}
            ],
            "tool_calls": [
                {"input": {"query": "refund policy"}, "output": {"result": "Policy found"}}
            ],
            "output_to_next_agent": "The customer's refund is eligible for processing.",
        }
        resp = EnrichmentResponse.from_dict(raw)
        assert resp.reasoning == "I need to search the knowledge base first."
        assert len(resp.llm_calls) == 1
        assert resp.llm_calls[0]["prompt"] == "Search for refund policy"
        assert len(resp.tool_calls) == 1
        assert resp.output_to_next_agent == "The customer's refund is eligible for processing."

    def test_parse_missing_fields_uses_defaults(self) -> None:
        raw = {"reasoning": "thinking..."}
        resp = EnrichmentResponse.from_dict(raw)
        assert resp.reasoning == "thinking..."
        assert resp.llm_calls == []
        assert resp.tool_calls == []
        assert resp.output_to_next_agent == ""

    def test_parse_extra_llm_calls_trimmed(self) -> None:
        raw = {
            "reasoning": "ok",
            "llm_calls": [
                {"prompt": "p1", "completion": "c1"},
                {"prompt": "p2", "completion": "c2"},
                {"prompt": "p3", "completion": "c3"},
            ],
            "tool_calls": [],
            "output_to_next_agent": "done",
        }
        resp = EnrichmentResponse.from_dict(raw, expected_llm_calls=2)
        assert len(resp.llm_calls) == 2

    def test_parse_extra_tool_calls_trimmed(self) -> None:
        raw = {
            "reasoning": "ok",
            "llm_calls": [],
            "tool_calls": [
                {"input": {}, "output": {}},
                {"input": {}, "output": {}},
            ],
            "output_to_next_agent": "done",
        }
        resp = EnrichmentResponse.from_dict(raw, expected_tool_calls=1)
        assert len(resp.tool_calls) == 1
