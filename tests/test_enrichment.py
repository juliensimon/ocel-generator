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


from ocelgen.enrichment.prompts import build_enrichment_prompt


class TestPromptBuilder:
    def test_build_prompt_basic(self) -> None:
        system, user = build_enrichment_prompt(
            domain_description="Customer support triage workflow",
            pattern_description="Linear chain: Research -> Analyze -> Summarize",
            agent_role="researcher",
            agent_persona="You are a support agent researching the issue",
            user_query="My refund hasn't arrived after 10 days",
            tool_names=["web_search", "file_reader"],
            tool_descriptions={
                "web_search": "Search the knowledge base",
                "file_reader": "Read customer order history",
            },
            expected_llm_calls=2,
            expected_tool_calls=1,
            previous_output=None,
        )
        assert "Customer support" in system
        assert "researcher" in user
        assert "refund" in user
        assert "web_search" in user
        assert '"llm_calls"' in user
        assert "2" in user  # expected_llm_calls

    def test_build_prompt_with_previous_output(self) -> None:
        _, user = build_enrichment_prompt(
            domain_description="Test domain",
            pattern_description="Test pattern",
            agent_role="analyst",
            agent_persona="You are an analyst",
            user_query="Test query",
            tool_names=[],
            tool_descriptions={},
            expected_llm_calls=1,
            expected_tool_calls=0,
            previous_output="The researcher found that...",
        )
        assert "The researcher found that..." in user

    def test_build_prompt_no_tools(self) -> None:
        _, user = build_enrichment_prompt(
            domain_description="Test domain",
            pattern_description="Test pattern",
            agent_role="summarizer",
            agent_persona="You are a summarizer",
            user_query="Test query",
            tool_names=[],
            tool_descriptions={},
            expected_llm_calls=1,
            expected_tool_calls=0,
            previous_output="Previous analysis results...",
        )
        assert "no tools" in user.lower() or "none" in user.lower() or "0" in user


from unittest.mock import MagicMock

from ocelgen.enrichment.enricher import enrich_log, _extract_steps_from_log
from ocelgen.generation.engine import generate
from ocelgen.export.ocel_json import ocel_log_to_dict
from ocelgen.scenarios.domain import DomainScenario


def _make_test_scenario() -> DomainScenario:
    return DomainScenario(
        name="test-domain",
        description="Test domain for unit tests",
        pattern="sequential",
        runs=3,
        noise=0.0,
        seed=42,
        user_queries=["Test query one", "Test query two", "Test query three"],
        agent_personas={
            "researcher": "Test researcher persona",
            "analyst": "Test analyst persona",
            "summarizer": "Test summarizer persona",
        },
        tool_descriptions={
            "web_search": "Test web search",
            "file_reader": "Test file reader",
            "calculator": "Test calculator",
            "code_interpreter": "Test code interpreter",
        },
    )


class TestExtractSteps:
    def test_extract_steps_from_sequential_run(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        steps = _extract_steps_from_log(result.log, "run-0000")
        # Sequential has 3 steps: research, analyze, summarize
        assert len(steps) == 3
        for step in steps:
            assert "agent_role" in step
            assert "invocation_id" in step
            assert "llm_call_ids" in step
            assert "tool_call_ids" in step

    def test_extract_steps_from_supervisor_run(self) -> None:
        result = generate("supervisor", num_runs=1, noise_rate=0.0, seed=42)
        steps = _extract_steps_from_log(result.log, "run-0000")
        assert len(steps) >= 3


class TestEnrichLog:
    def test_enrich_patches_llm_call_objects(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_response = {
            "reasoning": "I need to investigate this.",
            "llm_calls": [
                {"prompt": "Find info about test query", "completion": "I found that..."},
                {"prompt": "Analyze the findings", "completion": "The analysis shows..."},
            ],
            "tool_calls": [
                {"input": {"query": "test"}, "output": {"result": "found"}},
                {"input": {"query": "test2"}, "output": {"result": "found2"}},
                {"input": {"query": "test3"}, "output": {"result": "found3"}},
            ],
            "output_to_next_agent": "Here are my findings.",
        }

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        enrich_log(result.log, scenario, client=mock_client)

        llm_objs = [o for o in result.log.objects if o.type == "llm_call"]
        assert len(llm_objs) > 0
        enriched = [o for o in llm_objs if any(a.name == "prompt" for a in o.attributes)]
        assert len(enriched) > 0

    def test_enrich_preserves_ocel_validity(self) -> None:
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_response = {
            "reasoning": "Thinking...",
            "llm_calls": [
                {"prompt": "p", "completion": "c"},
                {"prompt": "p2", "completion": "c2"},
            ],
            "tool_calls": [
                {"input": {"q": "v"}, "output": {"r": "v"}},
                {"input": {"q": "v"}, "output": {"r": "v"}},
                {"input": {"q": "v"}, "output": {"r": "v"}},
            ],
            "output_to_next_agent": "Done.",
        }

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        enrich_log(result.log, scenario, client=mock_client)

        from ocelgen.validation.schema import validate_ocel_dict
        errors = validate_ocel_dict(ocel_log_to_dict(result.log))
        assert errors == [], f"OCEL validation failed after enrichment: {errors}"

    def test_enrich_replaces_user_query(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        scenario = _make_test_scenario()

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "reasoning": "ok",
            "llm_calls": [{"prompt": "p", "completion": "c"}, {"prompt": "p", "completion": "c"}],
            "tool_calls": [{"input": {}, "output": {}}, {"input": {}, "output": {}}, {"input": {}, "output": {}}],
            "output_to_next_agent": "done",
        }

        enrich_log(result.log, scenario, client=mock_client)

        run_obj = next(o for o in result.log.objects if o.id == "run-0000")
        query_attr = next(a for a in run_obj.attributes if a.name == "user_query")
        assert query_attr.value == "Test query one"
