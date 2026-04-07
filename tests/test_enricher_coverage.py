"""Integration tests for enricher edge cases: parallel groups, deviations, error recovery."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

from ocelgen.enrichment.enricher import (
    _build_deviation_context,
    _detect_parallel_groups,
    _get_object,
    _get_tool_names_for_step,
    _patch_attribute,
    _rewrite_timestamps,
    enrich_log,
)
from ocelgen.generation.engine import generate
from ocelgen.models.ocel import OcelObject, OcelObjectAttribute
from ocelgen.scenarios.domain import DomainScenario


def _make_scenario(runs: int = 3) -> DomainScenario:
    return DomainScenario(
        name="test-domain",
        description="Test domain",
        pattern="sequential",
        runs=runs,
        noise=0.0,
        seed=42,
        user_queries=["q1", "q2", "q3"],
        agent_personas={
            "researcher": "R persona",
            "analyst": "A persona",
            "summarizer": "S persona",
        },
        tool_descriptions={"web_search": "Search", "file_reader": "Read files"},
    )


def _mock_response():
    return {
        "reasoning": "Investigating.",
        "llm_calls": [
            {"prompt": "p1", "completion": "c1"},
            {"prompt": "p2", "completion": "c2"},
        ],
        "tool_calls": [
            {"input": {"q": "v"}, "output": {"r": "v"}},
            {"input": {"q": "v"}, "output": {"r": "v"}},
            {"input": {"q": "v"}, "output": {"r": "v"}},
        ],
        "output_to_next_agent": "Done.",
    }


class TestBuildDeviationContext:
    def test_wrong_tool_deviation(self) -> None:
        devs = [{"type": "wrong_tool_used", "step_id": "step1", "event_type": "tool_called"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None
        assert "WRONG TOOL" in ctx

    def test_skip_deviation(self) -> None:
        devs = [{"type": "skip_activity", "step_id": "step1", "event_type": "agent_invoked"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None
        assert "SKIPPED" in ctx

    def test_extra_deviation(self) -> None:
        devs = [{"type": "extra_step", "step_id": "step1", "event_type": "agent_invoked"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None
        assert "EXTRA" in ctx

    def test_reorder_deviation(self) -> None:
        devs = [{"type": "reorder_swap", "step_id": "step1", "event_type": "agent_invoked"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None
        assert "REORDERED" in ctx

    def test_generic_deviation(self) -> None:
        devs = [{"type": "timeout_error", "step_id": "step1", "event_type": "error_occurred"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None
        assert "Deviation detected" in ctx

    def test_no_matching_step_returns_none(self) -> None:
        devs = [{"type": "wrong_tool", "step_id": "step99", "event_type": "tool_called"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is None

    def test_deviation_without_step_id_matches_any(self) -> None:
        devs = [{"type": "timeout_error", "step_id": "", "event_type": "error_occurred"}]
        ctx = _build_deviation_context(devs, "step1")
        assert ctx is not None


class TestDetectParallelGroups:
    def test_no_parallel_markers(self) -> None:
        steps = [
            {"invocation_id": "run-0000-inv-step0"},
            {"invocation_id": "run-0000-inv-step1"},
            {"invocation_id": "run-0000-inv-step2"},
        ]
        groups = _detect_parallel_groups(steps)
        assert groups == {}

    def test_detects_parallel_workers(self) -> None:
        steps = [
            {"invocation_id": "run-0000-inv-setup"},
            {"invocation_id": "run-0000-parallel-worker-0"},
            {"invocation_id": "run-0000-parallel-worker-1"},
            {"invocation_id": "run-0000-parallel-worker-2"},
            {"invocation_id": "run-0000-inv-aggregator"},
        ]
        groups = _detect_parallel_groups(steps)
        # Aggregator (index 4) should map to workers [1, 2, 3]
        assert 4 in groups
        assert groups[4] == [1, 2, 3]

    def test_single_worker_not_grouped(self) -> None:
        steps = [
            {"invocation_id": "run-0000-inv-setup"},
            {"invocation_id": "run-0000-parallel-worker-0"},
            {"invocation_id": "run-0000-inv-end"},
        ]
        groups = _detect_parallel_groups(steps)
        # Single worker should not form a group
        assert groups == {}

    def test_workers_at_end_without_aggregator(self) -> None:
        steps = [
            {"invocation_id": "run-0000-inv-setup"},
            {"invocation_id": "run-0000-worker-0"},
            {"invocation_id": "run-0000-worker-1"},
        ]
        groups = _detect_parallel_groups(steps)
        # No step after workers, so no aggregator
        assert groups == {}


class TestGetObject:
    def test_returns_matching_object(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        obj = _get_object(result.log, "run-0000")
        assert obj is not None
        assert obj.id == "run-0000"

    def test_returns_none_for_missing(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        obj = _get_object(result.log, "nonexistent-id")
        assert obj is None


class TestPatchAttribute:
    def test_patches_existing_attribute(self) -> None:
        from datetime import datetime

        obj = OcelObject(
            id="test-obj",
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value="old", time=datetime.now(UTC)),
            ],
        )
        _patch_attribute(obj, "role", "new")
        assert obj.attributes[0].value == "new"

    def test_adds_new_attribute(self) -> None:
        from datetime import datetime

        obj = OcelObject(
            id="test-obj",
            type="agent",
            attributes=[
                OcelObjectAttribute(name="role", value="researcher", time=datetime.now(UTC)),
            ],
        )
        _patch_attribute(obj, "reasoning", "some reasoning")
        assert len(obj.attributes) == 2
        assert obj.attributes[1].name == "reasoning"
        assert obj.attributes[1].value == "some reasoning"

    def test_coerces_dict_value_to_json(self) -> None:
        from datetime import datetime

        obj = OcelObject(
            id="test-obj",
            type="tool_call",
            attributes=[
                OcelObjectAttribute(name="input", value="{}", time=datetime.now(UTC)),
            ],
        )
        _patch_attribute(obj, "input", {"key": "value"})
        assert obj.attributes[0].value == '{"key": "value"}'

    def test_coerces_none_to_empty_string(self) -> None:
        from datetime import datetime

        obj = OcelObject(
            id="test-obj",
            type="agent",
            attributes=[
                OcelObjectAttribute(name="output", value="old", time=datetime.now(UTC)),
            ],
        )
        _patch_attribute(obj, "output", None)
        assert obj.attributes[0].value == ""


class TestRewriteTimestamps:
    def test_no_events_for_run(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        # Should not raise for a nonexistent run
        _rewrite_timestamps(result.log, "run-9999")

    def test_timestamps_monotonically_increase(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        _rewrite_timestamps(result.log, "run-0000")
        run_events = [
            e
            for e in result.log.events
            if any(a.name == "run_id" and a.value == "run-0000" for a in e.attributes)
        ]
        times = [e.time for e in run_events]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]


class TestEnrichWithDeviations:
    def test_enrich_deviant_runs(self) -> None:
        """Enrich a log with deviations to exercise deviation-aware prompt paths."""
        result = generate("sequential", num_runs=5, noise_rate=1.0, seed=42)
        scenario = _make_scenario(runs=5)

        mock_client = MagicMock()
        mock_client.generate.return_value = _mock_response()
        mock_client.generate_queries.return_value = ["q1", "q2", "q3", "q4", "q5"]

        enrich_log(result.log, scenario, client=mock_client)

        # Should have been called for each step of each run
        assert mock_client.generate.call_count > 0

    def test_enrich_handles_llm_failure_gracefully(self) -> None:
        """Enricher should continue when LLM calls fail, returning failure count."""
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        scenario = _make_scenario(runs=2)

        mock_client = MagicMock()
        # First call succeeds, second fails, rest succeed
        mock_client.generate.side_effect = [
            _mock_response(),
            Exception("LLM failed"),
            _mock_response(),
            _mock_response(),
            _mock_response(),
            _mock_response(),
        ]
        mock_client.generate_queries.return_value = ["q1", "q2"]

        failed = enrich_log(result.log, scenario, client=mock_client)
        assert failed >= 1

    def test_enrich_query_expansion_failure_falls_back(self) -> None:
        """If generate_queries fails, enricher logs warning and falls back to cycling."""
        result = generate("sequential", num_runs=5, noise_rate=0.0, seed=42)
        scenario = DomainScenario(
            name="test",
            description="Test",
            pattern="sequential",
            runs=5,
            noise=0.0,
            seed=42,
            user_queries=["q1"],  # Fewer queries than runs
            agent_personas={"researcher": "R", "analyst": "A", "summarizer": "S"},
            tool_descriptions={},
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = _mock_response()
        mock_client.generate_queries.side_effect = Exception("API error")

        # Should not raise — falls back to cycling
        enrich_log(result.log, scenario, client=mock_client)


class TestEnrichParallelPattern:
    def test_enrich_parallel_pattern(self) -> None:
        """Exercise the parallel pattern enrichment path."""
        result = generate("parallel", num_runs=2, noise_rate=0.0, seed=42)
        scenario = DomainScenario(
            name="test-parallel",
            description="Test parallel domain",
            pattern="parallel",
            runs=2,
            noise=0.0,
            seed=42,
            user_queries=["parallel q1", "parallel q2"],
            agent_personas={},
            tool_descriptions={},
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = _mock_response()
        mock_client.generate_queries.return_value = ["pq1", "pq2"]

        enrich_log(result.log, scenario, client=mock_client)
        assert mock_client.generate.call_count > 0

    def test_enrich_supervisor_pattern(self) -> None:
        """Exercise the supervisor pattern enrichment path."""
        result = generate("supervisor", num_runs=2, noise_rate=0.0, seed=42)
        scenario = DomainScenario(
            name="test-supervisor",
            description="Test supervisor domain",
            pattern="supervisor",
            runs=2,
            noise=0.0,
            seed=42,
            user_queries=["supervisor q1", "supervisor q2"],
            agent_personas={},
            tool_descriptions={},
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = _mock_response()
        mock_client.generate_queries.return_value = ["sq1", "sq2"]

        enrich_log(result.log, scenario, client=mock_client)
        assert mock_client.generate.call_count > 0


class TestGetToolNamesForStep:
    def test_extracts_tool_names(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        from ocelgen.enrichment.enricher import _extract_steps_from_log

        steps = _extract_steps_from_log(result.log, "run-0000")
        # At least one step should have tool calls
        all_tools = []
        for step in steps:
            names = _get_tool_names_for_step(result.log, step)
            all_tools.extend(names)
        assert len(all_tools) > 0

    def test_empty_for_no_tools(self) -> None:
        result = generate("sequential", num_runs=1, noise_rate=0.0, seed=42)
        step = {
            "tool_call_ids": ["nonexistent-tool-id"],
        }
        names = _get_tool_names_for_step(result.log, step)
        assert names == []
