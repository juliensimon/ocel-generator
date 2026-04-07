"""Validate the live Hugging Face dataset against all semantic validators.

These tests download the published dataset from
https://huggingface.co/datasets/juliensimon/open-agent-traces
and run every validation layer against every domain.

They also serve as **code samples** showing how to load, inspect,
and validate OCEL 2.0 agent traces from Hugging Face.

Requires network access and the conformance extra:
    uv run --extra conformance python -m pytest tests/test_hf_dataset.py -v

Excluded from default test runs. Run explicitly or with:
    uv run pytest -m network
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.network

pm4py = pytest.importorskip(
    "pm4py", reason="pm4py not installed (install with: uv sync --extra conformance)"
)


from huggingface_hub import hf_hub_download
from pydantic import TypeAdapter

from ocelgen.models.ocel import OcelLog
from ocelgen.patterns.parallel import ParallelPattern
from ocelgen.patterns.sequential import SequentialPattern
from ocelgen.patterns.supervisor import SupervisorPattern
from ocelgen.scenarios.loader import build_registry
from ocelgen.validation.conformance import validate_workflow_conformance
from ocelgen.validation.integrity import (
    validate_referential_integrity,
    validate_type_attributes,
)
from ocelgen.validation.schema import validate_ocel_dict
from ocelgen.validation.temporal import validate_temporal_order

REPO_ID = "juliensimon/open-agent-traces"

_PATTERN_MAP = {
    "sequential": SequentialPattern,
    "supervisor": SupervisorPattern,
    "parallel": ParallelPattern,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_registry = build_registry(None)
_ALL_DOMAINS = sorted(_registry.keys())


def _load_ocel(domain: str) -> tuple[dict, OcelLog]:
    """Download and parse the OCEL file for a domain."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"ocel/{domain}/output.jsonocel",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log = TypeAdapter(OcelLog).validate_python(data)
    return data, log


# ---------------------------------------------------------------------------
# 1. JSON Schema validation — every domain
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    """Every OCEL file must pass the OCEL 2.0 JSON schema."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_schema(self, domain: str) -> None:
        data, _ = _load_ocel(domain)
        errors = validate_ocel_dict(data)
        assert errors == [], f"{domain} schema errors:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# 2. Referential integrity — every domain
# ---------------------------------------------------------------------------


class TestReferentialIntegrity:
    """Every objectId in a relationship must resolve to an existing object."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_integrity(self, domain: str) -> None:
        _, log = _load_ocel(domain)
        errors = validate_referential_integrity(log)
        assert errors == [], f"{domain} integrity errors:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# 3. Type attribute declarations — every domain
# ---------------------------------------------------------------------------


class TestTypeAttributes:
    """Every attribute on an instance must be declared in its type schema."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_attributes(self, domain: str) -> None:
        _, log = _load_ocel(domain)
        errors = validate_type_attributes(log)
        assert errors == [], f"{domain} attribute errors:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# 4. Temporal ordering — every domain
# ---------------------------------------------------------------------------


class TestTemporalOrder:
    """Conformant runs must have correct temporal ordering."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_conformant_temporal_order(self, domain: str) -> None:
        """Only conformant runs are checked — deviant runs intentionally
        violate ordering via strategies like SwappedOrder.

        Parallel patterns interleave sequence numbers across workers by
        design, so sequence monotonicity errors are expected and filtered.
        """
        scenario = _registry[domain]
        _, log = _load_ocel(domain)

        # Identify deviant runs
        deviant_run_ids = set()
        for obj in log.objects:
            if obj.type == "run":
                is_conformant = any(
                    a.name == "is_conformant" and a.value == "true" for a in obj.attributes
                )
                if not is_conformant:
                    deviant_run_ids.add(obj.id)

        errors = validate_temporal_order(log)

        # Filter out violations from deviant runs — they're expected
        conformant_errors = [e for e in errors if not any(rid in e for rid in deviant_run_ids)]

        # Parallel patterns interleave sequence numbers across workers
        if scenario.pattern == "parallel":
            conformant_errors = [e for e in conformant_errors if "sequence" not in e]

        assert conformant_errors == [], (
            f"{domain} temporal errors in conformant runs:\n" + "\n".join(conformant_errors)
        )


# ---------------------------------------------------------------------------
# 5. Workflow conformance — every domain
# ---------------------------------------------------------------------------


class TestWorkflowConformance:
    """Conformant runs must follow the normative workflow template."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_conformance(self, domain: str) -> None:
        scenario = _registry[domain]
        _, log = _load_ocel(domain)
        template = _PATTERN_MAP[scenario.pattern]().build_template()
        errors = validate_workflow_conformance(log, template)
        assert errors == [], f"{domain} conformance errors:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# 6. PM4Py round-trip — every domain
# ---------------------------------------------------------------------------


class TestPM4PyRoundTrip:
    """Every OCEL file must load in pm4py with correct event/object counts."""

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_loads_and_counts_match(self, domain: str) -> None:
        """Code sample: load an OCEL trace from HF and query it with pm4py."""
        data, _ = _load_ocel(domain)

        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"ocel/{domain}/output.jsonocel",
            repo_type="dataset",
        )
        ocel = pm4py.read.read_ocel2_json(path)

        assert len(ocel.events) == len(data["events"]), (
            f"{domain}: pm4py read {len(ocel.events)} events, expected {len(data['events'])}"
        )

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_event_types_present(self, domain: str) -> None:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"ocel/{domain}/output.jsonocel",
            repo_type="dataset",
        )
        ocel = pm4py.read.read_ocel2_json(path)
        event_types = set(ocel.events["ocel:activity"].unique())
        # Every domain must have these core event types
        assert "run_started" in event_types
        assert "run_completed" in event_types
        assert "agent_invoked" in event_types
        assert "agent_completed" in event_types

    @pytest.mark.parametrize("domain", _ALL_DOMAINS)
    def test_object_types_present(self, domain: str) -> None:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"ocel/{domain}/output.jsonocel",
            repo_type="dataset",
        )
        ocel = pm4py.read.read_ocel2_json(path)
        object_types = set(ocel.objects["ocel:type"].unique())
        assert "run" in object_types
        assert "agent" in object_types
        assert "llm_call" in object_types


# ---------------------------------------------------------------------------
# 7. Cross-domain consistency
# ---------------------------------------------------------------------------


class TestCrossDomain:
    """Cross-domain invariants that must hold across the entire dataset."""

    def test_total_event_count(self) -> None:
        """The README claims 17,019 events. Verify."""
        total = 0
        for domain in _ALL_DOMAINS:
            data, _ = _load_ocel(domain)
            total += len(data["events"])
        assert total == 17_019, f"Expected 17,019 total events, got {total}"

    def test_all_10_domains_present(self) -> None:
        assert len(_ALL_DOMAINS) == 10

    def test_all_3_patterns_represented(self) -> None:
        patterns = {_registry[d].pattern for d in _ALL_DOMAINS}
        assert patterns == {"sequential", "supervisor", "parallel"}

    def test_schema_columns_match_flatten(self) -> None:
        """Code sample: load via HF datasets and verify column schema."""
        from datasets import load_dataset

        ds = load_dataset(REPO_ID, "customer-support-triage")
        columns = set(ds["train"].column_names)
        expected = {
            "event_id",
            "event_type",
            "timestamp",
            "run_id",
            "sequence_number",
            "is_deviation",
            "deviation_type",
            "step_id",
            "agent_role",
            "model_name",
            "prompt",
            "completion",
            "tool_name",
            "tool_input",
            "tool_output",
            "message_content",
            "reasoning",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "cost_usd",
            "is_conformant",
            "pattern",
            "domain",
            "user_query",
        }
        assert columns == expected, f"Missing: {expected - columns}, Extra: {columns - expected}"

    def test_manifest_ground_truth_consistent(self) -> None:
        """Code sample: cross-check manifest.json against OCEL run objects."""
        for domain in _ALL_DOMAINS:
            manifest_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"ocel/{domain}/manifest.json",
                repo_type="dataset",
            )
            with open(manifest_path) as f:
                manifest = json.load(f)

            _, log = _load_ocel(domain)

            # Check run counts match
            run_objs = [o for o in log.objects if o.type == "run"]
            assert len(run_objs) == manifest["total_runs"], (
                f"{domain}: {len(run_objs)} run objects vs {manifest['total_runs']} in manifest"
            )

            # Check conformant/deviant split
            conformant_in_log = sum(
                1
                for o in run_objs
                if any(a.name == "is_conformant" and a.value == "true" for a in o.attributes)
            )
            assert conformant_in_log == manifest["conformant_runs"], (
                f"{domain}: {conformant_in_log} conformant in log vs "
                f"{manifest['conformant_runs']} in manifest"
            )

    def test_normative_model_matches_pattern(self) -> None:
        """Verify each domain's normative_model.json matches its pattern template."""
        for domain in _ALL_DOMAINS:
            scenario = _registry[domain]
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"ocel/{domain}/normative_model.json",
                repo_type="dataset",
            )
            with open(model_path) as f:
                model = json.load(f)

            template = _PATTERN_MAP[scenario.pattern]().build_template()
            expected = template.to_dict()

            assert model["name"] == expected["name"], (
                f"{domain}: normative model name '{model['name']}' != expected '{expected['name']}'"
            )
            assert len(model["steps"]) == len(expected["steps"]), (
                f"{domain}: {len(model['steps'])} steps in normative model "
                f"vs {len(expected['steps'])} expected"
            )
