"""Tests for HF upload utilities."""

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.upload.readme import generate_dataset_card


def _make_scenario() -> DomainScenario:
    return DomainScenario(
        name="test-domain",
        description="A test domain for unit tests",
        pattern="sequential",
        runs=10,
        noise=0.2,
        seed=42,
        user_queries=["query one"],
        agent_personas={"researcher": "A researcher"},
        tool_descriptions={"web_search": "Search"},
    )


class TestDatasetCard:
    def test_card_contains_domain_name(self) -> None:
        card = generate_dataset_card(
            scenarios=[_make_scenario()],
            namespace="testuser",
            domain_stats={"test-domain": {"num_events": 500, "num_objects": 200}},
        )
        assert "test-domain" in card

    def test_card_contains_schema_table(self) -> None:
        card = generate_dataset_card(
            scenarios=[_make_scenario()],
            namespace="testuser",
            domain_stats={"test-domain": {"num_events": 500, "num_objects": 200}},
        )
        assert "event_id" in card
        assert "event_type" in card
        assert "prompt" in card

    def test_card_contains_yaml_frontmatter(self) -> None:
        card = generate_dataset_card(
            scenarios=[_make_scenario()],
            namespace="testuser",
            domain_stats={"test-domain": {"num_events": 500, "num_objects": 200}},
        )
        assert card.startswith("---")
        assert "configs" in card

    def test_card_contains_usage_example(self) -> None:
        card = generate_dataset_card(
            scenarios=[_make_scenario()],
            namespace="testuser",
            domain_stats={"test-domain": {"num_events": 500, "num_objects": 200}},
        )
        assert "load_dataset" in card
        assert "testuser/open-agent-traces" in card


from pathlib import Path

from ocelgen.upload.hf_upload import build_repo_name, prepare_domain_files, prepare_upload_files


class TestBuildRepoName:
    def test_basic(self) -> None:
        assert build_repo_name("juliensimon", "customer-support-triage") == (
            "juliensimon/agent-traces-customer-support-triage"
        )


class TestPrepareDomainFiles:
    def test_creates_config_files(self, tmp_path: Path) -> None:
        from ocelgen.generation.engine import generate
        from ocelgen.upload.flatten import flatten_log

        scenario = _make_scenario()
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test-domain")

        files = prepare_domain_files(
            rows=rows,
            log=result.log,
            template=result.template,
            result=result,
            scenario=scenario,
            output_dir=tmp_path,
            seed=42,
        )

        assert (tmp_path / "data" / "test-domain" / "train.parquet").exists()
        assert (tmp_path / "ocel" / "test-domain" / "output.jsonocel").exists()
        assert (tmp_path / "ocel" / "test-domain" / "normative_model.json").exists()
        assert (tmp_path / "ocel" / "test-domain" / "manifest.json").exists()
        assert len(files) == 4


class TestPrepareUploadFiles:
    def test_creates_parquet_and_ocel_files(self, tmp_path: Path) -> None:
        from ocelgen.generation.engine import generate
        from ocelgen.upload.flatten import flatten_log

        scenario = _make_scenario()
        result = generate("sequential", num_runs=2, noise_rate=0.0, seed=42)
        rows = flatten_log(result.log, domain="test")

        files = prepare_upload_files(
            rows=rows,
            log=result.log,
            template=result.template,
            result=result,
            scenario=scenario,
            namespace="testuser",
            output_dir=tmp_path,
            seed=42,
        )

        assert (tmp_path / "data" / "train.parquet").exists()
        assert (tmp_path / "ocel" / "output.jsonocel").exists()
        assert (tmp_path / "ocel" / "normative_model.json").exists()
        assert (tmp_path / "ocel" / "manifest.json").exists()
        assert (tmp_path / "README.md").exists()
        assert len(files) == 5
