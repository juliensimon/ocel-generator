"""Tests for HF upload utilities."""

from ocelgen.upload.readme import generate_dataset_card
from ocelgen.scenarios.domain import DomainScenario


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
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "test-domain" in card

    def test_card_contains_schema_table(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "event_id" in card
        assert "event_type" in card
        assert "prompt" in card

    def test_card_contains_yaml_frontmatter(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert card.startswith("---")
        assert "dataset_info" in card

    def test_card_contains_usage_example(self) -> None:
        card = generate_dataset_card(
            scenario=_make_scenario(),
            namespace="testuser",
            num_events=500,
            num_objects=200,
        )
        assert "load_dataset" in card
        assert "testuser/agent-traces-test-domain" in card
