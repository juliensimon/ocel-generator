"""Tests for the domain scenarios module."""

import pytest

from ocelgen.scenarios import SCENARIO_REGISTRY, DomainScenario, get_scenario


class TestDomainScenario:
    def test_creation(self):
        scenario = DomainScenario(
            name="test-domain",
            description="A test domain",
            pattern="sequential",
            runs=10,
            noise=0.1,
            seed=42,
            user_queries=["Query A", "Query B", "Query C"],
            agent_personas={"researcher": "You are a researcher"},
            tool_descriptions={"web_search": "Search the web"},
        )
        assert scenario.name == "test-domain"
        assert scenario.description == "A test domain"
        assert scenario.pattern == "sequential"
        assert scenario.runs == 10
        assert scenario.noise == 0.1
        assert scenario.seed == 42
        assert len(scenario.user_queries) == 3
        assert len(scenario.agent_personas) == 1
        assert len(scenario.tool_descriptions) == 1

    def test_query_for_run_cycling(self):
        scenario = DomainScenario(
            name="test-domain",
            description="A test domain",
            pattern="sequential",
            runs=10,
            noise=0.1,
            seed=42,
            user_queries=["Query A", "Query B", "Query C"],
        )
        # First cycle
        assert scenario.query_for_run(0) == "Query A"
        assert scenario.query_for_run(1) == "Query B"
        assert scenario.query_for_run(2) == "Query C"
        # Second cycle (wraps around)
        assert scenario.query_for_run(3) == "Query A"
        assert scenario.query_for_run(4) == "Query B"
        assert scenario.query_for_run(5) == "Query C"
        # Large index: 99 % 3 == 0 → "Query A", 100 % 3 == 1 → "Query B"
        assert scenario.query_for_run(99) == "Query A"
        assert scenario.query_for_run(100) == "Query B"

    def test_query_for_run_cycling_simple(self):
        scenario = DomainScenario(
            name="test-domain",
            description="A test domain",
            pattern="sequential",
            runs=10,
            noise=0.1,
            seed=42,
            user_queries=["Q0", "Q1"],
        )
        assert scenario.query_for_run(0) == "Q0"
        assert scenario.query_for_run(1) == "Q1"
        assert scenario.query_for_run(2) == "Q0"
        assert scenario.query_for_run(3) == "Q1"
        assert scenario.query_for_run(10) == "Q0"
        assert scenario.query_for_run(11) == "Q1"


class TestRegistry:
    def test_10_domains(self):
        assert len(SCENARIO_REGISTRY) == 10

    def test_all_have_at_least_10_queries(self):
        for name, scenario in SCENARIO_REGISTRY.items():
            assert len(scenario.user_queries) >= 10, (
                f"Scenario '{name}' has only {len(scenario.user_queries)} queries"
            )

    def test_valid_patterns(self):
        valid_patterns = {"sequential", "supervisor", "parallel"}
        for name, scenario in SCENARIO_REGISTRY.items():
            assert scenario.pattern in valid_patterns, (
                f"Scenario '{name}' has invalid pattern '{scenario.pattern}'"
            )

    def test_unique_seeds(self):
        seeds = [scenario.seed for scenario in SCENARIO_REGISTRY.values()]
        assert len(seeds) == len(set(seeds)), "Scenario seeds are not unique"

    def test_get_scenario_works(self):
        for name in SCENARIO_REGISTRY:
            scenario = get_scenario(name)
            assert scenario.name == name

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent-domain")

    def test_all_have_at_least_2_personas(self):
        for name, scenario in SCENARIO_REGISTRY.items():
            assert len(scenario.agent_personas) >= 2, (
                f"Scenario '{name}' has only {len(scenario.agent_personas)} personas"
            )
