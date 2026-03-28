"""Domain scenario definitions for LLM-enriched trace generation."""

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario

__all__ = ["DomainScenario", "SCENARIO_REGISTRY", "get_scenario"]
