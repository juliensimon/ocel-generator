"""Domain scenario definitions for LLM-enriched trace generation."""

from ocelgen.scenarios.domain import DomainScenario
from ocelgen.scenarios.loader import build_registry, load_domains_from_yaml
from ocelgen.scenarios.registry import SCENARIO_REGISTRY, get_scenario

__all__ = [
    "DomainScenario",
    "SCENARIO_REGISTRY",
    "build_registry",
    "get_scenario",
    "load_domains_from_yaml",
]
