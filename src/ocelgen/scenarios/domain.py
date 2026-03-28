"""DomainScenario dataclass — defines a domain for enriched trace generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainScenario:
    name: str
    description: str
    pattern: str  # "sequential", "supervisor", "parallel"
    runs: int
    noise: float
    seed: int
    user_queries: list[str] = field(default_factory=list)
    agent_personas: dict[str, str] = field(default_factory=dict)
    tool_descriptions: dict[str, str] = field(default_factory=dict)

    def query_for_run(self, run_index: int) -> str:
        """Return the user query for a given run index, cycling through the bank."""
        return self.user_queries[run_index % len(self.user_queries)]
