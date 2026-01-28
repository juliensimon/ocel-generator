"""Workflow template models defining the normative process for a pattern.

A WorkflowTemplate is the "ground truth" process model that conformant runs
should follow. It consists of steps connected by edges, forming a directed
graph. Each step declares which agent role handles it and what tools it uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ocelgen.models.langchain import AgentRole, LLMModel, ToolKind


@dataclass
class WorkflowStep:
    """A single step in a workflow template."""

    id: str
    name: str
    agent_role: AgentRole
    model: LLMModel
    tools: list[ToolKind] = field(default_factory=list)
    is_start: bool = False
    is_end: bool = False
    # For parallel patterns: steps that can execute concurrently
    parallel_group: str | None = None
    # Expected number of LLM calls in this step
    expected_llm_calls: int = 1
    # Expected number of tool calls in this step
    expected_tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "agent_role": self.agent_role.value,
            "model": self.model.value,
            "tools": [t.value for t in self.tools],
            "is_start": self.is_start,
            "is_end": self.is_end,
            "parallel_group": self.parallel_group,
            "expected_llm_calls": self.expected_llm_calls,
            "expected_tool_calls": self.expected_tool_calls,
        }


@dataclass
class WorkflowEdge:
    """A directed edge between two workflow steps."""

    source: str  # step id
    target: str  # step id
    condition: str | None = None  # e.g., "success", "failure", routing label

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source, "target": self.target}
        if self.condition:
            d["condition"] = self.condition
        return d


@dataclass
class WorkflowTemplate:
    """Complete normative workflow definition for a pattern."""

    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)
    edges: list[WorkflowEdge] = field(default_factory=list)

    @property
    def start_step(self) -> WorkflowStep:
        for s in self.steps:
            if s.is_start:
                return s
        raise ValueError("No start step defined in workflow")

    @property
    def end_steps(self) -> list[WorkflowStep]:
        return [s for s in self.steps if s.is_end]

    def step_by_id(self, step_id: str) -> WorkflowStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"No step with id '{step_id}'")

    def successors(self, step_id: str) -> list[WorkflowStep]:
        """Get all steps reachable from the given step."""
        return [self.step_by_id(e.target) for e in self.edges if e.source == step_id]

    def predecessors(self, step_id: str) -> list[WorkflowStep]:
        """Get all steps that lead to the given step."""
        return [self.step_by_id(e.source) for e in self.edges if e.target == step_id]

    def topological_order(self) -> list[WorkflowStep]:
        """Return steps in topological order (respecting parallel groups)."""
        visited: set[str] = set()
        order: list[WorkflowStep] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            for pred in self.predecessors(step_id):
                visit(pred.id)
            order.append(self.step_by_id(step_id))

        for step in self.steps:
            visit(step.id)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "edges": [e.to_dict() for e in self.edges],
        }
