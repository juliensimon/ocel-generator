"""Parallel fan-out pattern: multiple agents work concurrently, then aggregate.

A planner splits work into parallel tasks, multiple workers execute
simultaneously (overlapping timestamps), and an aggregator combines results.
"""

from __future__ import annotations

from ocelgen.models.langchain import AgentRole, LLMModel, ToolKind
from ocelgen.models.workflow import WorkflowEdge, WorkflowStep, WorkflowTemplate
from ocelgen.patterns.base import BasePattern


class ParallelPattern(BasePattern):
    @property
    def name(self) -> str:
        return "parallel"

    @property
    def description(self) -> str:
        return "Fan-out to concurrent agents, then aggregate results"

    def build_template(self) -> WorkflowTemplate:
        steps = [
            WorkflowStep(
                id="split",
                name="Split Work",
                agent_role=AgentRole.PLANNER,
                model=LLMModel.GPT4O,
                tools=[],
                is_start=True,
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
            WorkflowStep(
                id="worker_a",
                name="Worker A",
                agent_role=AgentRole.RESEARCHER,
                model=LLMModel.GPT4O,
                tools=[ToolKind.WEB_SEARCH],
                parallel_group="workers",
                expected_llm_calls=1,
                expected_tool_calls=1,
            ),
            WorkflowStep(
                id="worker_b",
                name="Worker B",
                agent_role=AgentRole.ANALYST,
                model=LLMModel.CLAUDE_SONNET,
                tools=[ToolKind.CALCULATOR, ToolKind.DATABASE_QUERY],
                parallel_group="workers",
                expected_llm_calls=1,
                expected_tool_calls=2,
            ),
            WorkflowStep(
                id="worker_c",
                name="Worker C",
                agent_role=AgentRole.WRITER,
                model=LLMModel.CLAUDE_HAIKU,
                tools=[ToolKind.TEXT_SPLITTER],
                parallel_group="workers",
                expected_llm_calls=1,
                expected_tool_calls=1,
            ),
            WorkflowStep(
                id="aggregate",
                name="Aggregate Results",
                agent_role=AgentRole.AGGREGATOR,
                model=LLMModel.GPT4O,
                tools=[],
                is_end=True,
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
        ]
        edges = [
            WorkflowEdge(source="split", target="worker_a"),
            WorkflowEdge(source="split", target="worker_b"),
            WorkflowEdge(source="split", target="worker_c"),
            WorkflowEdge(source="worker_a", target="aggregate"),
            WorkflowEdge(source="worker_b", target="aggregate"),
            WorkflowEdge(source="worker_c", target="aggregate"),
        ]
        return WorkflowTemplate(
            name=self.name,
            description=self.description,
            steps=steps,
            edges=edges,
        )
