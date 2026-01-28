"""Supervisor/worker pattern: a supervisor delegates tasks to specialist workers.

The supervisor agent receives the task, decides which worker to route to,
the worker executes, and the supervisor aggregates results. This pattern
exercises routing_decided events that the sequential pattern does not.
"""

from __future__ import annotations

from ocelgen.models.langchain import AgentRole, LLMModel, ToolKind
from ocelgen.models.workflow import WorkflowEdge, WorkflowStep, WorkflowTemplate
from ocelgen.patterns.base import BasePattern


class SupervisorPattern(BasePattern):
    @property
    def name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return "Supervisor delegates to specialist workers, then aggregates"

    def build_template(self) -> WorkflowTemplate:
        steps = [
            WorkflowStep(
                id="plan",
                name="Plan",
                agent_role=AgentRole.SUPERVISOR,
                model=LLMModel.GPT4O,
                tools=[],
                is_start=True,
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
            WorkflowStep(
                id="research_task",
                name="Research Task",
                agent_role=AgentRole.RESEARCHER,
                model=LLMModel.GPT4O,
                tools=[ToolKind.WEB_SEARCH, ToolKind.FILE_READER],
                expected_llm_calls=2,
                expected_tool_calls=2,
            ),
            WorkflowStep(
                id="code_task",
                name="Code Task",
                agent_role=AgentRole.CODER,
                model=LLMModel.CLAUDE_SONNET,
                tools=[ToolKind.CODE_INTERPRETER, ToolKind.FILE_READER],
                expected_llm_calls=2,
                expected_tool_calls=2,
            ),
            WorkflowStep(
                id="review_task",
                name="Review Task",
                agent_role=AgentRole.REVIEWER,
                model=LLMModel.GPT4O,
                tools=[],
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
            WorkflowStep(
                id="aggregate",
                name="Aggregate",
                agent_role=AgentRole.SUPERVISOR,
                model=LLMModel.GPT4O,
                tools=[],
                is_end=True,
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
        ]
        edges = [
            WorkflowEdge(source="plan", target="research_task", condition="route_research"),
            WorkflowEdge(source="plan", target="code_task", condition="route_code"),
            WorkflowEdge(source="plan", target="review_task", condition="route_review"),
            WorkflowEdge(source="research_task", target="aggregate"),
            WorkflowEdge(source="code_task", target="aggregate"),
            WorkflowEdge(source="review_task", target="aggregate"),
        ]
        return WorkflowTemplate(
            name=self.name,
            description=self.description,
            steps=steps,
            edges=edges,
        )
