"""Sequential chain pattern: Research → Analyze → Summarize.

A simple linear pipeline where each agent completes before the next begins.
The researcher gathers information, the analyst processes it, and the
summarizer produces the final output.
"""

from __future__ import annotations

from ocelgen.models.langchain import AgentRole, LLMModel, ToolKind
from ocelgen.models.workflow import WorkflowEdge, WorkflowStep, WorkflowTemplate
from ocelgen.patterns.base import BasePattern


class SequentialPattern(BasePattern):
    @property
    def name(self) -> str:
        return "sequential"

    @property
    def description(self) -> str:
        return "Linear chain: Research → Analyze → Summarize"

    def build_template(self) -> WorkflowTemplate:
        steps = [
            WorkflowStep(
                id="research",
                name="Research",
                agent_role=AgentRole.RESEARCHER,
                model=LLMModel.GPT4O,
                tools=[ToolKind.WEB_SEARCH, ToolKind.FILE_READER],
                is_start=True,
                expected_llm_calls=2,
                expected_tool_calls=3,
            ),
            WorkflowStep(
                id="analyze",
                name="Analyze",
                agent_role=AgentRole.ANALYST,
                model=LLMModel.GPT4O,
                tools=[ToolKind.CALCULATOR, ToolKind.CODE_INTERPRETER],
                expected_llm_calls=2,
                expected_tool_calls=2,
            ),
            WorkflowStep(
                id="summarize",
                name="Summarize",
                agent_role=AgentRole.SUMMARIZER,
                model=LLMModel.CLAUDE_SONNET,
                tools=[],
                is_end=True,
                expected_llm_calls=1,
                expected_tool_calls=0,
            ),
        ]
        edges = [
            WorkflowEdge(source="research", target="analyze"),
            WorkflowEdge(source="analyze", target="summarize"),
        ]
        return WorkflowTemplate(
            name=self.name,
            description=self.description,
            steps=steps,
            edges=edges,
        )
