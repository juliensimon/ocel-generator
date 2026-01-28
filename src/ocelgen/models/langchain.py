"""LangChain domain enums: agent roles, tool kinds, LLM models."""

from __future__ import annotations

from enum import Enum


class AgentRole(str, Enum):
    """Roles an agent can play in a multi-agent workflow."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SUMMARIZER = "summarizer"
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    AGGREGATOR = "aggregator"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    WRITER = "writer"


class ToolKind(str, Enum):
    """Categories of tools agents can invoke."""

    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    CODE_INTERPRETER = "code_interpreter"
    FILE_READER = "file_reader"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    TEXT_SPLITTER = "text_splitter"
    VECTOR_SEARCH = "vector_search"


class LLMModel(str, Enum):
    """LLM models that agents can use."""

    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-3-5-sonnet"
    CLAUDE_HAIKU = "claude-3-5-haiku"
