"""Abstract base class for multi-agent workflow patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ocelgen.models.workflow import WorkflowTemplate


class BasePattern(ABC):
    """Defines a multi-agent workflow pattern and its normative model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the pattern (e.g., 'sequential')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the pattern."""

    @abstractmethod
    def build_template(self) -> WorkflowTemplate:
        """Construct the normative workflow template for this pattern."""
