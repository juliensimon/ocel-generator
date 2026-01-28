"""Serialize a WorkflowTemplate to JSON for conformance checking reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ocelgen.models.workflow import WorkflowTemplate


def template_to_dict(template: WorkflowTemplate) -> dict[str, Any]:
    """Convert a WorkflowTemplate to a serializable dict."""
    return template.to_dict()


def write_normative_model(template: WorkflowTemplate, path: Path) -> None:
    """Write the normative workflow model to a JSON file."""
    data = template_to_dict(template)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
