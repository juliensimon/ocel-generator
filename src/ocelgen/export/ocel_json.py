"""Serialize an OcelLog to OCEL 2.0 JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ocelgen.models.ocel import OcelLog


def ocel_log_to_dict(log: OcelLog) -> dict[str, Any]:
    """Convert an OcelLog to a dict matching the OCEL 2.0 JSON schema."""
    return log.to_serializable()


def write_ocel_json(log: OcelLog, path: Path) -> None:
    """Write an OcelLog to a .jsonocel file."""
    data = ocel_log_to_dict(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
