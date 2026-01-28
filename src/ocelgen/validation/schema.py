"""Validate OCEL 2.0 JSON output against the official schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


def _load_schema() -> dict[str, Any]:
    """Load the vendored OCEL 2.0 JSON schema."""
    base = Path(__file__).resolve().parent.parent.parent.parent
    schema_path = base / "schemas" / "ocel20-schema-json.json"
    with open(schema_path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
    return result


def validate_ocel_dict(data: dict[str, Any]) -> list[str]:
    """Validate an OCEL 2.0 dict against the schema.

    Returns a list of error messages (empty if valid).
    """
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in error.absolute_path)
        errors.append(f"{path}: {error.message}" if path else error.message)
    return errors


def validate_ocel_file(path: Path) -> list[str]:
    """Validate an OCEL 2.0 JSON file against the schema.

    Returns a list of error messages (empty if valid).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return validate_ocel_dict(data)
