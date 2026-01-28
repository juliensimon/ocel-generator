"""Tests for OCEL 2.0 models, serialization, and schema validation."""

from datetime import UTC, datetime

from ocelgen.export.ocel_json import ocel_log_to_dict, write_ocel_json
from ocelgen.models.ocel import (
    OcelAttributeDefinition,
    OcelEvent,
    OcelEventAttribute,
    OcelLog,
    OcelObject,
    OcelObjectAttribute,
    OcelRelationship,
)
from ocelgen.validation.schema import validate_ocel_dict, validate_ocel_file


def _make_minimal_log() -> OcelLog:
    """Create a small but complete OCEL 2.0 log for testing."""
    now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    log = OcelLog()

    # Register types
    log.add_event_type("run_started", [
        OcelAttributeDefinition(name="run_id", type="string"),
    ])
    log.add_event_type("agent_invoked", [
        OcelAttributeDefinition(name="run_id", type="string"),
    ])
    log.add_object_type("run", [
        OcelAttributeDefinition(name="status", type="string"),
    ])
    log.add_object_type("agent", [
        OcelAttributeDefinition(name="role", type="string"),
    ])

    # Create objects
    run_obj = OcelObject(
        id="run-001",
        type="run",
        attributes=[
            OcelObjectAttribute(name="status", value="completed", time=now),
        ],
    )
    agent_obj = OcelObject(
        id="agent-researcher",
        type="agent",
        attributes=[
            OcelObjectAttribute(name="role", value="researcher", time=now),
        ],
    )
    log.add_object(run_obj)
    log.add_object(agent_obj)

    # Create events
    log.add_event(OcelEvent(
        id="evt-001",
        type="run_started",
        time=now,
        attributes=[
            OcelEventAttribute(name="run_id", value="run-001"),
        ],
        relationships=[
            OcelRelationship(objectId="run-001", qualifier="started"),
        ],
    ))
    log.add_event(OcelEvent(
        id="evt-002",
        type="agent_invoked",
        time=now,
        attributes=[
            OcelEventAttribute(name="run_id", value="run-001"),
        ],
        relationships=[
            OcelRelationship(objectId="run-001", qualifier="part_of"),
            OcelRelationship(objectId="agent-researcher", qualifier="invoked"),
        ],
    ))

    return log


class TestOcelModels:
    def test_create_minimal_log(self) -> None:
        log = _make_minimal_log()
        assert len(log.events) == 2
        assert len(log.objects) == 2
        assert len(log.eventTypes) == 2
        assert len(log.objectTypes) == 2

    def test_add_object_deduplicates(self) -> None:
        log = _make_minimal_log()
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        duplicate = OcelObject(id="run-001", type="run", attributes=[
            OcelObjectAttribute(name="status", value="completed", time=now),
        ])
        log.add_object(duplicate)
        assert len(log.objects) == 2  # Not 3

    def test_add_event_type_deduplicates(self) -> None:
        log = _make_minimal_log()
        log.add_event_type("run_started", [])
        assert len(log.eventTypes) == 2

    def test_add_object_type_deduplicates(self) -> None:
        log = _make_minimal_log()
        log.add_object_type("run", [])
        assert len(log.objectTypes) == 2

    def test_merge_logs(self) -> None:
        log1 = _make_minimal_log()
        log2 = _make_minimal_log()
        # Give log2 unique event/object IDs
        for e in log2.events:
            e.id = e.id + "-b"
        for o in log2.objects:
            o.id = o.id + "-b"

        log1.merge(log2)
        assert len(log1.events) == 4
        assert len(log1.objects) == 4
        # Types should still be deduplicated
        assert len(log1.eventTypes) == 2
        assert len(log1.objectTypes) == 2


class TestOcelSerialization:
    def test_serialize_to_dict(self) -> None:
        log = _make_minimal_log()
        data = ocel_log_to_dict(log)

        assert "eventTypes" in data
        assert "objectTypes" in data
        assert "events" in data
        assert "objects" in data
        assert len(data["events"]) == 2
        assert len(data["objects"]) == 2

    def test_event_time_is_iso_string(self) -> None:
        log = _make_minimal_log()
        data = ocel_log_to_dict(log)
        # Pydantic serializes datetime as ISO string in json mode
        assert isinstance(data["events"][0]["time"], str)
        assert "2025-01-15" in data["events"][0]["time"]

    def test_attribute_values_are_strings(self) -> None:
        log = _make_minimal_log()
        data = ocel_log_to_dict(log)
        for event in data["events"]:
            for attr in event.get("attributes", []):
                assert isinstance(attr["value"], str)


class TestOcelSchemaValidation:
    def test_valid_log_passes_schema(self) -> None:
        log = _make_minimal_log()
        data = ocel_log_to_dict(log)
        errors = validate_ocel_dict(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_empty_log_passes_schema(self) -> None:
        log = OcelLog()
        data = ocel_log_to_dict(log)
        errors = validate_ocel_dict(data)
        assert errors == []

    def test_missing_required_field_fails(self) -> None:
        # Missing 'events' key
        data = {"eventTypes": [], "objectTypes": [], "objects": []}
        errors = validate_ocel_dict(data)
        assert len(errors) > 0
        assert any("events" in e for e in errors)

    def test_event_missing_id_fails(self) -> None:
        data = {
            "eventTypes": [],
            "objectTypes": [],
            "events": [{"type": "test", "time": "2025-01-15T10:00:00Z"}],
            "objects": [],
        }
        errors = validate_ocel_dict(data)
        assert len(errors) > 0

    def test_event_missing_time_fails(self) -> None:
        data = {
            "eventTypes": [],
            "objectTypes": [],
            "events": [{"id": "e1", "type": "test"}],
            "objects": [],
        }
        errors = validate_ocel_dict(data)
        assert len(errors) > 0

    def test_file_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log = _make_minimal_log()
        path = tmp_path / "test.jsonocel"
        write_ocel_json(log, path)
        errors = validate_ocel_file(path)
        assert errors == [], f"Roundtrip validation errors: {errors}"
