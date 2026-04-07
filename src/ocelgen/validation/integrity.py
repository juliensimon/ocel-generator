"""Referential integrity validation for OCEL 2.0 logs.

Checks that go beyond JSON schema validation:
- Every relationship objectId points to an existing object
- Every event type is declared in eventTypes
- Every object type is declared in objectTypes
- No duplicate event or object IDs
- Event attributes match their declared eventType schema
- Object attributes match their declared objectType schema
"""

from __future__ import annotations

from ocelgen.models.ocel import OcelLog


def validate_referential_integrity(log: OcelLog) -> list[str]:
    """Validate that all references in the log resolve to existing entities.

    Returns a list of error messages (empty if all references are valid).
    """
    errors: list[str] = []

    # Build lookup sets
    object_ids = {obj.id for obj in log.objects}
    event_type_names = {et.name for et in log.eventTypes}
    object_type_names = {ot.name for ot in log.objectTypes}

    # Check for duplicate event IDs
    seen_event_ids: set[str] = set()
    for event in log.events:
        if event.id in seen_event_ids:
            errors.append(f"Duplicate event ID: '{event.id}'")
        seen_event_ids.add(event.id)

    # Check for duplicate object IDs
    seen_object_ids: set[str] = set()
    for obj in log.objects:
        if obj.id in seen_object_ids:
            errors.append(f"Duplicate object ID: '{obj.id}'")
        seen_object_ids.add(obj.id)

    # Check event types are declared
    for event in log.events:
        if event.type not in event_type_names:
            errors.append(f"Event '{event.id}' has undeclared type '{event.type}'")

    # Check object types are declared
    for obj in log.objects:
        if obj.type not in object_type_names:
            errors.append(f"Object '{obj.id}' has undeclared type '{obj.type}'")

    # Check event relationship targets exist
    for event in log.events:
        for rel in event.relationships:
            if rel.objectId not in object_ids:
                errors.append(
                    f"Event '{event.id}' references non-existent object "
                    f"'{rel.objectId}' (qualifier: '{rel.qualifier}')"
                )

    # Check object relationship targets exist
    for obj in log.objects:
        for rel in obj.relationships:
            if rel.objectId not in object_ids:
                errors.append(
                    f"Object '{obj.id}' references non-existent object "
                    f"'{rel.objectId}' (qualifier: '{rel.qualifier}')"
                )

    return errors


def validate_type_attributes(log: OcelLog) -> list[str]:
    """Validate that event/object attributes match their declared type schemas.

    Checks that every attribute on an event or object instance has a
    corresponding declaration in its eventType or objectType.

    Returns a list of error messages (empty if all attributes are declared).
    """
    errors: list[str] = []

    # Build attribute schema lookups
    event_type_attrs: dict[str, set[str]] = {}
    for et in log.eventTypes:
        event_type_attrs[et.name] = {a.name for a in et.attributes}

    object_type_attrs: dict[str, set[str]] = {}
    for ot in log.objectTypes:
        object_type_attrs[ot.name] = {a.name for a in ot.attributes}

    # Check event attributes
    for event in log.events:
        declared = event_type_attrs.get(event.type, set())
        for attr in event.attributes:
            if attr.name not in declared:
                errors.append(
                    f"Event '{event.id}' (type '{event.type}') has "
                    f"undeclared attribute '{attr.name}'"
                )

    # Check object attributes
    for obj in log.objects:
        declared_obj = object_type_attrs.get(obj.type, set())
        for obj_attr in obj.attributes:
            if obj_attr.name not in declared_obj:
                errors.append(
                    f"Object '{obj.id}' (type '{obj.type}') has "
                    f"undeclared attribute '{obj_attr.name}'"
                )

    return errors
