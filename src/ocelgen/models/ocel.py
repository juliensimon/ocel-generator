"""OCEL 2.0 Pydantic models matching the official JSON schema.

The OCEL 2.0 standard defines four top-level collections:
- eventTypes / objectTypes: schema declarations with typed attributes
- events / objects: actual instances referencing those types

All attribute values are serialized as strings in the JSON output.
Object attributes are time-stamped to support attribute change tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OcelAttributeDefinition(BaseModel):
    """Schema-level attribute declaration (in eventTypes / objectTypes)."""

    name: str
    type: str  # "string", "integer", "float", "boolean", "time"


class OcelEventType(BaseModel):
    """Declaration of an event type and its attribute schema."""

    name: str
    attributes: list[OcelAttributeDefinition] = Field(default_factory=list)


class OcelObjectType(BaseModel):
    """Declaration of an object type and its attribute schema."""

    name: str
    attributes: list[OcelAttributeDefinition] = Field(default_factory=list)


class OcelRelationship(BaseModel):
    """Links an event to an object (or an object to another object)."""

    objectId: str
    qualifier: str


class OcelEventAttribute(BaseModel):
    """An attribute value on an event instance (no timestamp)."""

    name: str
    value: str


class OcelObjectAttribute(BaseModel):
    """A time-stamped attribute value on an object instance."""

    name: str
    value: str
    time: datetime


class OcelEvent(BaseModel):
    """A single event instance in the OCEL 2.0 log."""

    id: str
    type: str
    time: datetime
    attributes: list[OcelEventAttribute] = Field(default_factory=list)
    relationships: list[OcelRelationship] = Field(default_factory=list)


class OcelObject(BaseModel):
    """A single object instance in the OCEL 2.0 log."""

    id: str
    type: str
    attributes: list[OcelObjectAttribute] = Field(default_factory=list)
    relationships: list[OcelRelationship] = Field(default_factory=list)


class OcelLog(BaseModel):
    """Top-level OCEL 2.0 event log container.

    This is the root model that gets serialized to a .jsonocel file.
    """

    eventTypes: list[OcelEventType] = Field(default_factory=list)
    objectTypes: list[OcelObjectType] = Field(default_factory=list)
    events: list[OcelEvent] = Field(default_factory=list)
    objects: list[OcelObject] = Field(default_factory=list)

    def add_event_type(self, name: str, attributes: list[OcelAttributeDefinition]) -> None:
        """Register a new event type if not already present."""
        existing = {et.name for et in self.eventTypes}
        if name not in existing:
            self.eventTypes.append(OcelEventType(name=name, attributes=attributes))

    def add_object_type(self, name: str, attributes: list[OcelAttributeDefinition]) -> None:
        """Register a new object type if not already present."""
        existing = {ot.name for ot in self.objectTypes}
        if name not in existing:
            self.objectTypes.append(OcelObjectType(name=name, attributes=attributes))

    def add_event(self, event: OcelEvent) -> None:
        """Append an event to the log."""
        self.events.append(event)

    def add_object(self, obj: OcelObject) -> None:
        """Append an object to the log if not already present."""
        existing = {o.id for o in self.objects}
        if obj.id not in existing:
            self.objects.append(obj)

    def merge(self, other: OcelLog) -> None:
        """Merge another OcelLog into this one (for combining runs)."""
        for et in other.eventTypes:
            self.add_event_type(et.name, et.attributes)
        for ot in other.objectTypes:
            self.add_object_type(ot.name, ot.attributes)
        for event in other.events:
            self.add_event(event)
        for obj in other.objects:
            self.add_object(obj)

    def to_serializable(self) -> dict[str, Any]:
        """Convert to a plain dict matching the OCEL 2.0 JSON schema.

        Datetime values are formatted as ISO 8601 strings.
        """
        return self.model_dump(mode="json")
