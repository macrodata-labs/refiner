from __future__ import annotations

import base64
from typing import Any, cast


def lance_schema_to_payload(schema: Any) -> dict[str, object]:
    reduced = schema.__reduce__()
    if (
        not isinstance(reduced, tuple)
        or len(reduced) != 2
        or not isinstance(reduced[1], tuple)
        or not reduced[1]
        or not isinstance(reduced[1][0], str)
        or any(not isinstance(field, bytes) for field in reduced[1][1:])
    ):
        raise RuntimeError("Unsupported pylance LanceSchema serialization format")
    metadata = reduced[1][0]
    field_protos = reduced[1][1:]
    return {
        "metadata": metadata,
        "fields": [base64.b64encode(field).decode("ascii") for field in field_protos],
    }


def lance_schema_from_payload(lance: Any, payload: object) -> Any:
    if not isinstance(payload, dict):
        raise ValueError("Invalid Lance schema metadata payload")
    payload_dict = cast(dict[str, object], payload)
    metadata = payload_dict.get("metadata")
    fields = payload_dict.get("fields")
    if not isinstance(metadata, str) or not isinstance(fields, list):
        raise ValueError("Invalid Lance schema metadata payload")
    encoded_fields: list[str] = []
    for field in fields:
        if not isinstance(field, str):
            raise ValueError("Invalid Lance schema field payload")
        encoded_fields.append(field)
    factory = getattr(lance.schema.LanceSchema, "_from_protos", None)
    if not callable(factory):
        raise RuntimeError("Unsupported pylance LanceSchema serialization format")
    return factory(
        metadata,
        *(base64.b64decode(field) for field in encoded_fields),
    )
