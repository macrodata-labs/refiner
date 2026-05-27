from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True, slots=True)
class StructuredOutputSchema:
    model: type[BaseModel]
    name: str
    json_schema: dict[str, Any]
    strict: bool


class InferenceSchemaValidationError(ValueError):
    def __init__(
        self,
        *,
        schema_name: str,
        text: str,
        cause: ValidationError,
    ) -> None:
        super().__init__(
            f"response did not match structured output schema {schema_name!r}"
        )
        self.schema_name = schema_name
        self.text = text
        self.__cause__ = cause


def normalize_schema(
    schema: type[BaseModel] | None,
    *,
    strict: bool = True,
) -> StructuredOutputSchema | None:
    if schema is None:
        return None
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise TypeError("schema must be a pydantic BaseModel class")
    return StructuredOutputSchema(
        model=schema,
        name=schema.__name__,
        json_schema=schema.model_json_schema(),
        strict=strict,
    )


def validate_structured_output(
    text: str,
    schema: StructuredOutputSchema | None,
) -> BaseModel | None:
    if schema is None:
        return None
    try:
        return schema.model.model_validate_json(text)
    except ValidationError as err:
        raise InferenceSchemaValidationError(
            schema_name=schema.name,
            text=text,
            cause=err,
        ) from err


__all__ = [
    "InferenceSchemaValidationError",
    "StructuredOutputSchema",
    "normalize_schema",
    "validate_structured_output",
]
