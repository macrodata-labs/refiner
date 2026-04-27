from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import pyarrow as pa

_ASSET_TYPE_METADATA_KEY = b"asset_type"
_FILE_ASSET_TYPES = frozenset({b"unknown", b"image", b"audio", b"video", b"pdf"})
_FILE_FIELD_NAME = "__refiner_file__"

DTypeLike: TypeAlias = str | pa.DataType | pa.Field
DTypeMapping: TypeAlias = Mapping[str, DTypeLike]


null = pa.null
bool = pa.bool_
int8 = pa.int8
int16 = pa.int16
int32 = pa.int32
int64 = pa.int64
uint8 = pa.uint8
uint16 = pa.uint16
uint32 = pa.uint32
uint64 = pa.uint64
float32 = pa.float32
float64 = pa.float64
string = pa.string
large_string = pa.large_string
binary = pa.binary
large_binary = pa.large_binary
duration = pa.duration


def date() -> pa.DataType:
    return pa.date32()


def time(unit: str = "us") -> pa.DataType:
    return pa.time32(unit) if unit in {"s", "ms"} else pa.time64(unit)


def timestamp(unit: str = "us", timezone: str | None = None) -> pa.DataType:
    return pa.timestamp(unit, tz=timezone)


def list(dtype: DTypeLike, size: int | None = None) -> pa.DataType:
    child = _child_type_or_field(dtype)
    return pa.list_(child) if size is None else pa.list_(child, list_size=size)


def large_list(dtype: DTypeLike) -> pa.DataType:
    return pa.large_list(_child_type_or_field(dtype))


def struct(
    fields: Mapping[str, DTypeLike] | Sequence[pa.Field | tuple[str, DTypeLike]],
) -> pa.DataType:
    if isinstance(fields, Mapping):
        mapping = cast(Mapping[str, DTypeLike], fields)
        arrow_fields = [_to_field(name, dtype) for name, dtype in mapping.items()]
    else:
        arrow_fields = [
            item if isinstance(item, pa.Field) else _to_field(item[0], item[1])
            for item in fields
        ]
    return pa.struct(arrow_fields)


def map(key_type: DTypeLike, value_type: DTypeLike) -> pa.DataType:
    return pa.map_(_arrow_type(key_type), _arrow_type(value_type))


def file() -> pa.Field:
    return _file_field(b"unknown")


def image_file() -> pa.Field:
    return _file_field(b"image")


def audio_file() -> pa.Field:
    return _file_field(b"audio")


def video_file() -> pa.Field:
    return _file_field(b"video")


def pdf_file() -> pa.Field:
    return _file_field(b"pdf")


def schema_with_dtypes(
    schema: pa.Schema | None,
    dtypes: DTypeMapping | None,
    *,
    preserve_metadata: builtins.bool = True,
) -> pa.Schema | None:
    if not dtypes:
        return schema

    fields = builtins.list(schema) if schema is not None else []
    index_by_name = {field.name: idx for idx, field in enumerate(fields)}
    for name, dtype in dtypes.items():
        idx = index_by_name.get(name)
        if idx is None:
            index_by_name[name] = len(fields)
            fields.append(_to_field(name, dtype))
            continue
        fields[idx] = _replace_field_dtype(
            fields[idx],
            dtype,
            preserve_metadata=preserve_metadata,
        )
    return pa.schema(fields, metadata=schema.metadata if schema is not None else None)


def apply_dtypes_to_table(
    table: pa.Table,
    dtypes: DTypeMapping | None,
    *,
    strict: builtins.bool = True,
    preserve_metadata: builtins.bool = True,
) -> pa.Table:
    if not dtypes:
        return table
    out = table
    for name, dtype in dtypes.items():
        idx = out.schema.get_field_index(name)
        if idx < 0:
            if strict:
                raise KeyError(f"Unknown column for dtype: {name}")
            continue
        field = _replace_field_dtype(
            out.schema.field(idx),
            dtype,
            preserve_metadata=preserve_metadata,
        )
        column = out.column(idx)
        if column.type != field.type:
            column = column.cast(field.type)
        if out.schema.field(idx).equals(field, check_metadata=True):
            continue
        out = out.set_column(idx, field, column)
    return out


def is_file_field(field: pa.Field) -> builtins.bool:
    metadata = field.metadata or {}
    return metadata.get(_ASSET_TYPE_METADATA_KEY) in _FILE_ASSET_TYPES


def dtype_to_plan(dtype: DTypeLike) -> str | dict[str, object]:
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, pa.DataType):
        return str(dtype)
    if isinstance(dtype, pa.Field):
        out: dict[str, object] = {"type": str(dtype.type)}
        if dtype.metadata:
            out["metadata"] = {
                key.decode("utf-8", errors="replace"): value.decode(
                    "utf-8",
                    errors="replace",
                )
                for key, value in dtype.metadata.items()
            }
        return out
    raise TypeError(f"Unsupported dtype: {type(dtype)!r}")


def _file_field(asset_type: bytes) -> pa.Field:
    return pa.field(
        _FILE_FIELD_NAME,
        pa.string(),
        metadata={_ASSET_TYPE_METADATA_KEY: asset_type},
    )


def _replace_field_dtype(
    field: pa.Field,
    dtype: DTypeLike,
    *,
    preserve_metadata: builtins.bool = True,
) -> pa.Field:
    if isinstance(dtype, pa.Field):
        metadata = dict(field.metadata or {}) if preserve_metadata else {}
        if dtype.metadata:
            metadata.update(dtype.metadata)
        return pa.field(
            field.name,
            dtype.type,
            nullable=field.nullable,
            metadata=metadata or None,
        )
    if isinstance(dtype, pa.DataType):
        return pa.field(
            field.name,
            dtype,
            nullable=field.nullable,
            metadata=field.metadata if preserve_metadata else None,
        )
    if isinstance(dtype, str):
        return pa.field(
            field.name,
            pa.type_for_alias(dtype),
            nullable=field.nullable,
            metadata=field.metadata if preserve_metadata else None,
        )
    raise TypeError(f"Unsupported dtype: {type(dtype)!r}")


def _to_field(name: str, dtype: DTypeLike) -> pa.Field:
    if isinstance(dtype, pa.Field):
        return dtype.with_name(name)
    if isinstance(dtype, pa.DataType):
        return pa.field(name, dtype)
    if isinstance(dtype, str):
        return pa.field(name, pa.type_for_alias(dtype))
    raise TypeError(f"Unsupported dtype: {type(dtype)!r}")


def _child_type_or_field(dtype: DTypeLike) -> pa.DataType | pa.Field:
    if isinstance(dtype, pa.Field):
        return dtype.with_name("item")
    if isinstance(dtype, pa.DataType):
        return dtype
    if isinstance(dtype, str):
        return pa.type_for_alias(dtype)
    raise TypeError(f"Unsupported dtype: {type(dtype)!r}")


def _arrow_type(dtype: DTypeLike) -> pa.DataType:
    if isinstance(dtype, pa.Field):
        return dtype.type
    if isinstance(dtype, pa.DataType):
        return dtype
    if isinstance(dtype, str):
        return pa.type_for_alias(dtype)
    raise TypeError(f"Unsupported dtype: {type(dtype)!r}")


__all__ = [
    "DTypeLike",
    "DTypeMapping",
    "apply_dtypes_to_table",
    "audio_file",
    "binary",
    "bool",
    "date",
    "dtype_to_plan",
    "duration",
    "file",
    "float32",
    "float64",
    "image_file",
    "int8",
    "int16",
    "int32",
    "int64",
    "is_file_field",
    "large_binary",
    "large_list",
    "large_string",
    "list",
    "map",
    "null",
    "pdf_file",
    "schema_with_dtypes",
    "string",
    "struct",
    "time",
    "timestamp",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "video_file",
]
