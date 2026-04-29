from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import pyarrow as pa

_ASSET_TYPE_METADATA_KEY = b"asset_type"
_FILE_FIELD_NAME = "__refiner_file__"
_BYTES_WITH_PATH_TYPE = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)

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


def asset_path(asset_type: str) -> pa.Field:
    return _asset_field(asset_type, pa.string())


def asset_bytes(asset_type: str) -> pa.Field:
    return _asset_field(asset_type, pa.binary())


def asset_bytes_with_path(asset_type: str) -> pa.Field:
    return _asset_field(asset_type, _BYTES_WITH_PATH_TYPE)


def file_path() -> pa.Field:
    return asset_path("file")


def file_bytes() -> pa.Field:
    return asset_bytes("file")


def file_bytes_with_path() -> pa.Field:
    return asset_bytes_with_path("file")


def image_path() -> pa.Field:
    return asset_path("image")


def image_bytes() -> pa.Field:
    return asset_bytes("image")


def image_bytes_with_path() -> pa.Field:
    return asset_bytes_with_path("image")


def audio_path() -> pa.Field:
    return asset_path("audio")


def audio_bytes() -> pa.Field:
    return asset_bytes("audio")


def audio_bytes_with_path() -> pa.Field:
    return asset_bytes_with_path("audio")


def video_path() -> pa.Field:
    return asset_path("video")


def video_bytes() -> pa.Field:
    return asset_bytes("video")


def video_bytes_with_path() -> pa.Field:
    return asset_bytes_with_path("video")


def pdf_path() -> pa.Field:
    return asset_path("pdf")


def pdf_bytes() -> pa.Field:
    return asset_bytes("pdf")


def pdf_bytes_with_path() -> pa.Field:
    return asset_bytes_with_path("pdf")


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


def is_asset_field(field: pa.Field) -> builtins.bool:
    metadata = field.metadata or {}
    return builtins.bool(metadata.get(_ASSET_TYPE_METADATA_KEY))


def asset_type(field: pa.Field) -> str | None:
    metadata = field.metadata or {}
    value = metadata.get(_ASSET_TYPE_METADATA_KEY)
    return value.decode("utf-8", errors="replace") if value else None


def asset_storage(field: pa.Field) -> str | None:
    if not is_asset_field(field):
        return None
    field_type = field.type
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return "path"
    if pa.types.is_binary(field_type) or pa.types.is_large_binary(field_type):
        return "bytes"
    if _is_bytes_with_path_type(field_type):
        return "bytes_with_path"
    return None


def is_asset_path_field(field: pa.Field) -> builtins.bool:
    return asset_storage(field) == "path"


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


def _asset_field(asset_type: str, storage_type: pa.DataType) -> pa.Field:
    return pa.field(
        _FILE_FIELD_NAME,
        storage_type,
        metadata={_ASSET_TYPE_METADATA_KEY: asset_type.encode("utf-8")},
    )


def _is_bytes_with_path_type(field_type: pa.DataType) -> builtins.bool:
    if not pa.types.is_struct(field_type):
        return False
    bytes_idx = field_type.get_field_index("bytes")
    path_idx = field_type.get_field_index("path")
    if bytes_idx < 0 or path_idx < 0:
        return False
    bytes_type = field_type.field(bytes_idx).type
    path_type = field_type.field(path_idx).type
    return (
        pa.types.is_binary(bytes_type) or pa.types.is_large_binary(bytes_type)
    ) and (pa.types.is_string(path_type) or pa.types.is_large_string(path_type))


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
    "asset_bytes",
    "asset_bytes_with_path",
    "asset_path",
    "asset_storage",
    "asset_type",
    "audio_bytes",
    "audio_bytes_with_path",
    "audio_path",
    "binary",
    "bool",
    "date",
    "dtype_to_plan",
    "duration",
    "file_bytes",
    "file_bytes_with_path",
    "file_path",
    "float32",
    "float64",
    "image_bytes",
    "image_bytes_with_path",
    "image_path",
    "int8",
    "int16",
    "int32",
    "int64",
    "is_asset_field",
    "is_asset_path_field",
    "large_binary",
    "large_list",
    "large_string",
    "list",
    "map",
    "null",
    "pdf_bytes",
    "pdf_bytes_with_path",
    "pdf_path",
    "schema_with_dtypes",
    "string",
    "struct",
    "time",
    "timestamp",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "video_bytes",
    "video_bytes_with_path",
    "video_path",
]
