from __future__ import annotations

from collections.abc import Iterator, Sequence
import posixpath
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data import datatype
from refiner.pipeline.data.shard import (
    INTERNAL_ROW_COLUMNS,
    SOURCE_ROW_ID_COLUMN,
    RowRangeDescriptor,
    Shard,
)
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sinks.lance_utils import validate_lance_uri
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.shard_limit import (
    validate_num_shards,
    validate_shard_count,
)
from refiner.utils import check_required_dependencies

_LANCE_ROW_ADDRESS_COLUMN = "_rowaddr"
_LANCE_BLOB_METADATA_KEY = b"lance-encoding:blob"


def _import_lance() -> Any:
    check_required_dependencies("load_lance", [("lance", "pylance")], dist="lance")
    import lance

    return lance


def _is_classic_lance_blob(field: pa.Field) -> bool:
    value = (field.metadata or {}).get(_LANCE_BLOB_METADATA_KEY, b"")
    return value.lower() == b"true"


def _is_lance_blob_v2(field: pa.Field) -> bool:
    field_type = field.type
    return isinstance(field_type, pa.ExtensionType) and (
        field_type.extension_name == "lance.blob.v2"
    )


def _blob_reference_field(field: pa.Field) -> pa.Field:
    reference = datatype.blob_reference(datatype.asset_type(field) or "file")
    return pa.field(
        field.name,
        reference.type,
        nullable=field.nullable,
        metadata=reference.metadata,
    )


class LanceSource(BaseSource):
    """A version-pinned Lance source sharded by contiguous fragment groups."""

    name = "load_lance"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        version: int | str | None = None,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
        num_shards: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be >= 0 or None")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance columns must be unique")
        validate_num_shards(num_shards)

        self.input = DataFolder.resolve(input)
        if self.input.has_explicit_filesystem_configuration:
            raise ValueError(
                "load_lance does not support configured fsspec handles; pass a URI "
                "whose credentials and endpoint are available to Lance"
            )
        self.dataset_uri = self.input.abs_path()
        validate_lance_uri(self.dataset_uri)
        self.columns = tuple(columns) if columns is not None else None
        self.batch_size = int(batch_size)
        self.num_shards = num_shards
        self.max_rows = max_rows
        self._dataset_cache: Any | None = None

        dataset = _import_lance().dataset(self.dataset_uri, version=version)
        self.version = int(dataset.version)
        source_schema = dataset.schema
        selected_columns = set(self.columns) if self.columns is not None else None
        unsupported_blobs = [
            field.name
            for field in source_schema
            if _is_lance_blob_v2(field)
            and (selected_columns is None or field.name in selected_columns)
        ]
        if unsupported_blobs:
            raise ValueError(
                "Lance Blob V2 columns cannot be represented as physical Refiner "
                "blob references: " + ", ".join(unsupported_blobs)
            )
        self._blob_field_ids = {
            field.name: int(dataset.lance_schema.field(field.name).id())
            for field in source_schema
            if _is_classic_lance_blob(field)
        }
        normalized_schema = pa.schema(
            [
                _blob_reference_field(field)
                if field.name in self._blob_field_ids
                else field
                for field in source_schema
            ],
            metadata=source_schema.metadata,
        )
        reserved = set(INTERNAL_ROW_COLUMNS).intersection(source_schema.names)
        if reserved:
            raise ValueError(
                f"Lance dataset contains reserved column {sorted(reserved)[0]}"
            )
        if self.columns is None:
            projected_schema = normalized_schema
        else:
            missing = sorted(set(self.columns).difference(source_schema.names))
            if missing:
                raise ValueError(
                    "Lance columns do not exist in the dataset: " + ", ".join(missing)
                )
            projected_schema = pa.schema(
                [normalized_schema.field(name) for name in self.columns],
                metadata=normalized_schema.metadata,
            )
        self._schema = projected_schema

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self.input.required_refiner_extras()

    def _dataset(self) -> Any:
        if self._dataset_cache is None:
            self._dataset_cache = _import_lance().dataset(
                self.dataset_uri, version=self.version
            )
        return self._dataset_cache

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_dataset_cache"] = None
        return state

    def list_shards(self) -> list[Shard]:
        fragments = self._dataset().get_fragments()
        fragment_count = len(fragments)
        if fragment_count == 0:
            return []
        max_rows = getattr(self, "max_rows", None)
        if max_rows == 0:
            return []
        if max_rows is not None:
            rows = 0
            fragment_count = 0
            for fragment in fragments:
                fragment_count += 1
                rows += int(fragment.count_rows())
                if rows >= max_rows:
                    break
        if self.num_shards is None or self.num_shards <= 0:
            validate_shard_count(fragment_count, source="Lance automatic")
        shard_count = (
            fragment_count
            if self.num_shards is None or self.num_shards <= 0
            else min(self.num_shards, fragment_count)
        )
        base_size, remainder = divmod(fragment_count, shard_count)
        shards: list[Shard] = []
        start = 0
        for index in range(shard_count):
            end = start + base_size + (index < remainder)
            shards.append(
                Shard.from_row_range(
                    start=start,
                    end=end,
                    global_ordinal=index,
                )
            )
            start = end
        return shards

    def _fragment_blob_paths(self, fragment: Any) -> dict[str, str]:
        paths: dict[str, str] = {}
        selected = set(self.columns) if self.columns is not None else None
        for name, field_id in self._blob_field_ids.items():
            if selected is not None and name not in selected:
                continue
            matches = [
                data_file.path
                for data_file in fragment.data_files()
                if field_id in data_file.fields
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Lance blob column {name!r} must belong to exactly one data file"
                )
            relative_path = posixpath.normpath(matches[0])
            if (
                relative_path in {"", ".", ".."}
                or relative_path.startswith("../")
                or relative_path.startswith("/")
                or "\\" in relative_path
            ):
                raise ValueError(f"Invalid Lance data file path: {matches[0]}")
            paths[name] = self.input.abs_path(f"data/{relative_path}")
        return paths

    def _normalize_blob_columns(
        self,
        table: pa.Table,
        paths: dict[str, str],
    ) -> pa.Table:
        out = table
        for name, path in paths.items():
            index = out.schema.get_field_index(name)
            if index < 0:
                continue
            descriptions = out.column(index).combine_chunks()
            if not pa.types.is_struct(descriptions.type):
                raise TypeError(
                    f"Lance blob column {name!r} did not yield descriptions"
                )
            position_index = descriptions.type.get_field_index("position")
            size_index = descriptions.type.get_field_index("size")
            if position_index < 0 or size_index < 0:
                raise TypeError(
                    f"Lance blob column {name!r} has an invalid description"
                )
            field = self._schema.field(name)
            reference = pa.StructArray.from_arrays(
                [
                    pa.array([path] * len(descriptions), type=pa.string()),
                    descriptions.field(position_index),
                    descriptions.field(size_index),
                ],
                fields=list(field.type),
                mask=descriptions.is_null(),
            )
            out = out.set_column(index, field, reference)
        return out

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("LanceSource requires row-range shards")
        fragments = self._dataset().get_fragments()
        if (
            descriptor.start < 0
            or descriptor.start >= descriptor.end
            or descriptor.end > len(fragments)
        ):
            raise ValueError("Lance shard fragment range is invalid")
        rows_before_shard = (
            sum(int(fragment.count_rows()) for fragment in fragments[: descriptor.start])
            if self.max_rows is not None
            else 0
        )
        for fragment in fragments[descriptor.start : descriptor.end]:
            remaining = (
                self.max_rows - rows_before_shard
                if self.max_rows is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                return
            blob_paths = self._fragment_blob_paths(fragment)
            expected_rows = int(fragment.count_rows())
            rows_read = 0
            for batch in fragment.to_batches(
                columns=list(self.columns) if self.columns is not None else None,
                batch_size=self.batch_size,
                with_row_address=True,
                blob_handling="blobs_descriptions",
            ):
                if remaining is not None:
                    batch = batch.slice(0, min(batch.num_rows, remaining))
                    if batch.num_rows == 0:
                        break
                table = self._normalize_blob_columns(
                    pa.Table.from_batches([batch]),
                    blob_paths,
                )
                row_addresses = table.column(_LANCE_ROW_ADDRESS_COLUMN)
                rows_read += batch.num_rows
                rows_before_shard += batch.num_rows
                if remaining is not None:
                    remaining -= batch.num_rows
                yield Tabular(
                    set_or_append_column(
                        table.drop_columns([_LANCE_ROW_ADDRESS_COLUMN]),
                        SOURCE_ROW_ID_COLUMN,
                        row_addresses,
                    )
                )
                if remaining == 0:
                    break
            if rows_read != expected_rows and (
                self.max_rows is None or rows_before_shard < self.max_rows
            ):
                raise ValueError(
                    f"Lance fragment {fragment.fragment_id} yielded {rows_read} rows; "
                    f"expected {expected_rows}"
                )

    def describe(self) -> dict[str, object]:
        return {
            "path": self.dataset_uri,
            "version": self.version,
            "columns": list(self.columns) if self.columns is not None else None,
            "batch_size": self.batch_size,
            "num_shards": self.num_shards,
            "max_rows": self.max_rows,
        }


__all__ = [
    "LanceSource",
]
