from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import copy
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
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_MAX_AUTOMATIC_SHARDS,
    validate_explicit_num_shards,
)
from refiner.utils import check_required_dependencies

_LANCE_ROW_ADDRESS_COLUMN = "_rowaddr"
_LANCE_BLOB_METADATA_KEY = b"lance-encoding:blob"
_DEFAULT_READ_BATCH_ROWS = 65_536


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
    """A lazily version-pinned Lance source sharded by fragment groups."""

    name = "load_lance"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        version: int | str | None = None,
        columns: Sequence[str] | None = None,
        num_shards: int | None = None,
    ) -> None:
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance columns must be unique")
        validate_explicit_num_shards(num_shards)

        self.input = DataFolder.resolve(input)
        if self.input.has_explicit_filesystem_configuration:
            raise ValueError(
                "load_lance does not support configured fsspec handles; pass a URI "
                "whose credentials and endpoint are available to Lance"
            )
        self.dataset_uri = self.input.abs_path()
        validate_lance_uri(self.dataset_uri)
        self.columns = tuple(columns) if columns is not None else None
        self._read_batch_rows = _DEFAULT_READ_BATCH_ROWS
        self.num_shards = num_shards
        self._dataset_cache: Any | None = None
        self._requested_version = version
        self._version: int | None = None
        self._schema: pa.Schema | None = None
        self._blob_field_ids: dict[str, int] = {}
        if version is not None:
            self._initialize_from_dataset(
                _import_lance().dataset(self.dataset_uri, version=version)
            )

    def _initialize_from_dataset(self, dataset: Any) -> None:
        """Pin the opened dataset version and validate its projected schema."""
        self._version = int(dataset.version)
        self._dataset_cache = dataset
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
    def version(self) -> int:
        """Return the pinned version, resolving the latest version on first use."""
        self._dataset()
        if self._version is None:
            raise RuntimeError("Lance dataset version was not resolved")
        return self._version

    def with_read_batch_rows(self, max_rows: int | None) -> "LanceSource":
        source = copy(self)
        source._read_batch_rows = (
            _DEFAULT_READ_BATCH_ROWS if max_rows is None else max_rows
        )
        return source

    def with_max_read_batch_rows(self, max_rows: int) -> "LanceSource":
        source = copy(self)
        source._read_batch_rows = min(source._read_batch_rows, max_rows)
        return source

    @property
    def schema(self) -> pa.Schema:
        self._dataset()
        if self._schema is None:
            raise RuntimeError("Lance dataset schema was not resolved")
        return self._schema

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self.input.required_refiner_extras()

    def _dataset(self) -> Any:
        if self._dataset_cache is None:
            dataset = _import_lance().dataset(
                self.dataset_uri,
                version=(
                    self._version
                    if self._version is not None
                    else self._requested_version
                ),
            )
            self._initialize_from_dataset(dataset)
        if self._dataset_cache is None:
            raise RuntimeError("Lance dataset was not resolved")
        return self._dataset_cache

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_dataset_cache"] = None
        return state

    def list_shards(self) -> list[Shard]:
        fragment_count = len(self._dataset().get_fragments())
        if fragment_count == 0:
            return []
        shard_count = (
            min(fragment_count, DEFAULT_MAX_AUTOMATIC_SHARDS)
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
            field = self.schema.field(name)
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

    def _source_unit(
        self,
        table: pa.Table,
        row_addresses: pa.ChunkedArray | pa.Array,
        blob_paths: dict[str, str],
    ) -> SourceUnit:
        if table.num_columns == 0:
            return Tabular(pa.table({SOURCE_ROW_ID_COLUMN: row_addresses}))
        return Tabular(
            set_or_append_column(
                self._normalize_blob_columns(table, blob_paths),
                SOURCE_ROW_ID_COLUMN,
                row_addresses,
            )
        )

    def _read_fragment_prefix(
        self,
        fragment: Any,
        row_count: int,
        blob_paths: dict[str, str],
    ) -> Iterator[SourceUnit]:
        """Read a fragment prefix without scanning projected value columns."""
        source_schema = self.schema
        selected_columns = (
            list(self.columns)
            if self.columns is not None
            else list(source_schema.names)
        )
        blob_columns = [
            name for name in selected_columns if name in self._blob_field_ids
        ]
        value_columns = [
            name for name in selected_columns if name not in self._blob_field_ids
        ]
        rows_read = 0
        for address_batch in fragment.to_batches(
            columns=blob_columns,
            batch_size=self._read_batch_rows,
            batch_readahead=1,
            limit=row_count,
            with_row_address=True,
            blob_handling="blobs_descriptions",
        ):
            batch_rows = address_batch.num_rows
            if batch_rows == 0:
                continue
            value_table = (
                fragment.take(
                    pa.array(
                        range(rows_read, rows_read + batch_rows),
                        type=pa.uint64(),
                    ),
                    columns=value_columns,
                )
                if value_columns
                else None
            )
            if value_table is not None and value_table.num_rows != batch_rows:
                raise ValueError(
                    f"Lance fragment {fragment.fragment_id} indexed read yielded "
                    f"{value_table.num_rows} rows; expected {batch_rows}"
                )

            address_table = pa.Table.from_batches([address_batch])
            arrays: list[pa.ChunkedArray] = []
            fields: list[pa.Field] = []
            for name in selected_columns:
                source = address_table if name in self._blob_field_ids else value_table
                if source is None:
                    raise AssertionError(f"Missing Lance source column {name!r}")
                arrays.append(source.column(name))
                fields.append(source.schema.field(name))
            table = pa.Table.from_arrays(
                arrays,
                schema=pa.schema(fields, metadata=source_schema.metadata),
            )
            row_addresses = address_table.column(_LANCE_ROW_ADDRESS_COLUMN)
            rows_read += batch_rows
            yield self._source_unit(table, row_addresses, blob_paths)

        if rows_read != row_count:
            raise ValueError(
                f"Lance fragment {fragment.fragment_id} yielded {rows_read} rows; "
                f"expected {row_count}"
            )

    def _read_shard(
        self,
        shard: Shard,
        max_rows: int | None,
    ) -> Iterator[SourceUnit]:
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
        remaining_rows = max_rows
        for fragment_index in range(descriptor.start, descriptor.end):
            if remaining_rows is not None and remaining_rows == 0:
                break
            fragment = fragments[fragment_index]
            fragment_rows = int(fragment.count_rows())
            expected_rows = (
                fragment_rows
                if remaining_rows is None
                else min(fragment_rows, remaining_rows)
            )
            if expected_rows == 0:
                continue
            blob_paths = self._fragment_blob_paths(fragment)
            if expected_rows < fragment_rows:
                yield from self._read_fragment_prefix(
                    fragment,
                    expected_rows,
                    blob_paths,
                )
            else:
                rows_read = 0
                for batch in fragment.to_batches(
                    columns=list(self.columns) if self.columns is not None else None,
                    batch_size=self._read_batch_rows,
                    with_row_address=True,
                    blob_handling="blobs_descriptions",
                ):
                    table = self._normalize_blob_columns(
                        pa.Table.from_batches([batch]),
                        blob_paths,
                    )
                    row_addresses = table.column(_LANCE_ROW_ADDRESS_COLUMN)
                    rows_read += batch.num_rows
                    yield Tabular(
                        set_or_append_column(
                            table.drop_columns([_LANCE_ROW_ADDRESS_COLUMN]),
                            SOURCE_ROW_ID_COLUMN,
                            row_addresses,
                        )
                    )
                if rows_read != expected_rows:
                    raise ValueError(
                        f"Lance fragment {fragment.fragment_id} yielded "
                        f"{rows_read} rows; expected {expected_rows}"
                    )
            if remaining_rows is not None:
                remaining_rows -= expected_rows

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        return self._read_shard(shard, None)

    def read_shard_prefix(
        self,
        shard: Shard,
        max_rows: int,
    ) -> Iterator[SourceUnit]:
        return self._read_shard(shard, max_rows)

    def describe(self) -> dict[str, object]:
        return {
            "path": self.dataset_uri,
            "version": (
                self._version if self._version is not None else self._requested_version
            ),
            "columns": list(self.columns) if self.columns is not None else None,
            "read_batch_rows": self._read_batch_rows,
            "num_shards": self.num_shards,
        }


__all__ = [
    "LanceSource",
]
