from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.shard import (
    INTERNAL_ROW_COLUMNS,
    SOURCE_ROW_ID_COLUMN,
    RowRangeDescriptor,
    Shard,
)
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sinks.lance_utils import validate_lance_uri
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.utils import check_required_dependencies

_LANCE_ROW_ADDRESS_COLUMN = "_rowaddr"


def _import_lance() -> Any:
    check_required_dependencies("load_lance", [("lance", "pylance")], dist="lance")
    import lance

    return lance


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
        blob_handling: str | None = None,
        num_shards: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance columns must be unique")

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
        self.blob_handling = blob_handling
        self.num_shards = num_shards
        self._dataset_cache: Any | None = None

        dataset = _import_lance().dataset(self.dataset_uri, version=version)
        self.version = int(dataset.version)
        source_schema = dataset.schema
        reserved = set(INTERNAL_ROW_COLUMNS).intersection(source_schema.names)
        if reserved:
            raise ValueError(
                f"Lance dataset contains reserved column {sorted(reserved)[0]}"
            )
        if self.columns is None:
            projected_schema = source_schema
        else:
            missing = sorted(set(self.columns).difference(source_schema.names))
            if missing:
                raise ValueError(
                    "Lance columns do not exist in the dataset: " + ", ".join(missing)
                )
            projected_schema = pa.schema(
                [source_schema.field(name) for name in self.columns],
                metadata=source_schema.metadata,
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
        fragment_count = len(self._dataset().get_fragments())
        if fragment_count == 0:
            return []
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
        for fragment in fragments[descriptor.start : descriptor.end]:
            expected_rows = int(fragment.count_rows())
            rows_read = 0
            for batch in fragment.to_batches(
                columns=list(self.columns) if self.columns is not None else None,
                batch_size=self.batch_size,
                with_row_address=True,
                blob_handling=self.blob_handling,
            ):
                table = pa.Table.from_batches([batch])
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
                    f"Lance fragment {fragment.fragment_id} yielded {rows_read} rows; "
                    f"expected {expected_rows}"
                )

    def describe(self) -> dict[str, object]:
        return {
            "path": self.dataset_uri,
            "version": self.version,
            "columns": list(self.columns) if self.columns is not None else None,
            "batch_size": self.batch_size,
            "blob_handling": self.blob_handling,
            "num_shards": self.num_shards,
        }


__all__ = [
    "LanceSource",
]
