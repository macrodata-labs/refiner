from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.utils import check_required_dependencies

LANCE_FRAGMENT_ID_COLUMN = "__refiner_lance_fragment_id"
LANCE_ROW_POSITION_COLUMN = "__refiner_lance_row_position"
LANCE_INTERNAL_COLUMNS = frozenset(
    {LANCE_FRAGMENT_ID_COLUMN, LANCE_ROW_POSITION_COLUMN}
)


def _import_lance() -> Any:
    check_required_dependencies("load_lance", [("lance", "pylance")], dist="lance")
    import lance

    return lance


class LanceSource(BaseSource):
    """A version-pinned Lance source sharded one-to-one by fragment."""

    name = "load_lance"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        version: int | str | None = None,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
        blob_handling: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if columns is not None:
            invalid = LANCE_INTERNAL_COLUMNS.intersection(columns)
            if invalid:
                raise ValueError(f"{sorted(invalid)[0]} is an internal column")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance columns must be unique")

        self.input = DataFolder.resolve(input)
        self.dataset_uri = self.input.abs_path()
        self.columns = tuple(columns) if columns is not None else None
        self.batch_size = int(batch_size)
        self.blob_handling = blob_handling

        dataset = _import_lance().dataset(self.dataset_uri, version=version)
        self.version = int(dataset.version)
        source_schema = dataset.schema
        reserved = LANCE_INTERNAL_COLUMNS.intersection(source_schema.names)
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
        self._schema = projected_schema.append(
            pa.field(LANCE_FRAGMENT_ID_COLUMN, pa.uint64(), nullable=False)
        ).append(pa.field(LANCE_ROW_POSITION_COLUMN, pa.uint64(), nullable=False))

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self.input.required_refiner_extras()

    def _dataset(self) -> Any:
        return _import_lance().dataset(self.dataset_uri, version=self.version)

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_row_range(
                start=index,
                end=index + 1,
                global_ordinal=index,
            )
            for index, _ in enumerate(self._dataset().get_fragments())
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("LanceSource requires row-range shards")
        if descriptor.end != descriptor.start + 1:
            raise ValueError("Lance shards must identify exactly one fragment")
        fragments = self._dataset().get_fragments()
        if descriptor.start < 0 or descriptor.start >= len(fragments):
            raise ValueError(
                f"Lance fragment index {descriptor.start} is out of bounds"
            )
        fragment = fragments[descriptor.start]
        fragment_id = int(fragment.fragment_id)
        expected_rows = int(fragment.count_rows())
        next_position = 0
        for batch in fragment.to_batches(
            columns=list(self.columns) if self.columns is not None else None,
            batch_size=self.batch_size,
            blob_handling=self.blob_handling,
        ):
            positions = pa.array(
                range(next_position, next_position + batch.num_rows),
                type=pa.uint64(),
            )
            next_position += batch.num_rows
            fragment_ids = pa.array(
                [fragment_id] * batch.num_rows,
                type=pa.uint64(),
            )
            yield Tabular(
                pa.Table.from_batches([batch])
                .append_column(LANCE_FRAGMENT_ID_COLUMN, fragment_ids)
                .append_column(LANCE_ROW_POSITION_COLUMN, positions)
            )
        if next_position != expected_rows:
            raise ValueError(
                f"Lance fragment {fragment_id} yielded {next_position} rows; "
                f"expected {expected_rows}"
            )

    def describe(self) -> dict[str, object]:
        return {
            "path": self.dataset_uri,
            "version": self.version,
            "columns": list(self.columns) if self.columns is not None else None,
            "batch_size": self.batch_size,
            "blob_handling": self.blob_handling,
        }


__all__ = [
    "LANCE_FRAGMENT_ID_COLUMN",
    "LANCE_INTERNAL_COLUMNS",
    "LANCE_ROW_POSITION_COLUMN",
    "LanceSource",
]
