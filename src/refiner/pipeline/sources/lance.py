from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.shard import LanceFragmentDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.utils import check_required_dependencies

LANCE_ROW_POSITION_COLUMN = "__refiner_lance_row_position"


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
        if columns is not None and LANCE_ROW_POSITION_COLUMN in columns:
            raise ValueError(f"{LANCE_ROW_POSITION_COLUMN} is an internal column")
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
            pa.field(LANCE_ROW_POSITION_COLUMN, pa.uint64(), nullable=False)
        )

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _dataset(self) -> Any:
        return _import_lance().dataset(self.dataset_uri, version=self.version)

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_lance_fragment(
                dataset_uri=self.dataset_uri,
                version=self.version,
                fragment_id=int(fragment.fragment_id),
                num_rows=int(fragment.count_rows()),
                global_ordinal=index,
            )
            for index, fragment in enumerate(self._dataset().get_fragments())
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, LanceFragmentDescriptor):
            raise TypeError("LanceSource requires Lance-fragment shards")
        if descriptor.dataset_uri != self.dataset_uri:
            raise ValueError("Lance shard belongs to a different dataset")
        if descriptor.version != self.version:
            raise ValueError("Lance shard belongs to a different dataset version")

        fragment = self._dataset().get_fragment(descriptor.fragment_id)
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
            yield Tabular(
                pa.Table.from_batches([batch]).append_column(
                    LANCE_ROW_POSITION_COLUMN,
                    positions,
                )
            )
        if next_position != descriptor.num_rows:
            raise ValueError(
                f"Lance fragment {descriptor.fragment_id} yielded {next_position} rows; "
                f"expected {descriptor.num_rows}"
            )

    def describe(self) -> dict[str, object]:
        return {
            "path": self.dataset_uri,
            "version": self.version,
            "columns": list(self.columns) if self.columns is not None else None,
            "batch_size": self.batch_size,
            "blob_handling": self.blob_handling,
        }


__all__ = ["LANCE_ROW_POSITION_COLUMN", "LanceSource"]
