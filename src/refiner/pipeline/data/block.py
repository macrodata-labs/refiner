from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa

from refiner.pipeline.data.row import ArrowRowView, Row

_SHARD_ID_COLUMN = "__shard_id"


class Block(ABC):
    @classmethod
    @abstractmethod
    def from_rows(cls, rows: Sequence[Row]) -> "Block":
        raise NotImplementedError

    @abstractmethod
    def iter_rows(self) -> Iterator[Row]:
        raise NotImplementedError

    @abstractmethod
    def with_table(self, table: pa.Table) -> "Block":
        raise NotImplementedError

    @property
    @abstractmethod
    def table(self) -> pa.Table:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Row]:
        return self.iter_rows()


@dataclass(frozen=True, slots=True)
class TabularBlock(Block):
    table: pa.Table

    @classmethod
    def from_rows(cls, rows: Sequence[Row]) -> "TabularBlock":
        records: list[dict[str, object]] = []
        for row in rows:
            record = row.to_dict()
            if row.shard_id is not None:
                record[_SHARD_ID_COLUMN] = row.shard_id
            records.append(record)
        return cls(pa.Table.from_pylist(records))

    def iter_rows(self) -> Iterator[Row]:
        names = tuple(str(name) for name in self.table.column_names)
        columns = tuple(self.table.column(name) for name in names)
        index_by_name = {name: i for i, name in enumerate(names)}
        for row_idx in range(self.table.num_rows):
            shard_id = None
            if _SHARD_ID_COLUMN in index_by_name:
                shard = columns[index_by_name[_SHARD_ID_COLUMN]][row_idx].as_py()
                shard_id = shard if isinstance(shard, str) else None
            yield ArrowRowView(
                names=names,
                columns=columns,
                index_by_name=index_by_name,
                row_idx=row_idx,
                shard_id=shard_id,
            )

    def with_table(self, table: pa.Table) -> "TabularBlock":
        return TabularBlock(table)


__all__ = ["Block", "TabularBlock"]
