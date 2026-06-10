from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.steps import FlatMapStep, MapResult


class TaskSource(BaseSource):
    """Synthetic source that yields one task row per rank."""

    name = "task"

    def __init__(self, *, num_tasks: int) -> None:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")
        self._num_tasks = num_tasks

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_row_range(start=rank, end=rank + 1, global_ordinal=rank)
            for rank in range(self._num_tasks)
        ]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("TaskSource requires row-range shards")
        rank = int(descriptor.start)
        if rank < 0 or rank >= self._num_tasks:
            raise ValueError(f"Invalid task rank {rank} for {self._num_tasks} tasks")
        yield DictRow({"task_rank": rank})

    def describe(self) -> dict[str, int]:
        return {"num_tasks": self._num_tasks}


@dataclass(frozen=True, slots=True)
class TaskStep(FlatMapStep):
    """Apply a task callback and normalize zero, one, or many output rows."""

    fn: Callable[[int, int], Any]
    num_tasks: int
    index: int
    op_name: str | None = "task"

    def apply_row_many(self, row: Row) -> Iterator[MapResult]:
        result = self.fn(row["task_rank"], self.num_tasks)

        if result is None:
            return
        if isinstance(result, Row):
            yield result
            return
        if isinstance(result, Mapping):
            yield dict(result)
            return
        if isinstance(result, Iterable) and not isinstance(
            result, (str, bytes, bytearray, memoryview)
        ):
            for item in result:
                if isinstance(item, Row):
                    yield item
                elif isinstance(item, Mapping):
                    yield dict(item)
                else:
                    yield {"result": item}
            return
        yield {"result": result}


__all__ = ["TaskSource", "TaskStep"]
