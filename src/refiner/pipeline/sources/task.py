from __future__ import annotations

from collections.abc import Iterator

from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource
from refiner.pipeline.data.row import DictRow, Row

_TASK_SOURCE_PATH = "memory://tasks"


class TaskSource(BaseSource):
    """Synthetic source that yields one task row per rank."""

    name = "task"

    def __init__(self, *, num_tasks: int) -> None:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")
        self._num_tasks = num_tasks

    def list_shards(self) -> list[Shard]:
        return [
            Shard(
                descriptor=FilePartsDescriptor(
                    (
                        FilePart(
                            path=_TASK_SOURCE_PATH,
                            start=rank,
                            end=rank + 1,
                            unit="rows",
                        ),
                    )
                ),
                global_ordinal=rank,
            )
            for rank in range(self._num_tasks)
        ]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        part = shard.descriptor.parts[0]
        if part.path != _TASK_SOURCE_PATH:
            raise ValueError(f"Unknown task shard path: {part.path!r}")
        rank = int(part.start)
        if rank < 0 or rank >= self._num_tasks:
            raise ValueError(f"Invalid task rank {rank} for {self._num_tasks} tasks")
        yield DictRow({"task_rank": rank})

    def describe(self) -> dict[str, int]:
        return {"num_tasks": self._num_tasks}


__all__ = ["TaskSource"]
