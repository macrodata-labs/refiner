from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from refiner.readers.row import DictRow
from refiner.readers import Row


class RefinerStep(ABC):
    """Base marker for executable processing steps."""


MapResult: TypeAlias = Row | Mapping[str, Any] | None
MapFn: TypeAlias = Callable[[Row], MapResult]
BatchItem: TypeAlias = Row | Mapping[str, Any] | None
BatchFn: TypeAlias = Callable[[list[Row]], Iterable[BatchItem]]


class RowStep(RefinerStep, ABC):
    @abstractmethod
    def apply_row(self, row: Row) -> MapResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnRowStep(RowStep):
    fn: MapFn

    def apply_row(self, row: Row) -> MapResult:
        return self.fn(row)


class BatchStep(RefinerStep, ABC):
    batch_size: int

    @abstractmethod
    def apply_batch(self, rows: list[Row]) -> Iterable[BatchItem]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnBatchStep(BatchStep):
    fn: BatchFn
    batch_size: int

    def __post_init__(self) -> None:
        if self.batch_size <= 1:
            raise ValueError("batch_size for batch steps must be > 1")

    def apply_batch(self, rows: list[Row]) -> Iterable[BatchItem]:
        for i in range(0, len(rows), self.batch_size):
            yield from self.fn(rows[i : i + self.batch_size])


def normalize_row_result(row: Row, result: MapResult) -> Row | None:
    """Normalize a user map() function's output.

    Contract:
        - None => drop the row
        - Row  => replace the row
        - Mapping[str, Any] => treated as a patch to merge into the input row
    """

    if result is None:
        return None
    if isinstance(result, Row):
        return result
    if isinstance(result, Mapping):
        return row.update(result)
    raise TypeError(f"Unsupported map() result type: {type(result)!r}")


def normalize_batch_item(item: BatchItem) -> Row | None:
    """Normalize a batch step item into a Row."""
    if item is None:
        return None
    if isinstance(item, Row):
        return item
    if isinstance(item, Mapping):
        return DictRow(item)
    raise TypeError(f"Unsupported batch item type: {type(item)!r}")


__all__ = [
    "RefinerStep",
    "RowStep",
    "BatchStep",
    "FnRowStep",
    "FnBatchStep",
    "MapResult",
    "MapFn",
    "BatchItem",
    "BatchFn",
    "normalize_row_result",
    "normalize_batch_item",
]
