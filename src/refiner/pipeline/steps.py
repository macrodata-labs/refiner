from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from refiner.pipeline.expressions import Expr
from refiner.pipeline.data.row import DictRow, Row


class RefinerStep(ABC):
    """Base marker for executable processing steps."""

    index: int
    op_name: str | None = None


MapResult: TypeAlias = Row | Mapping[str, Any]
MapFn: TypeAlias = Callable[[Row], MapResult]
AsyncMapFn: TypeAlias = Callable[[Row], Awaitable[MapResult] | MapResult]
PredicateFn: TypeAlias = Callable[[Row], bool]
BatchItem: TypeAlias = Row | Mapping[str, Any] | None
BatchFn: TypeAlias = Callable[[list[Row]], Iterable[BatchItem]]
FlatMapFn: TypeAlias = Callable[[Row], Iterable[BatchItem]]
AsyncMapFn: TypeAlias = Callable[[Row], Awaitable[MapResult] | MapResult]


class RowStep(RefinerStep, ABC):
    @abstractmethod
    def apply_row(self, row: Row) -> MapResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnRowStep(RowStep):
    fn: MapFn
    index: int
    op_name: str | None = None

    def apply_row(self, row: Row) -> MapResult:
        return self.fn(row)


class AsyncRowStep(RefinerStep, ABC):
    max_in_flight: int
    preserve_order: bool

    @abstractmethod
    def apply_row_async(self, row: Row) -> Awaitable[MapResult] | MapResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnAsyncRowStep(AsyncRowStep):
    fn: AsyncMapFn
    index: int
    max_in_flight: int = 16
    preserve_order: bool = True
    op_name: str | None = None

    def __post_init__(self) -> None:
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")

    def apply_row_async(self, row: Row) -> Awaitable[MapResult] | MapResult:
        return self.fn(row)


class BatchStep(RefinerStep, ABC):
    batch_size: int

    @abstractmethod
    def apply_batch(self, rows: list[Row]) -> Iterable[BatchItem]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnBatchStep(BatchStep):
    fn: BatchFn
    index: int
    batch_size: int
    op_name: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 1:
            raise ValueError("batch_size for batch steps must be > 1")

    def apply_batch(self, rows: list[Row]) -> Iterable[BatchItem]:
        for i in range(0, len(rows), self.batch_size):
            yield from self.fn(rows[i : i + self.batch_size])


class FlatMapStep(RefinerStep, ABC):
    @abstractmethod
    def apply_row_many(self, row: Row) -> Iterable[BatchItem]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnFlatMapStep(FlatMapStep):
    fn: FlatMapFn
    index: int
    op_name: str | None = None

    def apply_row_many(self, row: Row) -> Iterable[BatchItem]:
        return self.fn(row)


@dataclass(frozen=True, slots=True)
class FilterRowStep(RefinerStep):
    predicate: PredicateFn
    index: int
    op_name: str | None = "filter"

    def apply_predicate(self, row: Row) -> bool:
        return bool(self.predicate(row))


@dataclass(frozen=True, slots=True)
class SelectStep(RefinerStep):
    columns: tuple[str, ...]
    index: int
    op_name: str | None = "select"


@dataclass(frozen=True, slots=True)
class WithColumnsStep(RefinerStep):
    assignments: Mapping[str, Expr]
    index: int
    op_name: str | None = "with_columns"


@dataclass(frozen=True, slots=True)
class DropStep(RefinerStep):
    columns: tuple[str, ...]
    index: int
    op_name: str | None = "drop"


@dataclass(frozen=True, slots=True)
class RenameStep(RefinerStep):
    mapping: Mapping[str, str]
    index: int
    op_name: str | None = "rename"


@dataclass(frozen=True, slots=True)
class CastStep(RefinerStep):
    dtypes: Mapping[str, str]
    index: int
    op_name: str | None = "cast"


@dataclass(frozen=True, slots=True)
class FilterExprStep(RefinerStep):
    predicate: Expr
    index: int
    op_name: str | None = "filter"


VectorizedOp: TypeAlias = (
    SelectStep | WithColumnsStep | DropStep | RenameStep | CastStep | FilterExprStep
)


@dataclass(frozen=True, slots=True)
class VectorizedSegmentStep(RefinerStep):
    """A fused shard-local vectorized segment.

    Adjacent expression-backed operations are fused during pipeline construction so
    row->Arrow and Arrow->row conversion happens only once per segment.
    """

    ops: tuple[VectorizedOp, ...]
    op_name: str | None = "vectorized"


def normalize_row_result(row: Row, result: MapResult) -> Row:
    """Normalize a user map() function's output.

    Contract:
        - Row  => replace the row
        - Mapping[str, Any] => treated as a patch to merge into the input row
    """

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
    "AsyncRowStep",
    "BatchStep",
    "FlatMapStep",
    "FnRowStep",
    "AsyncRowStep",
    "FnAsyncRowStep",
    "FnBatchStep",
    "FnFlatMapStep",
    "FilterRowStep",
    "MapResult",
    "MapFn",
    "AsyncMapFn",
    "PredicateFn",
    "BatchItem",
    "BatchFn",
    "FlatMapFn",
    "AsyncMapFn",
    "SelectStep",
    "WithColumnsStep",
    "DropStep",
    "RenameStep",
    "CastStep",
    "FilterExprStep",
    "VectorizedOp",
    "VectorizedSegmentStep",
    "normalize_row_result",
    "normalize_batch_item",
]
