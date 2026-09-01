from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import pyarrow as pa

from refiner.pipeline.expressions import Expr
from refiner.pipeline.data.datatype import DTypeMapping
from refiner.pipeline.data.row import Row


class RefinerStep(ABC):
    """Base marker for executable processing steps."""

    index: int
    op_name: str | None = None


MapResult: TypeAlias = Row | dict[str, Any]
MapFn: TypeAlias = Callable[[Row], MapResult]
AsyncMapFn: TypeAlias = Callable[[Row], Awaitable[MapResult] | MapResult]
AsyncMapFactory: TypeAlias = Callable[[], AsyncMapFn]
PredicateFn: TypeAlias = Callable[[Row], bool]
BatchFn: TypeAlias = Callable[[list[Row]], Iterable[Row]]
BatchFactory: TypeAlias = Callable[[], BatchFn]
AsyncBatchFn: TypeAlias = Callable[
    [list[Row]], Awaitable[Iterable[Row]] | Iterable[Row]
]
AsyncBatchFactory: TypeAlias = Callable[[], AsyncBatchFn]
FlatMapFn: TypeAlias = Callable[[Row], Iterable[MapResult]]
TableResult: TypeAlias = pa.Table
TableFn: TypeAlias = Callable[[pa.Table], TableResult]
TableFactory: TypeAlias = Callable[[], TableFn]


def _validate_fn_or_factory(
    fn: Callable[..., Any] | None,
    factory: Callable[[], Callable[..., Any]] | None,
    *,
    op_name: str,
) -> None:
    if (fn is None) == (factory is None):
        raise ValueError(f"{op_name} requires exactly one of fn or factory")


class RowStep(RefinerStep, ABC):
    @abstractmethod
    def apply_row(self, row: Row) -> MapResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnRowStep(RowStep):
    fn: MapFn
    index: int
    op_name: str | None = None
    dtypes: DTypeMapping | None = None

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
    fn: AsyncMapFn | None
    index: int
    max_in_flight: int = 16
    preserve_order: bool = True
    op_name: str | None = None
    dtypes: DTypeMapping | None = None
    factory: AsyncMapFactory | None = None

    def __post_init__(self) -> None:
        _validate_fn_or_factory(self.fn, self.factory, op_name="map_async")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")

    def apply_row_async(self, row: Row) -> Awaitable[MapResult] | MapResult:
        if self.fn is None:
            raise RuntimeError("map_async factory was not initialized")
        return self.fn(row)


class BatchStep(RefinerStep, ABC):
    batch_size: int

    @abstractmethod
    def apply_batch(self, rows: list[Row]) -> Iterable[Row]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnBatchStep(BatchStep):
    fn: BatchFn | None
    index: int
    batch_size: int
    op_name: str | None = None
    dtypes: DTypeMapping | None = None
    factory: BatchFactory | None = None

    def __post_init__(self) -> None:
        _validate_fn_or_factory(self.fn, self.factory, op_name="batch_map")
        if self.batch_size <= 1:
            raise ValueError("batch_size for batch steps must be > 1")

    def apply_batch(self, rows: list[Row]) -> Iterable[Row]:
        if self.fn is None:
            raise RuntimeError("batch_map factory was not initialized")
        for i in range(0, len(rows), self.batch_size):
            yield from self.fn(rows[i : i + self.batch_size])


class AsyncBatchStep(RefinerStep, ABC):
    batch_size: int
    max_in_flight: int
    preserve_order: bool

    @abstractmethod
    def apply_batch_async(
        self, rows: list[Row]
    ) -> Awaitable[Iterable[Row]] | Iterable[Row]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnAsyncBatchStep(AsyncBatchStep):
    fn: AsyncBatchFn | None
    index: int
    batch_size: int
    max_in_flight: int = 2
    preserve_order: bool = True
    op_name: str | None = None
    dtypes: DTypeMapping | None = None
    factory: AsyncBatchFactory | None = None

    def __post_init__(self) -> None:
        _validate_fn_or_factory(self.fn, self.factory, op_name="batch_map_async")
        if self.batch_size <= 1:
            raise ValueError("batch_size for async batch steps must be > 1")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")

    def apply_batch_async(
        self, rows: list[Row]
    ) -> Awaitable[Iterable[Row]] | Iterable[Row]:
        if self.fn is None:
            raise RuntimeError("batch_map_async factory was not initialized")
        return self.fn(rows)


class FlatMapStep(RefinerStep, ABC):
    @abstractmethod
    def apply_row_many(self, row: Row) -> Iterable[MapResult]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FnFlatMapStep(FlatMapStep):
    fn: FlatMapFn
    index: int
    op_name: str | None = None
    dtypes: DTypeMapping | None = None

    def apply_row_many(self, row: Row) -> Iterable[MapResult]:
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
    dtypes: DTypeMapping
    index: int
    op_name: str | None = "cast"


@dataclass(frozen=True, slots=True)
class FilterExprStep(RefinerStep):
    predicate: Expr
    index: int
    op_name: str | None = "filter"


@dataclass(frozen=True, slots=True)
class FnTableStep(RefinerStep):
    fn: TableFn | None
    index: int
    op_name: str | None = "map_table"
    factory: TableFactory | None = None

    def __post_init__(self) -> None:
        _validate_fn_or_factory(self.fn, self.factory, op_name="map_table")

    def apply_table(self, table: pa.Table) -> pa.Table:
        if self.fn is None:
            raise RuntimeError("map_table factory was not initialized")
        return self.fn(table)


VectorizedOp: TypeAlias = (
    SelectStep
    | WithColumnsStep
    | DropStep
    | RenameStep
    | CastStep
    | FilterExprStep
    | FnTableStep
)


@dataclass(frozen=True, slots=True)
class VectorizedSegmentStep(RefinerStep):
    """A fused shard-local vectorized segment.

    Adjacent expression-backed operations are fused during pipeline construction so
    row->Arrow and Arrow->row conversion happens only once per segment.
    """

    ops: tuple[VectorizedOp, ...]
    op_name: str | None = "vectorized"


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
    "AsyncBatchStep",
    "FnAsyncBatchStep",
    "FnFlatMapStep",
    "FilterRowStep",
    "MapResult",
    "MapFn",
    "AsyncMapFn",
    "AsyncMapFactory",
    "PredicateFn",
    "BatchFn",
    "BatchFactory",
    "AsyncBatchFn",
    "AsyncBatchFactory",
    "FlatMapFn",
    "TableResult",
    "TableFn",
    "TableFactory",
    "SelectStep",
    "WithColumnsStep",
    "DropStep",
    "RenameStep",
    "CastStep",
    "FilterExprStep",
    "FnTableStep",
    "VectorizedOp",
    "VectorizedSegmentStep",
]
