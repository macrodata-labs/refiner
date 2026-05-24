from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import FlatMapFn, MapResult


HAND_TRACKING_FLUSH_COLUMN = "__refiner_hand_tracking_flush__"


HandTrackingBatchFn = Callable[[list[Row]], Iterable[MapResult]]
HandTrackingFlushPredicate = Callable[[Row], bool]


def run_hand_tracking(
    fn: HandTrackingBatchFn,
    *,
    batch_size: int,
    flush_when: HandTrackingFlushPredicate | None = None,
    include_flush_row: bool = False,
) -> FlatMapFn:
    """Build a buffered ``flat_map`` function for episode-level hand tracking.

    The returned function buffers episode rows and calls ``fn(batch)`` when the
    episode batch is full. It is meant for pipelines that keep one row per
    episode/video but still want to run fixed-size model batches inside the
    transform.

    Put model/runtime state in the closure captured by ``fn``. For example,
    initialize models lazily inside ``fn`` and reuse them across later calls.

    A flush sentinel can be handled by passing
    ``flush_when=is_hand_tracking_flush_row``. By default, that sentinel is not
    included in the model batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    buffer: list[Row] = []

    def apply(row: Row) -> Iterable[MapResult]:
        should_flush = bool(flush_when(row)) if flush_when is not None else False
        if not should_flush or include_flush_row:
            buffer.append(row)

        if not should_flush and len(buffer) < batch_size:
            return []

        return _run_buffered_batch(fn, buffer)

    return apply


def hand_tracking_flush_row(**values: Any) -> dict[str, Any]:
    """Create a sentinel row that flushes ``run_hand_tracking`` buffers."""

    return {HAND_TRACKING_FLUSH_COLUMN: True, **values}


def is_hand_tracking_flush_row(row: Row) -> bool:
    """Return whether a row is the hand-tracking flush sentinel."""

    return bool(row.get(HAND_TRACKING_FLUSH_COLUMN, False))


def _run_buffered_batch(
    fn: HandTrackingBatchFn,
    buffer: list[Row],
) -> list[MapResult]:
    if not buffer:
        return []
    batch = list(buffer)
    buffer.clear()
    return list(fn(batch))


__all__ = [
    "HAND_TRACKING_FLUSH_COLUMN",
    "HandTrackingBatchFn",
    "HandTrackingFlushPredicate",
    "hand_tracking_flush_row",
    "is_hand_tracking_flush_row",
    "run_hand_tracking",
]
