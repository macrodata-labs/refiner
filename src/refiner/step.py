from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeAlias

from .readers import Row


class RefinerStep(ABC):
    pass


MapResult: TypeAlias = Row | Mapping[str, Any] | None
MapFn: TypeAlias = Callable[[Row], MapResult]


def apply_map_result(row: Row, result: MapResult) -> Row | None:
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


def map_rows(rows: Iterator[Row], fn: MapFn) -> Iterator[Row]:
    """Apply a map-style function over a stream of rows (best-effort utility)."""
    for row in rows:
        out = apply_map_result(row, fn(row))
        if out is None:
            continue
        yield out
