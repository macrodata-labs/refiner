from __future__ import annotations

from refiner.pipeline.data.row import Row


class RowBuffer:
    __slots__ = ("_buf", "_head")

    def __init__(self):
        self._buf: list[Row] = []
        self._head = 0

    def __len__(self) -> int:
        return len(self._buf) - self._head

    def append(self, row: Row) -> None:
        self._buf.append(row)

    def extend(self, rows: list[Row]) -> None:
        if rows:
            self._buf.extend(rows)

    def take(self, n: int) -> list[Row]:
        if n <= 0:
            return []
        available = len(self)
        if available <= 0:
            return []
        if n > available:
            n = available
        start = self._head
        end = start + n
        out = self._buf[start:end]
        self._head = end
        self._maybe_compact()
        return out

    def take_all(self) -> list[Row]:
        return self.take(len(self))

    def peek(self, n: int) -> list[Row]:
        if n <= 0:
            return []
        available = len(self)
        if available <= 0:
            return []
        if n > available:
            n = available
        start = self._head
        end = start + n
        return self._buf[start:end]

    def discard(self, n: int) -> None:
        if n <= 0:
            return
        available = len(self)
        if available <= 0:
            return
        if n > available:
            n = available
        self._head += n
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        if self._head == 0:
            return
        if self._head >= 1024 and self._head * 2 >= len(self._buf):
            self._buf = self._buf[self._head :]
            self._head = 0
        elif self._head == len(self._buf):
            self._buf.clear()
            self._head = 0


__all__ = ["RowBuffer"]
