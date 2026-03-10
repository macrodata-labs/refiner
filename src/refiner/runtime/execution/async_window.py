from __future__ import annotations

import heapq
from collections.abc import Coroutine
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from refiner.runtime.async_runtime import submit

T = TypeVar("T")


@dataclass(slots=True)
class AsyncWindow(Generic[T]):
    max_in_flight: int
    preserve_order: bool = True
    _futures: set[Future[tuple[int, T]]] = field(default_factory=set, init=False)
    _ready: list[tuple[int, T]] = field(default_factory=list, init=False)
    _next_submit: int = field(default=0, init=False)
    _next_yield: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")

    @property
    def in_flight(self) -> int:
        return len(self._futures)

    @property
    def capacity(self) -> int:
        return max(0, self.max_in_flight - len(self._futures))

    def submit(self, coro: Coroutine[object, object, T]) -> None:
        idx = self._next_submit
        self._next_submit += 1

        async def _tagged() -> tuple[int, T]:
            return idx, await coro

        self._futures.add(submit(_tagged()))

    def drain(self, *, flush: bool) -> list[T]:
        out: list[T] = []
        while self._futures and (flush or len(self._futures) >= self.max_in_flight):
            done, pending = wait(
                self._futures,
                return_when=ALL_COMPLETED if flush else FIRST_COMPLETED,
            )
            self._futures = set(pending)
            if self.preserve_order:
                for future in done:
                    heapq.heappush(self._ready, future.result())
                while self._ready and self._ready[0][0] == self._next_yield:
                    _, value = heapq.heappop(self._ready)
                    self._next_yield += 1
                    out.append(value)
            else:
                for future in done:
                    _, value = future.result()
                    out.append(value)
        return out


__all__ = ["AsyncWindow"]
