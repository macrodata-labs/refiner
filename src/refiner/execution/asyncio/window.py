from __future__ import annotations

import heapq
from collections.abc import Coroutine
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from refiner.execution.asyncio.runtime import submit

T = TypeVar("T")


@dataclass(slots=True)
class AsyncWindow(Generic[T]):
    """Bound concurrent coroutine execution while optionally preserving output order.

    Fields:
        max_in_flight: Maximum number of submitted futures allowed before submit
            blocks for at least one completion.
        preserve_order: If true, completed values are yielded in submit order.
        _futures: Futures currently running in the shared async runtime.
        _ready: Completed values that can be returned immediately. This is used
            for unordered completions and for values unblocked by submit backpressure.
        _drain_queue: Min-heap of ordered completions keyed by submit index. Values
            stay here until every earlier submission has been yielded.
        _next_submit: Submit index to assign to the next coroutine or direct result.
        _next_yield: Next submit index allowed to leave `_drain_queue`.
    """

    max_in_flight: int
    preserve_order: bool = True
    _futures: set[Future[tuple[int, T]]] = field(default_factory=set, init=False)
    _ready: list[T] = field(default_factory=list, init=False)
    _drain_queue: list[tuple[int, T]] = field(default_factory=list, init=False)
    _next_submit: int = field(default=0, init=False)
    _next_yield: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")

    def submit_blocking(self, coro: Coroutine[object, object, T]) -> None:
        """Submit a coroutine, blocking first if the in-flight window is full."""
        if len(self._futures) >= self.max_in_flight:
            self._wait_until(return_when=FIRST_COMPLETED)

        idx = self._next_submit
        self._next_submit += 1

        async def _tagged() -> tuple[int, T]:
            return idx, await coro

        self._futures.add(submit(_tagged()))

    def submit_result(self, value: T) -> None:
        """Submit an already-computed value through the same ordering path."""
        idx = self._next_submit
        self._next_submit += 1
        self._store_result(idx, value)

    def take_completed(self) -> list[T]:
        """Return currently available results without waiting for more futures."""
        self._collect_done({future for future in self._futures if future.done()})
        return self._take_ready()

    def drain(self) -> list[T]:
        """Wait for all in-flight work, then return every available result."""
        self._wait_until(return_when=ALL_COMPLETED)
        return self._take_ready()

    def cancel_pending(self) -> None:
        """Best-effort cancel of in-flight work and discard buffered results."""
        for future in self._futures:
            future.cancel()
        self._futures.clear()
        self._ready.clear()
        self._drain_queue.clear()

    def _collect_done(self, done: set[Future[tuple[int, T]]]) -> None:
        """Remove completed futures from the window and store their results."""
        if not done:
            return
        self._futures.difference_update(done)
        for future in done:
            idx, value = future.result()
            self._store_result(idx, value)

    def _wait_until(self, *, return_when: str) -> None:
        """Block until the requested completion condition and collect results."""
        while self._futures and (
            return_when == ALL_COMPLETED or len(self._futures) >= self.max_in_flight
        ):
            done, _ = wait(self._futures, return_when=return_when)
            self._collect_done(done)

    def _store_result(self, idx: int, value: T) -> None:
        """Route one completed value into `_ready`, respecting submit order."""
        if not self.preserve_order:
            self._ready.append(value)
            return

        heapq.heappush(self._drain_queue, (idx, value))
        while self._drain_queue and self._drain_queue[0][0] == self._next_yield:
            _, value = heapq.heappop(self._drain_queue)
            self._next_yield += 1
            self._ready.append(value)

    def _take_ready(self) -> list[T]:
        """Return and clear the ready-to-yield result buffer."""
        out = self._ready
        self._ready = []
        return out

    def __len__(self) -> int:
        return len(self._futures)


__all__ = ["AsyncWindow"]
