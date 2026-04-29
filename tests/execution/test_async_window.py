from __future__ import annotations

import asyncio
import time

from refiner.execution.asyncio.window import AsyncWindow


async def _delayed_value(value: int, delay_s: float) -> int:
    await asyncio.sleep(delay_s)
    return value


def test_async_window_submit_blocking_enforces_max_in_flight() -> None:
    window = AsyncWindow[int](max_in_flight=1, preserve_order=True)

    assert window.submit_blocking(_delayed_value(1, 0.03)) is None

    start = time.perf_counter()
    assert window.submit_blocking(_delayed_value(2, 0.0)) is None
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.02
    assert window.take_completed() == [1]
    assert window.drain() == [2]


def test_async_window_take_completed_returns_ready_results_without_blocking() -> None:
    window = AsyncWindow[int](max_in_flight=2, preserve_order=False)

    assert window.submit_blocking(_delayed_value(1, 0.03)) is None
    assert window.submit_blocking(_delayed_value(2, 0.0)) is None

    time.sleep(0.05)
    polled = window.take_completed()

    assert sorted(polled) == [1, 2]
    assert window.drain() == []


def test_async_window_ready_results_preserve_order() -> None:
    window = AsyncWindow[int](max_in_flight=2, preserve_order=True)

    assert window.submit_blocking(_delayed_value(1, 0.03)) is None
    assert window.submit_result(2) is None

    assert window.take_completed() == []
    assert window.drain() == [1, 2]
