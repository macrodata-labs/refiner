from __future__ import annotations

import asyncio
import time

import pytest

from refiner.execution.asyncio.window import AsyncWindow


async def _delayed_value(value: int, delay_s: float) -> int:
    await asyncio.sleep(delay_s)
    return value


async def _delayed_failure(delay_s: float) -> int:
    await asyncio.sleep(delay_s)
    raise RuntimeError("request failed")


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

    deadline = time.perf_counter() + 1.0
    polled: list[int] = []
    while time.perf_counter() < deadline and len(polled) < 2:
        polled.extend(window.take_completed())
        if len(polled) < 2:
            time.sleep(0.01)

    assert sorted(polled) == [1, 2]
    assert window.drain() == []


def test_async_window_ready_results_preserve_order() -> None:
    window = AsyncWindow[int](max_in_flight=2, preserve_order=True)

    assert window.submit_blocking(_delayed_value(1, 0.03)) is None
    assert window.submit_result(2) is None

    assert window.take_completed() == []
    assert window.drain() == [1, 2]


def test_async_window_drain_fails_fast_and_cancels_pending() -> None:
    window = AsyncWindow[int](max_in_flight=2, preserve_order=True)

    assert window.submit_blocking(_delayed_failure(0.03)) is None
    assert window.submit_blocking(_delayed_value(2, 1.0)) is None

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="request failed"):
        window.drain()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5
    assert len(window) == 0


def test_async_window_cancel_pending_clears_window() -> None:
    window = AsyncWindow[int](max_in_flight=2, preserve_order=True)

    assert window.submit_blocking(_delayed_value(1, 1.0)) is None
    assert window.submit_result(2) is None

    window.cancel_pending()

    assert len(window) == 0
    assert window.take_completed() == []
    assert window.drain() == []
