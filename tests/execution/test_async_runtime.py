from __future__ import annotations

import asyncio

from refiner.execution.asyncio.runtime import io, io_executor


def test_io_executor_returns_stable_thread_pool() -> None:
    assert io_executor() is io_executor()


def test_io_proxy_runs_work_via_run_in_executor() -> None:
    async def _run() -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(io, lambda: 7)

    assert asyncio.run(_run()) == 7
