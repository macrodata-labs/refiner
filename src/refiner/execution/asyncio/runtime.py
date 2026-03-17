from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def _default_io_workers() -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    return min(4, cpu_count)


class AsyncRuntime:
    """Process-local asyncio loop shared by async row steps."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._io_executor: ThreadPoolExecutor | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()

            def _run() -> None:
                loop = asyncio.new_event_loop()
                self._loop = loop
                asyncio.set_event_loop(loop)
                self._ready.set()
                try:
                    loop.run_forever()
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

            self._thread = threading.Thread(
                target=_run,
                name="refiner-async-runtime",
                daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(10):
                raise RuntimeError("async runtime failed to initialize")

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        self._ensure_started()
        loop = self._loop
        if loop is None:
            raise RuntimeError("async runtime loop is not initialized")
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def io_executor(self) -> ThreadPoolExecutor:
        executor = self._io_executor
        if executor is not None:
            return executor
        with self._lock:
            executor = self._io_executor
            if executor is not None:
                return executor
            executor = ThreadPoolExecutor(
                max_workers=_default_io_workers(),
                thread_name_prefix="refiner-io",
            )
            self._io_executor = executor
            return executor

    def shutdown(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None or loop.is_closed():
            executor = self._io_executor
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                self._io_executor = None
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        self._loop = None
        self._thread = None
        executor = self._io_executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._io_executor = None


_runtime = AsyncRuntime()


def submit(coro: Coroutine[Any, Any, T]) -> Future[T]:
    return _runtime.submit(coro)


def io_executor() -> ThreadPoolExecutor:
    return _runtime.io_executor()


atexit.register(_runtime.shutdown)


__all__ = ["AsyncRuntime", "io_executor", "submit"]
