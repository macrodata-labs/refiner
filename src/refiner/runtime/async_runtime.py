from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncIslandRuntime:
    """Process-local shared asyncio runtime used by async islands."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
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

            thread = threading.Thread(
                target=_run,
                name="refiner-async-island-runtime",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            if not self._ready.wait(10):
                raise RuntimeError("async island runtime failed to initialize")

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        self._ensure_started()
        loop = self._loop
        if loop is None:
            raise RuntimeError("async island runtime loop is not initialized")
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def shutdown(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        if loop.is_closed():
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        self._loop = None
        self._thread = None


_runtime = AsyncIslandRuntime()


def get_async_island_runtime() -> AsyncIslandRuntime:
    return _runtime


def submit(coro: Coroutine[Any, Any, T]) -> Future[T]:
    return _runtime.submit(coro)


atexit.register(_runtime.shutdown)


__all__ = [
    "AsyncIslandRuntime",
    "get_async_island_runtime",
    "submit",
]
