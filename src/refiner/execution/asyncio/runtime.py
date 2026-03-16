from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRuntime:
    """Process-local asyncio loop shared by async row steps."""

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

    def shutdown(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None or loop.is_closed():
            return

        async def _shutdown_loop() -> None:
            current = asyncio.current_task()
            tasks = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not current and not task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await loop.shutdown_asyncgens()
            await loop.shutdown_default_executor()
            loop.stop()

        if thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(_shutdown_loop(), loop)
            future.result(timeout=10.0)
            thread.join(timeout=10.0)
        self._loop = None
        self._thread = None


_runtime = AsyncRuntime()


def submit(coro: Coroutine[Any, Any, T]) -> Future[T]:
    return _runtime.submit(coro)


def shutdown() -> None:
    _runtime.shutdown()


atexit.register(_runtime.shutdown)


__all__ = ["AsyncRuntime", "shutdown", "submit"]
