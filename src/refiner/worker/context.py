from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Generator, Protocol

from refiner.run import RunHandle

if TYPE_CHECKING:
    from refiner.platform.client.models import FinalizedShardWorker


class RuntimeLifecycleContext(Protocol):
    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]: ...


_ACTIVE_RUN_HANDLE: ContextVar[RunHandle | None] = ContextVar(
    "refiner_active_run_handle",
    default=None,
)
_ACTIVE_RUNTIME_LIFECYCLE: ContextVar[RuntimeLifecycleContext | None] = ContextVar(
    "refiner_active_runtime_lifecycle",
    default=None,
)
_ACTIVE_STEP_INDEX: ContextVar[int | None] = ContextVar(
    "refiner_active_step_index",
    default=None,
)


def get_active_run_handle() -> RunHandle:
    run_handle = _ACTIVE_RUN_HANDLE.get()
    if run_handle is None:
        return RunHandle(job_id="local", stage_index=0, worker_id="local")
    return run_handle


def get_active_runtime_lifecycle() -> RuntimeLifecycleContext | None:
    return _ACTIVE_RUNTIME_LIFECYCLE.get()


def get_active_step_index() -> int | None:
    return _ACTIVE_STEP_INDEX.get()


@contextmanager
def set_active_run_context(
    *,
    run_handle: RunHandle,
    runtime_lifecycle: RuntimeLifecycleContext,
) -> Generator[None, None, None]:
    run_token: Token[RunHandle | None] = _ACTIVE_RUN_HANDLE.set(run_handle)
    lifecycle_token: Token[RuntimeLifecycleContext | None] = (
        _ACTIVE_RUNTIME_LIFECYCLE.set(runtime_lifecycle)
    )
    try:
        yield
    finally:
        _ACTIVE_RUNTIME_LIFECYCLE.reset(lifecycle_token)
        _ACTIVE_RUN_HANDLE.reset(run_token)


@contextmanager
def set_active_step_index(step_index: int | None) -> Generator[None, None, None]:
    token: Token[int | None] = _ACTIVE_STEP_INDEX.set(step_index)
    try:
        yield
    finally:
        _ACTIVE_STEP_INDEX.reset(token)


__all__ = [
    "RuntimeLifecycleContext",
    "get_active_run_handle",
    "get_active_runtime_lifecycle",
    "get_active_step_index",
    "set_active_run_context",
    "set_active_step_index",
]
