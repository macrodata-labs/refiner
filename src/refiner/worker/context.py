from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from refiner.worker.lifecycle.base import RuntimeLifecycle


@dataclass(frozen=True, slots=True)
class RunHandle:
    job_id: str
    stage_index: int
    client: Any | None = None
    workspace_slug: str | None = None
    worker_name: str | None = None
    worker_id: str | None = None

    def with_worker(
        self,
        *,
        worker_name: str | None = None,
        worker_id: str | None = None,
    ) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=self.stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=worker_name if worker_name is not None else self.worker_name,
            worker_id=worker_id if worker_id is not None else self.worker_id,
        )

    def with_stage(self, stage_index: int) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=self.worker_name,
            worker_id=self.worker_id,
        )


_ACTIVE_RUN_HANDLE: ContextVar[RunHandle | None] = ContextVar(
    "refiner_active_run_handle",
    default=None,
)
_ACTIVE_RUNTIME_LIFECYCLE: ContextVar["RuntimeLifecycle" | None] = ContextVar(
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


def get_active_runtime_lifecycle() -> RuntimeLifecycle | None:
    return _ACTIVE_RUNTIME_LIFECYCLE.get()


def get_active_step_index() -> int | None:
    return _ACTIVE_STEP_INDEX.get()


@contextmanager
def set_active_run_context(
    *,
    run_handle: RunHandle,
    runtime_lifecycle: RuntimeLifecycle,
) -> Generator[None, None, None]:
    run_token: Token[RunHandle | None] = _ACTIVE_RUN_HANDLE.set(run_handle)
    lifecycle_token: Token["RuntimeLifecycle" | None] = _ACTIVE_RUNTIME_LIFECYCLE.set(
        runtime_lifecycle
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
    "RunHandle",
    "get_active_run_handle",
    "get_active_runtime_lifecycle",
    "get_active_step_index",
    "set_active_run_context",
    "set_active_step_index",
]
