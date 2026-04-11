from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from loguru import logger as _base_logger

if TYPE_CHECKING:
    from loguru import Record
    from refiner.platform.client.api import MacrodataClient
    from refiner.worker.lifecycle.base import RuntimeLifecycle


@dataclass(frozen=True, slots=True)
class RunHandle:
    job_id: str
    stage_index: int
    worker_id: str
    client: "MacrodataClient" | None = None
    workspace_slug: str | None = None
    worker_name: str | None = None

    @staticmethod
    def worker_token_for(worker_id: str) -> str:
        digest = hashlib.blake2b(digest_size=6)
        digest.update(worker_id.encode("utf-8"))
        return digest.hexdigest()

    @property
    def worker_token(self) -> str:
        return self.worker_token_for(self.worker_id)


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


def _patch_log_record(record: "Record") -> None:
    run_handle = _ACTIVE_RUN_HANDLE.get()
    if run_handle is None:
        return
    extra = record["extra"]
    if not isinstance(extra, dict):
        return
    extra.setdefault("job_id", run_handle.job_id)
    extra.setdefault("stage_index", int(run_handle.stage_index))
    extra.setdefault("worker_id", run_handle.worker_id)
    if run_handle.worker_name is not None:
        extra.setdefault("worker_name", run_handle.worker_name)


logger = _base_logger.patch(_patch_log_record)


def get_active_run_handle() -> RunHandle:
    run_handle = _ACTIVE_RUN_HANDLE.get()
    if run_handle is None:
        raise RuntimeError("no active run context")
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
    "logger",
    "set_active_run_context",
    "set_active_step_index",
]
