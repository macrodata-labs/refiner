from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Generator

from loguru import logger as _base_logger

if TYPE_CHECKING:
    from refiner.services.manager import ServiceManager
    from refiner.worker.lifecycle import FinalizedShardWorker, RuntimeLifecycle


def worker_token_for(worker_id: str) -> str:
    digest = hashlib.blake2b(digest_size=6)
    digest.update(worker_id.encode("utf-8"))
    return digest.hexdigest()


_ACTIVE_JOB_ID: ContextVar[str] = ContextVar(
    "refiner_active_job_id",
    default="local",
)
_ACTIVE_STAGE_INDEX: ContextVar[int] = ContextVar(
    "refiner_active_stage_index",
    default=0,
)
_ACTIVE_STEP_INDEX: ContextVar[int | None] = ContextVar(
    "refiner_active_step_index",
    default=None,
)
_ACTIVE_WORKER_ID: ContextVar[str] = ContextVar(
    "refiner_active_worker_id",
    default="local",
)
_ACTIVE_WORKER_NAME: ContextVar[str | None] = ContextVar(
    "refiner_active_worker_name",
    default=None,
)
_ACTIVE_RUNTIME_LIFECYCLE: ContextVar[RuntimeLifecycle | None] = ContextVar(
    "refiner_active_runtime_lifecycle",
    default=None,
)
_ACTIVE_SERVICE_MANAGER: ContextVar["ServiceManager" | None] = ContextVar(
    "refiner_active_service_manager",
    default=None,
)
_ACTIVE_LOGGER: ContextVar[Any | None] = ContextVar(
    "refiner_active_logger",
    default=None,
)


class _ContextLogger:
    def __getattr__(self, name: str) -> Any:
        return getattr(_ACTIVE_LOGGER.get() or _base_logger, name)


logger = _ContextLogger()


def get_active_job_id() -> str:
    return _ACTIVE_JOB_ID.get()


def get_active_stage_index() -> int:
    return _ACTIVE_STAGE_INDEX.get()


def get_active_worker_id() -> str:
    return _ACTIVE_WORKER_ID.get()


def get_active_worker_name() -> str | None:
    return _ACTIVE_WORKER_NAME.get()


def get_active_worker_token() -> str:
    return worker_token_for(get_active_worker_id())


def get_finalized_workers(
    *, stage_index: int | None = None
) -> list["FinalizedShardWorker"]:
    runtime_lifecycle = _ACTIVE_RUNTIME_LIFECYCLE.get()
    if runtime_lifecycle is None:
        return []
    return runtime_lifecycle.finalized_workers(stage_index=stage_index)


def get_active_step_index() -> int | None:
    return _ACTIVE_STEP_INDEX.get()


def get_active_service_manager() -> ServiceManager | None:
    return _ACTIVE_SERVICE_MANAGER.get()


@contextmanager
def set_active_run_context(
    *,
    job_id: str,
    stage_index: int,
    worker_id: str,
    worker_name: str | None,
    runtime_lifecycle: RuntimeLifecycle,
    service_manager: ServiceManager | None = None,
) -> Generator[None, None, None]:
    job_token: Token[str] = _ACTIVE_JOB_ID.set(job_id)
    stage_token: Token[int] = _ACTIVE_STAGE_INDEX.set(stage_index)
    worker_id_token: Token[str] = _ACTIVE_WORKER_ID.set(worker_id)
    worker_name_token: Token[str | None] = _ACTIVE_WORKER_NAME.set(worker_name)
    lifecycle_token: Token[RuntimeLifecycle | None] = _ACTIVE_RUNTIME_LIFECYCLE.set(
        runtime_lifecycle
    )
    service_manager_token: Token[ServiceManager | None] = _ACTIVE_SERVICE_MANAGER.set(
        service_manager
    )
    logger_token: Token[Any | None] = _ACTIVE_LOGGER.set(
        _base_logger.bind(
            job_id=job_id,
            stage_index=stage_index,
            worker_id=worker_id,
            worker_name=worker_name,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_SERVICE_MANAGER.reset(service_manager_token)
        _ACTIVE_LOGGER.reset(logger_token)
        _ACTIVE_RUNTIME_LIFECYCLE.reset(lifecycle_token)
        _ACTIVE_WORKER_NAME.reset(worker_name_token)
        _ACTIVE_WORKER_ID.reset(worker_id_token)
        _ACTIVE_STAGE_INDEX.reset(stage_token)
        _ACTIVE_JOB_ID.reset(job_token)


@contextmanager
def set_active_step_index(step_index: int | None) -> Generator[None, None, None]:
    token: Token[int | None] = _ACTIVE_STEP_INDEX.set(step_index)
    try:
        yield
    finally:
        _ACTIVE_STEP_INDEX.reset(token)


__all__ = [
    "get_active_job_id",
    "get_active_service_manager",
    "get_active_stage_index",
    "get_active_step_index",
    "get_active_worker_id",
    "get_active_worker_name",
    "get_active_worker_token",
    "get_finalized_workers",
    "logger",
    "set_active_run_context",
    "set_active_step_index",
    "worker_token_for",
]
