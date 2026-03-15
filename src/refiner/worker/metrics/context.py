from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Generator, Protocol


class UserMetricsEmitter(Protocol):
    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        per: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def force_flush_user_metrics(self) -> None: ...

    def force_flush_resource_metrics(self) -> None: ...

    def force_flush_logs(self) -> None: ...

    def shutdown(self) -> None: ...


class _NoopUserMetricsEmitter(UserMetricsEmitter):
    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        pass

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        pass

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        per: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        pass

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


NOOP_USER_METRICS_EMITTER: UserMetricsEmitter = _NoopUserMetricsEmitter()
_ACTIVE_USER_METRICS_EMITTER: ContextVar[UserMetricsEmitter] = ContextVar(
    "refiner_active_user_metrics_emitter",
    default=NOOP_USER_METRICS_EMITTER,
)
_ACTIVE_STEP_INDEX: ContextVar[int | None] = ContextVar(
    "refiner_active_step_index",
    default=None,
)
_ACTIVE_WORKER_ID: ContextVar[str] = ContextVar(
    "refiner_active_worker_id",
    default="local",
)
_ACTIVE_RUNTIME_LIFECYCLE: ContextVar[Any | None] = ContextVar(
    "refiner_active_runtime_lifecycle",
    default=None,
)
_ACTIVE_RUNTIME_STAGE_INDEX: ContextVar[int | None] = ContextVar(
    "refiner_active_runtime_stage_index",
    default=None,
)


def get_active_user_metrics_emitter() -> UserMetricsEmitter:
    return _ACTIVE_USER_METRICS_EMITTER.get()


def get_active_step_index() -> int | None:
    return _ACTIVE_STEP_INDEX.get()


def get_active_worker_id() -> str:
    return _ACTIVE_WORKER_ID.get()


def get_active_runtime_lifecycle() -> Any | None:
    return _ACTIVE_RUNTIME_LIFECYCLE.get()


def get_active_runtime_stage_index() -> int | None:
    return _ACTIVE_RUNTIME_STAGE_INDEX.get()


@contextmanager
def set_active_user_metrics_emitter(
    emitter: UserMetricsEmitter,
) -> Generator[None, None, None]:
    token: Token[UserMetricsEmitter] = _ACTIVE_USER_METRICS_EMITTER.set(emitter)
    try:
        yield
    finally:
        _ACTIVE_USER_METRICS_EMITTER.reset(token)


@contextmanager
def set_active_step_index(step_index: int | None) -> Generator[None, None, None]:
    token: Token[int | None] = _ACTIVE_STEP_INDEX.set(step_index)
    try:
        yield
    finally:
        _ACTIVE_STEP_INDEX.reset(token)


@contextmanager
def set_active_worker_runtime(
    *,
    worker_id: str,
    runtime_lifecycle: Any,
    stage_index: int | None,
) -> Generator[None, None, None]:
    worker_token: Token[str] = _ACTIVE_WORKER_ID.set(worker_id)
    lifecycle_token: Token[Any | None] = _ACTIVE_RUNTIME_LIFECYCLE.set(
        runtime_lifecycle
    )
    stage_token: Token[int | None] = _ACTIVE_RUNTIME_STAGE_INDEX.set(stage_index)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME_STAGE_INDEX.reset(stage_token)
        _ACTIVE_RUNTIME_LIFECYCLE.reset(lifecycle_token)
        _ACTIVE_WORKER_ID.reset(worker_token)


__all__ = [
    "UserMetricsEmitter",
    "NOOP_USER_METRICS_EMITTER",
    "get_active_user_metrics_emitter",
    "get_active_step_index",
    "get_active_worker_id",
    "get_active_runtime_lifecycle",
    "get_active_runtime_stage_index",
    "set_active_step_index",
    "set_active_worker_runtime",
    "set_active_user_metrics_emitter",
]
