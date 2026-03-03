from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Generator, Protocol


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
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
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
        shard_id: str,
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


def get_active_user_metrics_emitter() -> UserMetricsEmitter:
    return _ACTIVE_USER_METRICS_EMITTER.get()


def get_active_step_index() -> int | None:
    return _ACTIVE_STEP_INDEX.get()


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


__all__ = [
    "UserMetricsEmitter",
    "NOOP_USER_METRICS_EMITTER",
    "get_active_user_metrics_emitter",
    "get_active_step_index",
    "set_active_step_index",
    "set_active_user_metrics_emitter",
]
