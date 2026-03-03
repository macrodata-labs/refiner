from __future__ import annotations

from refiner.runtime.metrics_context import (
    get_active_step_index,
    get_active_user_metrics_emitter,
)
def is_not_empty(value: str) -> bool:
    return value.strip() != ""


def log_counter(
    label: str,
    value: float | int,
    shard_id: str,
) -> None:
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    if not is_not_empty(shard_id):
        raise ValueError("shard_id must be non-empty")

    if value < 0:
        raise ValueError("value must be >= 0")

    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_counter(
        label=label,
        value=value,
        shard_id=shard_id,
        step_index=get_active_step_index(),
        unit="sec",
    )


def log_gauge(
    label: str,
    value: float | int,
    shard_id: str,
) -> None:
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    if not is_not_empty(shard_id):
        raise ValueError("shard_id must be non-empty")
    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_gauge(
        label=label,
        value=value,
        shard_id=shard_id,
        step_index=get_active_step_index(),
        unit="meter",
    )


def log_histogram(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    unit: str | None = "Document",
) -> None:
    emitter = get_active_user_metrics_emitter()
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    if not is_not_empty(shard_id):
        raise ValueError("shard_id must be non-empty")
    value = float(value)
    emitter.emit_user_histogram(
        label=label,
        value=value,
        shard_id=shard_id,
        step_index=get_active_step_index(),
        unit=unit,
    )


__all__ = ["log_counter", "log_gauge", "log_histogram"]
