from __future__ import annotations

from refiner.runtime.metrics_context import (
    get_active_step_index,
    get_active_user_metrics_emitter,
)


def _validate_label(label: str) -> str:
    normalized = label.strip()
    if not normalized:
        raise ValueError("label must be non-empty")
    return normalized


def _validate_shard_id(shard_id: str) -> str:
    normalized = shard_id.strip()
    if not normalized:
        raise ValueError("shard_id must be non-empty")
    return normalized


def _validate_value(value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be a number")
    return float(value)


def _validate_unit(unit: str | None, label: str) -> str:
    if unit is None:
        return label
    if not isinstance(unit, str):
        raise TypeError("unit must be a string")
    normalized = unit.strip()
    if not normalized:
        raise ValueError("unit must be non-empty when provided")
    return normalized


def metric_counter(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    unit: str | None = None,
) -> None:
    normalized_label = _validate_label(label)
    normalized_shard_id = _validate_shard_id(shard_id)
    normalized_value = _validate_value(value)
    if normalized_value < 0:
        raise ValueError("counter value must be >= 0")
    normalized_unit = _validate_unit(unit, normalized_label)
    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_counter(
        label=normalized_label,
        value=normalized_value,
        shard_id=normalized_shard_id,
        step_index=get_active_step_index(),
        unit=normalized_unit,
    )


def metric_gauge(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    unit: str | None = None,
) -> None:
    normalized_label = _validate_label(label)
    normalized_shard_id = _validate_shard_id(shard_id)
    normalized_value = _validate_value(value)
    normalized_unit = _validate_unit(unit, normalized_label)
    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_gauge(
        label=normalized_label,
        value=normalized_value,
        shard_id=normalized_shard_id,
        step_index=get_active_step_index(),
        unit=normalized_unit,
    )


def metric_histogram(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    unit: str | None = None,
) -> None:
    normalized_label = _validate_label(label)
    normalized_shard_id = _validate_shard_id(shard_id)
    normalized_value = _validate_value(value)
    normalized_unit = _validate_unit(unit, normalized_label)
    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_histogram(
        label=normalized_label,
        value=normalized_value,
        shard_id=normalized_shard_id,
        step_index=get_active_step_index(),
        unit=normalized_unit,
    )


__all__ = ["metric_counter", "metric_gauge", "metric_histogram"]
