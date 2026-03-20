from __future__ import annotations

from collections.abc import Callable

from refiner.worker.context import get_active_step_index
from refiner.worker.metrics.context import get_active_user_metrics_emitter


def is_not_empty(value: str) -> bool:
    return value.strip() != ""


def log_throughput(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    unit: str | None = None,
    step_index: int | None = None,
) -> None:
    """
    Used to measure the rate of "label" in "unit" over time.
    Example: if we just processed 6 documents since the last call to this function, we call
    log_throughput("documents_extracted", value=6, unit="docs", shard_id=...)
    Users will be able to see how many documents were processed in a given time period, as well as average rate
    """
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    if not is_not_empty(shard_id):
        raise ValueError("shard_id must be non-empty")

    if value < 0:
        raise ValueError("value must be >= 0")

    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_counter(
        label=label,
        value=float(value),
        shard_id=shard_id,
        step_index=step_index if step_index is not None else get_active_step_index(),
        unit=unit,
    )


def log_gauge(
    label: str,
    value: float | int,
    *,
    kind: str | None = None,
    unit: str | None = None,
    step_index: int | None = None,
) -> None:
    """
    Instantaneous measure of "label" in "unit" at the current point in time
    Example: current memory usage is 9gb out of 12gb:
    log_gauge("memory", 9, kind=usage, unit="GB")

    OR

    log_gauge("temperature", 65, unit="C")

    For multiple values for the same plot/quantity, either send separate requests or use log_gauges
    """
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    emitter = get_active_user_metrics_emitter()
    emitter.emit_user_gauge(
        label=label,
        value=float(value),
        kind=kind,
        step_index=step_index if step_index is not None else get_active_step_index(),
        unit=unit,
    )


def log_gauges(
    label: str,
    unit: str | None = None,
    step_index: int | None = None,
    **values: float | int,
) -> None:
    """
    Convenience method when you want to log multiple values for the same label
    log_gauges("memory", unit="GB", used=9, allocated=10)
    """
    for kind, value in values.items():
        log_gauge(label, value, kind=kind, unit=unit, step_index=step_index)


def register_gauge(
    label: str,
    callback: Callable[[], float | int],
    *,
    kind: str | None = None,
    unit: str | None = None,
    step_index: int | None = None,
) -> None:
    if not is_not_empty(label):
        raise ValueError("label must be non-empty")
    get_active_user_metrics_emitter().register_user_gauge(
        label=label,
        callback=callback,
        kind=kind,
        step_index=step_index if step_index is not None else get_active_step_index(),
        unit=unit,
    )


def log_histogram(
    label: str,
    value: float | int,
    shard_id: str,
    *,
    per: str = "row",
    unit: str | None = None,
    step_index: int | None = None,
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
        per=per,
        step_index=step_index if step_index is not None else get_active_step_index(),
        unit=unit,
    )


__all__ = [
    "log_throughput",
    "log_gauge",
    "log_gauges",
    "register_gauge",
    "log_histogram",
]
