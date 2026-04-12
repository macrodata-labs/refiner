from __future__ import annotations

import os
from collections.abc import Generator


def parse_cpu_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(x) for x in os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is None or count <= 0:
        raise RuntimeError("Unable to determine available CPUs")
    return list(range(int(count)))


def _read_cgroup_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def cpu_quota_percent() -> float | None:
    cpu_max = _read_cgroup_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, period, *_ = cpu_max.split()
        if quota != "max":
            quota_us = _parse_int(quota)
            period_us = _parse_int(period)
            if quota_us is not None and period_us and quota_us > 0 and period_us > 0:
                return (float(quota_us) / float(period_us)) * 100.0

    quota_us = _parse_int(_read_cgroup_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_us = _parse_int(_read_cgroup_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota_us is not None and period_us and quota_us > 0 and period_us > 0:
        return (float(quota_us) / float(period_us)) * 100.0
    return None


def cpu_observer_callback():
    from opentelemetry.metrics import CallbackOptions, Observation

    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    process = psutil.Process() if psutil else None
    quota_percent = cpu_quota_percent()

    def observe(
        _options: CallbackOptions,
    ) -> Generator[Observation, CallbackOptions, None]:
        yield Observation(
            process.cpu_percent(interval=None) if process else 0.0, {"kind": "used"}
        )
        if quota_percent is not None:
            yield Observation(quota_percent, {"kind": "quota"})

    return observe


__all__ = [
    "available_cpu_ids",
    "cpu_observer_callback",
    "cpu_quota_percent",
    "parse_cpu_ids",
]
