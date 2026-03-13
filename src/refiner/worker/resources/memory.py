from __future__ import annotations

import warnings
from collections.abc import Generator

from opentelemetry.metrics import CallbackOptions, Observation

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def set_memory_soft_limit_mb(mem_mb: int) -> tuple[int, int] | None:
    if resource is None:
        warnings.warn(
            "mem_mb_per_worker requested but resource limits are unavailable; running without memory limit",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    target_bytes = int(mem_mb) * 1024 * 1024
    if target_bytes <= 0:
        raise ValueError("mem_mb_per_worker must be > 0")

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        target_bytes = min(target_bytes, hard)

    resource.setrlimit(resource.RLIMIT_AS, (target_bytes, hard))
    return (soft, hard)


def restore_memory_soft_limit(previous: tuple[int, int]) -> None:
    if resource is None:
        return
    soft, hard = previous
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


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


def memory_limit_mb() -> float | None:
    memory_max = _read_cgroup_text("/sys/fs/cgroup/memory.max")
    if memory_max and memory_max != "max":
        value = _parse_int(memory_max)
        if value and value > 0:
            return float(value) / (1024.0 * 1024.0)

    legacy_limit = _parse_int(
        _read_cgroup_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    )
    if legacy_limit and legacy_limit > 0 and legacy_limit < (1 << 60):
        return float(legacy_limit) / (1024.0 * 1024.0)
    return None


def memory_observer_callback():
    process = psutil.Process() if psutil else None
    limit_mb = memory_limit_mb()

    def observe(
        _options: CallbackOptions,
    ) -> Generator[Observation, CallbackOptions, None]:
        used_mb = process.memory_info().rss / (1024.0 * 1024.0) if process else 0.0
        yield Observation(used_mb, {"kind": "used"})
        if limit_mb is not None:
            yield Observation(limit_mb, {"kind": "limit"})

    return observe


__all__ = [
    "memory_limit_mb",
    "memory_observer_callback",
    "restore_memory_soft_limit",
    "set_memory_soft_limit_mb",
]
