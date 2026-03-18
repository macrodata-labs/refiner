from __future__ import annotations

from collections.abc import Generator

from opentelemetry.metrics import CallbackOptions, Observation

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


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
]
