"""OpenTelemetry observable gauge callbacks for worker CPU, memory, and network."""

from __future__ import annotations

from typing import Generator

from opentelemetry.metrics import CallbackOptions, Observation

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cpu_quota_vcpu() -> float | None:
    # cgroup v2
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) >= 2 and parts[0] != "max":
            quota_us = _parse_int(parts[0])
            period_us = _parse_int(parts[1])
            if quota_us is not None and period_us and quota_us > 0 and period_us > 0:
                return float(quota_us) / float(period_us)

    # cgroup v1
    quota_us = _parse_int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_us = _parse_int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota_us is not None and period_us and quota_us > 0 and period_us > 0:
        return float(quota_us) / float(period_us)
    return None


def _memory_limit_mb() -> float | None:
    # cgroup v2
    memory_max = _read_text("/sys/fs/cgroup/memory.max")
    if memory_max and memory_max != "max":
        bytes_value = _parse_int(memory_max)
        if bytes_value and bytes_value > 0:
            return float(bytes_value) / (1024.0 * 1024.0)

    # cgroup v1
    memory_limit = _parse_int(_read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if memory_limit and memory_limit > 0 and memory_limit < (1 << 60):
        return float(memory_limit) / (1024.0 * 1024.0)
    return None


def get_cpu_callback():
    process = psutil.Process() if psutil else None
    quota_vcpu = _cpu_quota_vcpu()

    def get_cpu(
        _options: CallbackOptions,
    ) -> Generator[Observation, CallbackOptions, None]:
        # psutil cpu_percent is "percent of one full core" for this process.
        used_pct = process.cpu_percent(interval=None) if process else 0.0
        yield Observation(used_pct, {"kind": "used"})
        if quota_vcpu is not None:
            yield Observation(quota_vcpu * 100.0, {"kind": "quota"})

    return get_cpu


def get_memory_callback():
    process = psutil.Process() if psutil else None
    memory_limit_mb = _memory_limit_mb()

    def get_memory(
        _options: CallbackOptions,
    ) -> Generator[Observation, CallbackOptions, None]:
        used_mb = process.memory_info().rss / (1024.0 * 1024.0) if process else 0.0
        yield Observation(used_mb, {"kind": "used"})
        if memory_limit_mb is not None:
            yield Observation(memory_limit_mb, {"kind": "limit"})

    return get_memory


def get_network_in(
    options: CallbackOptions,
) -> Generator[Observation, CallbackOptions, None]:
    if psutil:
        net = psutil.net_io_counters()  # This is actually per whole machine
        value = float(net.bytes_recv) if net else 0.0
    else:
        value = 0.0
    yield Observation(value, {})


def get_network_out(
    options: CallbackOptions,
) -> Generator[Observation, CallbackOptions, None]:
    del options
    if psutil:
        net = psutil.net_io_counters()  # This is actually per whole machine
        value = float(net.bytes_sent) if net else 0.0
    else:
        value = 0.0
    yield Observation(value, {})
