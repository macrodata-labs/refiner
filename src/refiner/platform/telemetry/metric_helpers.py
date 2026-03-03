"""OpenTelemetry observable gauge callbacks for worker CPU, memory, and network."""

from __future__ import annotations

from typing import Generator

from opentelemetry.metrics import CallbackOptions, Observation

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def get_cpu_usage_callback():
    process = psutil.Process() if psutil else None

    def get_cpu_usage(
        options: CallbackOptions,
    ) -> Generator[Observation, CallbackOptions, None]:
        value = process.cpu_percent(interval=None) if process else 0.0
        yield Observation(value, {})

    return get_cpu_usage


def get_memory_usage_callback():
    process = psutil.Process() if psutil else None
    def get_memory_usage(options: CallbackOptions) -> Generator[Observation, CallbackOptions, None]:
        value = process.memory_info().rss / (1024 * 1024) if process else 0.0
        yield Observation(value, {})

    return get_memory_usage


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
