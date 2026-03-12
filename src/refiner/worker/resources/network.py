from __future__ import annotations

from collections.abc import Generator

from opentelemetry.metrics import CallbackOptions, Observation

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def network_in_observer_callback(
    _options: CallbackOptions,
) -> Generator[Observation, CallbackOptions, None]:
    counters = psutil.net_io_counters() if psutil else None
    yield Observation(float(counters.bytes_recv) if counters else 0.0, {})


def network_out_observer_callback(
    _options: CallbackOptions,
) -> Generator[Observation, CallbackOptions, None]:
    counters = psutil.net_io_counters() if psutil else None
    yield Observation(float(counters.bytes_sent) if counters else 0.0, {})


__all__ = ["network_in_observer_callback", "network_out_observer_callback"]
