from __future__ import annotations

from refiner.worker.metrics.otel import OtelTelemetryEmitter, _observable_gauge_name


class _FakeMeter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def create_observable_gauge(
        self,
        name: str,
        *,
        callbacks: list[object],
        unit: str | None = None,
    ) -> object:
        self.calls.append((name, unit))
        return {"name": name, "unit": unit, "callbacks": callbacks}


def test_observable_gauge_name_includes_metric_identity() -> None:
    assert (
        _observable_gauge_name(
            label="in_flight",
            kind=None,
            unit="rows",
            step_index=1,
        )
        == "refiner.user.observable_gauge.in_flight__rows__step_1"
    )
    assert (
        _observable_gauge_name(
            label="episodes_in_flight",
            kind=None,
            unit="episodes",
            step_index=None,
        )
        == "refiner.user.observable_gauge.episodes_in_flight__episodes"
    )


def test_register_user_gauge_uses_distinct_instrument_names() -> None:
    emitter = OtelTelemetryEmitter.__new__(OtelTelemetryEmitter)
    emitter._user_meter = _FakeMeter()
    emitter._user_observable_gauges = {}

    emitter.register_user_gauge(
        label="in_flight",
        callback=lambda: 1,
        kind=None,
        step_index=1,
        unit="rows",
    )
    emitter.register_user_gauge(
        label="episodes_in_flight",
        callback=lambda: 1,
        kind=None,
        step_index=None,
        unit="episodes",
    )

    assert emitter._user_meter.calls == [
        ("refiner.user.observable_gauge.in_flight__rows__step_1", "rows"),
        ("refiner.user.observable_gauge.episodes_in_flight__episodes", "episodes"),
    ]


def test_register_user_gauge_dedupes_same_identity() -> None:
    emitter = OtelTelemetryEmitter.__new__(OtelTelemetryEmitter)
    emitter._user_meter = _FakeMeter()
    emitter._user_observable_gauges = {}

    emitter.register_user_gauge(
        label="in_flight",
        callback=lambda: 1,
        kind=None,
        step_index=1,
        unit="rows",
    )
    emitter.register_user_gauge(
        label="in_flight",
        callback=lambda: 2,
        kind=None,
        step_index=1,
        unit="rows",
    )

    assert emitter._user_meter.calls == [
        ("refiner.user.observable_gauge.in_flight__rows__step_1", "rows")
    ]


def test_observable_gauge_name_is_capped() -> None:
    name = _observable_gauge_name(
        label="x" * 400,
        kind=None,
        unit="rows",
        step_index=1,
    )

    assert len(name) <= 255
    assert name.startswith("refiner.user.observable_gauge.")
