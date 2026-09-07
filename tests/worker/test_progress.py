from __future__ import annotations

from io import StringIO
from importlib import import_module

import pytest

import refiner as mdr
from refiner.worker.metrics.emitter import UserMetricsEmitter


class _RecordingEmitter(UserMetricsEmitter):
    def __init__(self) -> None:
        self.gauges: list[dict[str, object]] = []

    def emit_user_gauge(self, **kwargs: object) -> None:
        self.gauges.append(kwargs)


def test_progress_wraps_iterable_and_emits_absolute_snapshots(monkeypatch) -> None:
    emitter = _RecordingEmitter()
    module = import_module("refiner.progress_api")
    monkeypatch.setattr(module, "get_active_user_metrics_emitter", lambda: emitter)
    monkeypatch.setattr(module, "get_active_step_index", lambda: 4)

    tracker = mdr.progress(
        ["a", "b", "c"],
        desc="Colorizing",
        unit="frame",
        mininterval=0,
        disable=True,
    )

    assert list(tracker) == ["a", "b", "c"]
    assert tracker.n == 3
    assert tracker.total == 3
    completed = [
        call["value"] for call in emitter.gauges if call["kind"] == "completed"
    ]
    assert completed == [0.0, 1.0, 2.0, 3.0, 3.0]
    assert all(call["label"] == "progress.Colorizing" for call in emitter.gauges)
    assert all(call["step_index"] == 4 for call in emitter.gauges)


def test_progress_supports_manual_tqdm_style_updates_and_postfix(monkeypatch) -> None:
    emitter = _RecordingEmitter()
    module = import_module("refiner.progress_api")
    monkeypatch.setattr(module, "get_active_user_metrics_emitter", lambda: emitter)
    output = StringIO()

    with mdr.progress(
        total=10,
        desc="Encoding",
        unit="video",
        mininterval=0,
        file=output,
        disable=False,
    ) as tracker:
        tracker.update(4)
        tracker.set_description("Encoding camera 2")
        tracker.set_postfix(gpu=72, state="active")

    assert tracker.n == 4
    assert "Encoding camera 2: 4/10 video" in output.getvalue()
    assert "gpu=72" in output.getvalue()
    assert output.getvalue().endswith("\n")
    postfix = [call for call in emitter.gauges if call["kind"] == "postfix.gpu"]
    assert postfix[-1]["value"] == 72.0
    assert not any(call["kind"] == "postfix.state" for call in emitter.gauges)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total": -1}, "total must be >= 0"),
        ({"initial": -1}, "initial must be >= 0"),
        ({"mininterval": -1}, "mininterval must be >= 0"),
        ({"unit": ""}, "unit must be non-empty"),
    ],
)
def test_progress_validates_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        mdr.progress(disable=True, **kwargs)


def test_manual_progress_is_not_iterable() -> None:
    tracker = mdr.progress(total=1, disable=True)
    with pytest.raises(TypeError, match="not iterable"):
        next(iter(tracker))
    tracker.close()
