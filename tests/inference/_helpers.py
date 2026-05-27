from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel, Field

from refiner.worker.metrics.emitter import UserMetricsEmitter

anthropic_provider = importlib.import_module("refiner.inference.providers.anthropic")
google_provider = importlib.import_module("refiner.inference.providers.google")
openai_provider = importlib.import_module("refiner.inference.providers.openai")
runtime_module = importlib.import_module("refiner.inference.internal.runtime")
transport_module = importlib.import_module("refiner.inference.internal.transport")


class _Caption(BaseModel):
    title: str
    objects: list[str]


class _ConstrainedCaption(BaseModel):
    title: str = Field(min_length=2, max_length=20)
    score: int = Field(ge=0, le=5)
    objects: list[str] = Field(min_length=1, max_length=3)


class _Segment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str


class _Segments(BaseModel):
    segments: list[_Segment]


class _MetricRecordingEmitter(UserMetricsEmitter):
    def __init__(self) -> None:
        self.counters: list[dict[str, Any]] = []
        self.gauges: list[dict[str, Any]] = []
        self.registered_gauges: list[dict[str, Any]] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        self.gauges.append(kwargs)

    def register_user_gauge(self, **kwargs) -> None:
        self.registered_gauges.append(kwargs)

    def emit_user_histogram(self, **kwargs) -> None:
        del kwargs

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
