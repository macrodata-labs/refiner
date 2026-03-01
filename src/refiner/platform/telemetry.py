"""Worker telemetry emitters for platform observability."""

from __future__ import annotations

import logging
from typing import Protocol

from .metric_helpers import (
    get_cpu_usage,
    get_memory_usage,
    get_network_in,
    get_network_out,
)


class WorkerTelemetry(Protocol):
    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None: ...

    def force_flush_metrics(self) -> None: ...


class OtelTelemetryEmitter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        job_id: str,
        stage_index: int,
        worker_id: str,
    ):
        # Lazy imports keep non-observability flows working when OTel deps are absent.
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs import LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.metrics import CallbackOptions, Observation
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
        from opentelemetry.sdk.resources import Resource

        headers = {"Authorization": f"Bearer {api_key}"}
        resource = Resource.create(
            {
                "service.name": "refiner-worker",
                "job.id": job_id,
                "stage.index": stage_index,
                "worker.id": worker_id,
            }
        )

        metric_exporter = OTLPMetricExporter(
            endpoint=f"{base_url}/api/metrics",
            headers=headers,
        )
        # metric_exporter = ConsoleMetricExporter()
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=2_000,
        )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )

        meter = self._meter_provider.get_meter("refiner.platform.telemetry")
        meter.create_observable_gauge(
            "refiner.worker.cpu_usage",
            callbacks=[get_cpu_usage()],
            unit="%",
        )
        meter.create_observable_gauge(
            "refiner.worker.memory_usage",
            callbacks=[get_memory_usage],
            unit="MB",
        )
        meter.create_observable_gauge(
            "refiner.worker.network_in",
            callbacks=[get_network_in],
            unit="B",
        )
        meter.create_observable_gauge(
            "refiner.worker.network_out",
            callbacks=[get_network_out],
            unit="B",
        )
        self._user_counter = meter.create_counter(
            "refiner.user.counter",
            unit="records",
        )
        self._user_histogram = meter.create_histogram(
            "refiner.user.histogram",
            explicit_bucket_boundaries_advisory=[],
        )
        self._user_gauge = meter.create_gauge("refiner.user.gauge")

        log_exporter = OTLPLogExporter(
            endpoint=f"{base_url}/api/logs",
            headers=headers,
        )
        self._logger_provider = LoggerProvider(resource=resource)
        self._logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(log_exporter)
        )
        py_logger = logging.getLogger("refiner.observer")
        py_logger.setLevel(logging.INFO)
        py_logger.propagate = False
        py_logger.addHandler(LoggingHandler(logger_provider=self._logger_provider))

    @staticmethod
    def _attrs_with_step(
        *, attrs: dict[str, str | int], step_index: int | None
    ) -> dict[str, str | int]:
        if step_index is not None:
            attrs["step.index"] = int(step_index)
        return attrs

    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        attrs: dict[str, str | int] = {"label": label, "shard_id": shard_id}
        if unit:
            attrs["unit"] = unit
        self._user_counter.add(
            value,
            attributes=self._attrs_with_step(
                attrs=attrs,
                step_index=step_index,
            ),
        )

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        attrs_base: dict[str, str | int] = {"label": label, "shard_id": shard_id}
        if unit:
            attrs_base["unit"] = unit
        attrs = self._attrs_with_step(attrs=attrs_base, step_index=step_index)

        self._user_gauge.set(value, attributes=attrs)

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        attrs: dict[str, str | int] = {"label": label, "shard_id": shard_id}
        if unit:
            attrs["unit"] = unit
        self._user_histogram.record(
            value,
            attributes=self._attrs_with_step(
                attrs=attrs,
                step_index=step_index,
            ),
        )

    def force_flush_metrics(self) -> None:
        self._meter_provider.force_flush()


class NoopTelemetryEmitter:
    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, shard_id, step_index, unit

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, shard_id, step_index, unit

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, shard_id, step_index, unit

    def force_flush_metrics(self) -> None:
        return None


__all__ = ["WorkerTelemetry", "OtelTelemetryEmitter", "NoopTelemetryEmitter"]
