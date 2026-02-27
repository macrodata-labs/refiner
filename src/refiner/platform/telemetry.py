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
    def record_records_processed(self, delta: int) -> None: ...


class OtelTelemetryEmitter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        job_id: str,
        stage_id: str,
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
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        headers = {"Authorization": f"Bearer {api_key}"}
        resource = Resource.create(
            {
                "service.name": "refiner-worker",
                "job.id": job_id,
                "stage.id": stage_id,
                "worker.id": worker_id,
            }
        )

        metric_exporter = OTLPMetricExporter(
            endpoint=f"{base_url}/api/metrics",
            headers=headers,
        )
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
        self._records_processed_counter = meter.create_counter(
            "refiner.worker.records_processed",
            unit="records",
        )

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

    def record_records_processed(self, delta: int) -> None:
        if delta > 0:
            self._records_processed_counter.add(delta)


class NoopTelemetryEmitter:
    def record_records_processed(self, delta: int) -> None:
        del delta


__all__ = ["WorkerTelemetry", "OtelTelemetryEmitter", "NoopTelemetryEmitter"]
