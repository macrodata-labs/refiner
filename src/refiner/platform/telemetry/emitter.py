"""Worker telemetry emitters for platform observability."""

from __future__ import annotations

from typing import Any

from .metric_helpers import (
    get_cpu_usage_callback,
    get_memory_usage_callback,
    get_network_in,
    get_network_out,
)
from refiner.runtime.metrics_context import UserMetricsEmitter

_USER_METRIC_EXPORT_INTERVAL_MS = 10_000
_RESOURCE_METRIC_EXPORT_INTERVAL_MS = 10_000


class OtelTelemetryEmitter(UserMetricsEmitter):
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
        from opentelemetry._logs.severity import SeverityNumber
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
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
            endpoint=f"{base_url}/api/observability/metrics",
            headers=headers,
        )
        user_metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=_USER_METRIC_EXPORT_INTERVAL_MS,
        )
        self._user_meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[user_metric_reader],
        )

        resource_metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=f"{base_url}/api/observability/metrics",
                headers=headers,
            ),
            export_interval_millis=_RESOURCE_METRIC_EXPORT_INTERVAL_MS,
        )
        self._resource_meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[resource_metric_reader],
        )

        resource_meter = self._resource_meter_provider.get_meter(
            "refiner.platform.telemetry.resource"
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.cpu_usage",
            callbacks=[get_cpu_usage_callback()],
            unit="%",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.memory_usage",
            callbacks=[get_memory_usage_callback()],
            unit="MB",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.network_in",
            callbacks=[get_network_in],
            unit="B",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.network_out",
            callbacks=[get_network_out],
            unit="B",
        )

        user_meter = self._user_meter_provider.get_meter(
            "refiner.platform.telemetry.user"
        )
        self._user_counter = user_meter.create_counter(
            "refiner.user.counter",
            unit="records",
        )
        self._user_histogram = user_meter.create_histogram(
            "refiner.user.histogram",
            explicit_bucket_boundaries_advisory=[],
        )
        self._user_gauge = user_meter.create_gauge("refiner.user.gauge")

        log_exporter = OTLPLogExporter(
            endpoint=f"{base_url}/api/observability/logs",
            headers=headers,
        )
        self._logger_provider = LoggerProvider(resource=resource)
        self._logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(log_exporter)
        )
        self._otel_logger = self._logger_provider.get_logger("refiner.loguru")
        self._severity_by_level = {
            "TRACE": SeverityNumber.TRACE,
            "DEBUG": SeverityNumber.DEBUG,
            "INFO": SeverityNumber.INFO,
            "SUCCESS": SeverityNumber.INFO2,
            "WARNING": SeverityNumber.WARN,
            "ERROR": SeverityNumber.ERROR,
            "CRITICAL": SeverityNumber.FATAL,
        }

        self._loguru_logger: Any | None = None
        self._loguru_sink_id: int | None = None
        self._install_loguru_bridge()

    def _install_loguru_bridge(self) -> None:
        try:
            from loguru import logger as loguru_logger
        except Exception:
            return

        def _forward_loguru(message: Any) -> None:
            record = message.record
            level_name = str(record["level"].name).upper()
            severity_number = self._severity_by_level.get(level_name)
            text = str(record.get("message") or "")
            timestamp_ns = int(record["time"].timestamp() * 1_000_000_000)
            attrs: dict[str, Any] = {
                "logger.name": str(record.get("name") or ""),
                "code.filepath": str(record.get("file").path),
                "code.lineno": int(record.get("line") or 0),
                "code.function": str(record.get("function") or ""),
            }
            for key, value in dict(record.get("extra") or {}).items():
                attrs[f"loguru.extra.{key}"] = str(value)

            self._otel_logger.emit(
                timestamp=timestamp_ns,
                observed_timestamp=timestamp_ns,
                severity_number=severity_number,
                severity_text=level_name,
                body=text,
                attributes=attrs,
            )

        self._loguru_sink_id = loguru_logger.add(
            _forward_loguru,
            level="INFO",
            enqueue=False,
            catch=True,
        )
        self._loguru_logger = loguru_logger

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

    def force_flush_user_metrics(self) -> None:
        self._user_meter_provider.force_flush()

    def force_flush_resource_metrics(self) -> None:
        self._resource_meter_provider.force_flush()

    def force_flush_logs(self) -> None:
        self._logger_provider.force_flush()

    def shutdown(self) -> None:
        try:
            self.force_flush_logs()
        finally:
            self._logger_provider.shutdown()
            if self._loguru_logger is not None and self._loguru_sink_id is not None:
                try:
                    self._loguru_logger.remove(self._loguru_sink_id)
                except Exception:
                    pass
                self._loguru_sink_id = None


__all__ = ["OtelTelemetryEmitter"]
