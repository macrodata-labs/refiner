"""OpenTelemetry-backed worker telemetry emitters."""

from __future__ import annotations

from typing import Any

from .context import UserMetricsEmitter
from refiner.worker.resources.cpu import cpu_observer_callback
from refiner.worker.resources.memory import memory_observer_callback
from refiner.worker.resources.network import (
    network_in_observer_callback,
    network_out_observer_callback,
)

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
            "refiner.worker.resource"
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.cpu",
            callbacks=[cpu_observer_callback()],
            unit="%",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.memory",
            callbacks=[memory_observer_callback()],
            unit="MB",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.network_in",
            callbacks=[network_in_observer_callback],
            unit="B",
        )
        resource_meter.create_observable_gauge(
            "refiner.worker.network_out",
            callbacks=[network_out_observer_callback],
            unit="B",
        )

        user_meter = self._user_meter_provider.get_meter("refiner.worker.user")
        self._user_counter = user_meter.create_counter(
            "refiner.user.counter", unit="records"
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
        from loguru import logger as loguru_logger

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
            attributes=self._attrs_with_step(attrs=attrs, step_index=step_index),
        )

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        attrs: dict[str, str | int] = {"label": label}
        if kind:
            attrs["kind"] = kind
        if unit:
            attrs["unit"] = unit
        self._user_gauge.set(
            value,
            attributes=self._attrs_with_step(attrs=attrs, step_index=step_index),
        )

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        per: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        attrs: dict[str, str | int] = {
            "label": label,
            "shard_id": shard_id,
            "per": per,
        }
        if unit:
            attrs["unit"] = unit
        self._user_histogram.record(
            value,
            attributes=self._attrs_with_step(attrs=attrs, step_index=step_index),
        )

    def force_flush_user_metrics(self) -> None:
        self._user_meter_provider.force_flush()

    def force_flush_resource_metrics(self) -> None:
        self._resource_meter_provider.force_flush()

    def force_flush_logs(self) -> None:
        self._logger_provider.force_flush()

    def shutdown(self) -> None:
        try:
            self.force_flush_user_metrics()
            self.force_flush_resource_metrics()
            self.force_flush_logs()
        except Exception:
            pass
        finally:
            try:
                self._logger_provider.shutdown()
            except Exception:
                pass
            try:
                self._resource_meter_provider.shutdown()
            except Exception:
                pass
            try:
                self._user_meter_provider.shutdown()
            except Exception:
                pass
            if self._loguru_logger is not None and self._loguru_sink_id is not None:
                try:
                    self._loguru_logger.remove(self._loguru_sink_id)
                except Exception:
                    pass


__all__ = ["OtelTelemetryEmitter"]
