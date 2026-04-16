from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger as _base_logger


class UserMetricsEmitter:
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
        return None

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, kind, step_index, unit
        return None

    def register_user_gauge(
        self,
        *,
        label: str,
        callback: Callable[[], float | int],
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, callback, kind, step_index, unit
        return None

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
        del label, value, shard_id, per, step_index, unit
        return None

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


NOOP_USER_METRICS_EMITTER: UserMetricsEmitter = UserMetricsEmitter()


class LocalLogEmitter(UserMetricsEmitter):
    def __init__(self, *, rundir: str, stage_index: int, worker_id: str) -> None:
        log_path = (
            Path(rundir).expanduser().resolve()
            / f"stage-{stage_index}"
            / "logs"
            / f"worker-{worker_id}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = _base_logger
        self._sink_id = self._logger.add(
            str(log_path),
            enqueue=True,
            catch=True,
        )

    def force_flush_logs(self) -> None:
        self._logger.complete()

    def shutdown(self) -> None:
        try:
            self.force_flush_logs()
        except Exception:
            pass
        finally:
            try:
                self._logger.remove(self._sink_id)
            except Exception:
                pass


__all__ = [
    "LocalLogEmitter",
    "UserMetricsEmitter",
    "NOOP_USER_METRICS_EMITTER",
]
