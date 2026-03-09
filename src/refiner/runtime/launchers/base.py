from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4
import re
import time

from loguru import logger

from refiner.platform import CredentialsError, MacrodataClient
from refiner.platform.client import JobContext

if TYPE_CHECKING:
    from refiner.ledger import BaseLedger
    from refiner.ledger.shard import Shard
    from refiner.pipeline import RefinerPipeline


class BaseLauncher(ABC):
    def __init__(
        self,
        *,
        pipeline: RefinerPipeline,
        name: str,
        job_id: str | None = None,
        num_workers: int | None = None,
        heartbeat_every_rows: int | None = None,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
    ):
        if not name.strip():
            raise ValueError("name must be non-empty")
        self.pipeline = pipeline
        self.name = name
        self.job_id = job_id or self._build_local_job_id(name)
        self.ledger: BaseLedger | None = None
        self.cpus_per_worker: int | None = None
        self.mem_mb_per_worker: int | None = None
        if num_workers is not None:
            if num_workers <= 0:
                raise ValueError("num_workers must be > 0")
            self.num_workers = num_workers
        if heartbeat_every_rows is not None:
            if heartbeat_every_rows <= 0:
                raise ValueError("heartbeat_every_rows must be > 0")
            self.heartbeat_every_rows = heartbeat_every_rows
        if cpus_per_worker is not None:
            if cpus_per_worker <= 0:
                raise ValueError("cpus_per_worker must be > 0")
            self.cpus_per_worker = cpus_per_worker
        if mem_mb_per_worker is not None:
            if mem_mb_per_worker <= 0:
                raise ValueError("mem_mb_per_worker must be > 0")
            self.mem_mb_per_worker = mem_mb_per_worker

    @staticmethod
    def _build_local_job_id(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "job"
        return f"{slug}-{int(time.time())}-{uuid4().hex[:8]}"

    def _warn(self, message: str) -> None:
        logger.warning(message)

    def _info(self, message: str) -> None:
        logger.info(message)

    def _job_tracking_url(
        self, *, client: MacrodataClient, job_id: str, workspace_slug: str | None = None
    ) -> str:
        if workspace_slug:
            return f"{client.base_url}/jobs/{workspace_slug}/{job_id}"
        return f"{client.base_url}/jobs/{job_id}"

    def _observer_client_or_none(self) -> MacrodataClient | None:
        try:
            return MacrodataClient()
        except CredentialsError:
            self._warn(
                "observability disabled: no Macrodata API key found. "
                "Run `macrodata login` or set MACRODATA_API_KEY to enable it."
            )
            return None

    def _setup_observer(
        self, *, shards: list["Shard"], fail_open: bool = True
    ) -> "_ObserverLaunchContext | None":
        client = self._observer_client_or_none()
        if client is None:
            return None
        try:
            job = client.create_job(name=self.name, pipeline=self.pipeline)
            self.job_id = job.job_id
            client.register_stage_shards(
                job_id=job.job_id,
                stage_id=job.stage_id,
                shards=shards,
            )
            return _ObserverLaunchContext(client=client, job=job)
        except Exception as e:  # noqa: BLE001
            if fail_open:
                self._warn(
                    "observability setup failed (continuing without it): "
                    f"{type(e).__name__}: {e}"
                )
                return None
            raise

    def _finish_observer_terminal(
        self, observer_ctx: "_ObserverLaunchContext | None", *, status: str
    ) -> None:
        if observer_ctx is None:
            return
        try:
            observer_ctx.client.report_stage_finished(
                job_id=observer_ctx.job.job_id,
                stage_id=observer_ctx.job.stage_id,
                status=status,
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"observability finish_stage failed: {type(e).__name__}: {e}")
        try:
            observer_ctx.client.report_job_finished(
                job_id=observer_ctx.job.job_id, status=status
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"observability finish_job failed: {type(e).__name__}: {e}")

    def seed_ledger(self, *, shards: list["Shard"]) -> None:
        if self.ledger is None:
            raise ValueError("launcher.ledger is not configured")
        self.ledger.seed_shards(shards)

    @abstractmethod
    def launch(self):
        raise NotImplementedError


__all__ = ["BaseLauncher"]


@dataclass(frozen=True, slots=True)
class _ObserverLaunchContext:
    client: MacrodataClient
    job: JobContext
