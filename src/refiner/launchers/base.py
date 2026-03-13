from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4
import re
import time

from loguru import logger

from refiner.platform.auth import CredentialsError
from refiner.platform.client.api import MacrodataClient
from refiner.platform.client.http import sanitize_terminal_text
from refiner.platform.client.models import RunHandle
from refiner.platform.manifest import build_run_manifest
from refiner.pipeline.planning import compile_pipeline_plan

if TYPE_CHECKING:
    from refiner.pipeline.data.shard import Shard
    from refiner.pipeline import RefinerPipeline


class BaseLauncher(ABC):
    def __init__(
        self,
        *,
        pipeline: RefinerPipeline,
        name: str,
        job_id: str | None = None,
        num_workers: int | None = None,
        heartbeat_interval_seconds: int | None = None,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
    ):
        if not name.strip():
            raise ValueError("name must be non-empty")
        self.pipeline = pipeline
        self.name = name
        self.job_id = job_id or self._build_local_job_id(name)
        self.cpus_per_worker: int | None = None
        self.mem_mb_per_worker: int | None = None
        if num_workers is not None:
            if num_workers <= 0:
                raise ValueError("num_workers must be > 0")
            self.num_workers = num_workers
        if heartbeat_interval_seconds is not None:
            if heartbeat_interval_seconds <= 0:
                raise ValueError("heartbeat_interval_seconds must be > 0")
            self.heartbeat_interval_seconds = heartbeat_interval_seconds
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
        safe_base_url = sanitize_terminal_text(client.base_url).strip().rstrip("/")
        safe_job_id = sanitize_terminal_text(job_id).strip() or job_id
        safe_workspace_slug = (
            sanitize_terminal_text(workspace_slug).strip() if workspace_slug else None
        )
        if safe_workspace_slug:
            return f"{safe_base_url}/jobs/{safe_workspace_slug}/{safe_job_id}"
        return f"{safe_base_url}/jobs/{safe_job_id}"

    def _platform_client_or_none(self) -> MacrodataClient | None:
        try:
            return MacrodataClient()
        except CredentialsError:
            self._warn(
                "platform integration disabled: no API key found in "
                "MACRODATA_API_KEY or local credentials. "
                "Run `macrodata login` or set MACRODATA_API_KEY."
            )
            return None

    def _require_platform_client(self) -> MacrodataClient:
        client = self._platform_client_or_none()
        if client is None:
            raise RuntimeError(
                "platform runtime requires Macrodata authentication. "
                "Run `macrodata login` or set MACRODATA_API_KEY."
            )
        return client

    def _compiled_plan(self) -> dict[str, object]:
        return compile_pipeline_plan(self.pipeline)

    def _run_manifest(self) -> dict[str, object]:
        return build_run_manifest()

    def _setup_platform(
        self, *, shards: list["Shard"], fail_open: bool = True
    ) -> RunHandle | None:
        client = self._platform_client_or_none()
        if client is None:
            if not fail_open:
                raise RuntimeError(
                    "platform runtime requires Macrodata authentication. "
                    "Run `macrodata login` or set MACRODATA_API_KEY."
                )
            return None
        try:
            job = client.create_job(
                name=self.name,
                executor={"type": "refiner-local"},
                plan=self._compiled_plan(),
                manifest=self._run_manifest(),
            )
            self.job_id = job.job_id
            client.shard_register(
                job_id=job.job_id,
                stage_index=job.stage_index,
                shards=shards,
            )
            return job
        except Exception as e:  # noqa: BLE001
            if fail_open:
                self._warn(
                    "platform setup failed (continuing without it): "
                    f"{type(e).__name__}: {e}"
                )
                return None
            raise

    def _finish_platform_terminal(
        self, platform_run: RunHandle | None, *, status: str
    ) -> None:
        if platform_run is None or platform_run.client is None:
            return
        try:
            platform_run.client.report_stage_finished(
                job_id=platform_run.job_id,
                stage_index=platform_run.stage_index,
                status=status,
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"platform finish_stage failed: {type(e).__name__}: {e}")
        try:
            platform_run.client.report_job_finished(
                job_id=platform_run.job_id, status=status
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"platform finish_job failed: {type(e).__name__}: {e}")

    @abstractmethod
    def launch(self):
        raise NotImplementedError


__all__ = ["BaseLauncher"]
