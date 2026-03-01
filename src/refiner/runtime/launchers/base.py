from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4
import re
import sys
import time

from refiner.platform import CredentialsError, ObserverClient, current_api_key
from refiner.platform.observer_client import ObserverJobContext

if TYPE_CHECKING:
    from refiner.ledger import BaseLedger
    from refiner.ledger.shard import Shard
    from refiner.pipeline import RefinerPipeline


class BaseLauncher(ABC):
    def __init__(
        self, *, pipeline: RefinerPipeline, name: str, run_id: str | None = None
    ):
        if not name.strip():
            raise ValueError("name must be non-empty")
        self.pipeline = pipeline
        self.name = name
        self.run_id = run_id or self._build_run_id(name)
        self.ledger: BaseLedger | None = None

    @staticmethod
    def _build_run_id(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "job"
        return f"{slug}-{int(time.time())}-{uuid4().hex[:8]}"

    def _warn(self, message: str) -> None:
        print(f"[refiner] {message}", file=sys.stderr)

    def _observer_client_or_none(self) -> ObserverClient | None:
        try:
            api_key = current_api_key()
        except CredentialsError:
            self._warn(
                "observability disabled: no Macrodata API key found. "
                "Run `macrodata login` or set MACRODATA_API_KEY to enable it."
            )
            return None
        return ObserverClient(api_key=api_key)

    def _setup_observer(
        self, *, shards: list["Shard"], fail_open: bool = True
    ) -> "_ObserverLaunchContext | None":
        client = self._observer_client_or_none()
        if client is None:
            return None
        try:
            job = client.submit_job(name=self.name, pipeline=self.pipeline)
            client.register_stage_shards(
                job_id=job.job_id,
                stage_index=job.stage_index,
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
            observer_ctx.client.finish_stage(
                job_id=observer_ctx.job.job_id,
                stage_index=observer_ctx.job.stage_index,
                status=status,
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"observability finish_stage failed: {type(e).__name__}: {e}")
        try:
            observer_ctx.client.finish_job(
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
    client: ObserverClient
    job: ObserverJobContext
