from __future__ import annotations

from collections.abc import Iterable

from refiner.platform.client import MacrodataClient

from ..config import load_ledger_config_from_env
from ..shard import Shard
from .base import BaseLedger, LedgerConfig


class CloudLedger(BaseLedger):
    """Cloud-backed shard ledger over cloud-controller endpoints."""

    def __init__(
        self,
        *,
        run_id: str,
        worker_id: int | None,
        job_id: str,
        stage_id: str,
        api_key: str,
        config: LedgerConfig | None = None,
    ) -> None:
        cfg = config or load_ledger_config_from_env()
        super().__init__(run_id=run_id, worker_id=worker_id, config=cfg)
        if not job_id:
            raise ValueError("job_id must be non-empty")
        if not stage_id:
            raise ValueError("stage_id must be non-empty")
        self.job_id = job_id
        self.stage_id = stage_id
        self.client = MacrodataClient(api_key=api_key)

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        self.client.cloud_ledger_register_stage_shards(
            job_id=self.job_id,
            stage_id=self.stage_id,
            shards=list(shards),
        )

    def claim(self, previous: Shard | None = None) -> Shard | None:
        payload = self.client.cloud_ledger_claim_shard(
            job_id=self.job_id,
            stage_id=self.stage_id,
            worker_id=self._require_worker_id_str(),
            previous_shard_id=previous.id if previous is not None else None,
        )
        shard_payload = payload.get("shard")
        if shard_payload is None:
            return None
        if not isinstance(shard_payload, dict):
            raise ValueError("cloud ledger returned invalid shard payload")
        path = shard_payload.get("path")
        start = shard_payload.get("start")
        end = shard_payload.get("end")
        if not isinstance(path, str) or not path:
            raise ValueError("cloud ledger shard.path must be non-empty string")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("cloud ledger shard.start/end must be int")
        return Shard(path=path, start=start, end=end)

    def heartbeat(self, shard: Shard) -> None:
        self.client.cloud_ledger_heartbeat_shard(
            job_id=self.job_id,
            stage_id=self.stage_id,
            worker_id=self._require_worker_id_str(),
            shard_id=shard.id,
        )

    def complete(self, shard: Shard) -> None:
        self.client.cloud_ledger_complete_shard(
            job_id=self.job_id,
            stage_id=self.stage_id,
            worker_id=self._require_worker_id_str(),
            shard_id=shard.id,
        )

    def fail(self, shard: Shard, error: str | None = None) -> None:
        self.client.cloud_ledger_fail_shard(
            job_id=self.job_id,
            stage_id=self.stage_id,
            worker_id=self._require_worker_id_str(),
            shard_id=shard.id,
            error=error,
        )


__all__ = ["CloudLedger"]
