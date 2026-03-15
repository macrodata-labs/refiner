from __future__ import annotations

from collections.abc import Iterable

from refiner.pipeline.data.shard import Shard
from refiner.run import RunHandle
from refiner.platform.client.models import FinalizedShardWorker


class PlatformRuntimeLifecycle:
    def __init__(
        self,
        *,
        run: RunHandle,
    ) -> None:
        if not run.job_id:
            raise ValueError("job_id must be non-empty")
        if run.stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        self.run = run

    def _require_worker_id(self) -> str:
        if not self.run.worker_id:
            raise ValueError("worker_id is required for runtime shard operations")
        return self.run.worker_id

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        self.run.client.shard_register(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index,
            shards=list(shards),
        )

    def claim(self, previous: Shard | None = None) -> Shard | None:
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        claim = self.run.client.shard_claim(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index,
            worker_id=self._require_worker_id(),
            previous_shard_id=previous.id if previous is not None else None,
        )
        if claim.shard is None:
            return None
        return Shard.from_dict(
            {
                "descriptor": claim.shard.descriptor,
                "global_ordinal": claim.shard.global_ordinal,
                "start_key": claim.shard.start_key,
                "end_key": claim.shard.end_key,
            }
        )

    def heartbeat(self, shards: Iterable[Shard]) -> None:
        shard_ids = [shard.id for shard in shards]
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        self.run.client.shard_heartbeat(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index,
            worker_id=self._require_worker_id(),
            shard_ids=shard_ids,
        )

    def complete(self, shard: Shard) -> None:
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        self.run.client.shard_finish(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index,
            worker_id=self._require_worker_id(),
            shard_id=shard.id,
            status="completed",
        )

    def fail(self, shard: Shard, error: str | None = None) -> None:
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        self.run.client.shard_finish(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index,
            worker_id=self._require_worker_id(),
            shard_id=shard.id,
            status="failed",
            error=error,
        )

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        if self.run.client is None:
            raise ValueError("platform runtime requires a client")
        response = self.run.client.shard_finalized_workers(
            job_id=self.run.job_id,
            stage_index=self.run.stage_index if stage_index is None else stage_index,
        )
        return list(response.shards)
