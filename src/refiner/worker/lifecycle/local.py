from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from refiner.pipeline.data.shard import Shard
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.context import RunHandle


class LocalRuntimeLifecycle:
    def __init__(
        self,
        *,
        run: RunHandle,
        rundir: str,
        assigned_shards: Iterable[Shard],
    ) -> None:
        self.run = run
        self.rundir = rundir
        self._assigned_shards = iter(assigned_shards)

    def claim(self, previous: Shard | None = None) -> Shard | None:
        return next(self._assigned_shards, None)

    def heartbeat(self, shards: list[Shard]) -> None:
        pass

    def complete(self, shard: Shard) -> None:
        path = (
            Path(self.rundir)
            / f"stage-{self.run.stage_index}"
            / "completed"
            / f"{self.run.worker_id}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"shard_id": shard.id}, sort_keys=True))
            handle.write("\n")
        return None

    def fail(self, shard: Shard, error: str | None = None) -> None:
        pass

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        target_stage = self.run.stage_index if stage_index is None else stage_index
        directory = Path(self.rundir) / f"stage-{target_stage}" / "completed"
        if not directory.exists():
            return []
        out: list[FinalizedShardWorker] = []
        for path in sorted(directory.glob("*.jsonl")):
            worker_id = path.stem
            try:
                lines = path.read_text().splitlines()
            except Exception:
                continue
            for line in lines:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                shard_id = (
                    payload.get("shard_id") if isinstance(payload, dict) else None
                )
                if isinstance(shard_id, str):
                    out.append(
                        FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)
                    )
        out.sort(key=lambda row: row.shard_id)
        return out


__all__ = [
    "LocalRuntimeLifecycle",
]
