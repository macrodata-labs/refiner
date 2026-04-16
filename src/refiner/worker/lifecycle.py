from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

import msgspec

from refiner.pipeline.data.shard import Shard
from refiner.worker.context import worker_token_for


class FinalizedShardWorker(msgspec.Struct, frozen=True):
    shard_id: str
    worker_id: str

    @property
    def worker_token(self) -> str:
        return worker_token_for(self.worker_id)


class RuntimeLifecycle(Protocol):
    def claim(self, previous: Shard | None = None) -> Shard | None: ...

    def heartbeat(self, shards: list[Shard]) -> None: ...

    def complete(self, shard: Shard) -> None: ...

    def fail(self, shard: Shard, error: str | None = None) -> None: ...

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]: ...


def read_finalized_workers(
    *, rundir: str, stage_index: int
) -> list[FinalizedShardWorker]:
    directory = Path(rundir) / f"stage-{stage_index}" / "completed"
    if not directory.exists():
        return []

    rows: list[FinalizedShardWorker] = []
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
            shard_id = payload.get("shard_id") if isinstance(payload, dict) else None
            if isinstance(shard_id, str):
                rows.append(
                    FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)
                )
    rows.sort(key=lambda row: row.shard_id)
    return rows


class LocalRuntimeLifecycle:
    def __init__(
        self,
        *,
        stage_index: int,
        worker_id: str,
        rundir: str,
        assigned_shards: Iterable[Shard],
    ) -> None:
        self.stage_index = stage_index
        self.worker_id = worker_id
        self.rundir = rundir
        self._assigned_shards = iter(assigned_shards)

    def claim(self, previous: Shard | None = None) -> Shard | None:
        return next(self._assigned_shards, None)

    def heartbeat(self, shards: list[Shard]) -> None:
        del shards

    def complete(self, shard: Shard) -> None:
        path = (
            Path(self.rundir)
            / f"stage-{self.stage_index}"
            / "completed"
            / f"{self.worker_id}.jsonl"
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
        target_stage = self.stage_index if stage_index is None else stage_index
        return read_finalized_workers(
            rundir=self.rundir,
            stage_index=target_stage,
        )


__all__ = [
    "FinalizedShardWorker",
    "LocalRuntimeLifecycle",
    "RuntimeLifecycle",
    "read_finalized_workers",
]
