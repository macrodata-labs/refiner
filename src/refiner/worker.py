from __future__ import annotations

from dataclasses import dataclass

from refiner.ledger import BaseLedger
from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class WorkerRunStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    output_rows: int = 0


class Worker:
    def __init__(
        self,
        rank: int,
        ledger: BaseLedger,
        pipeline: RefinerPipeline,
        *,
        heartbeat_every_rows: int = 4096,
    ):
        self.rank = rank
        self.ledger = ledger
        self.pipeline = pipeline
        self.heartbeat_every_rows = heartbeat_every_rows

    def run(self) -> WorkerRunStats:
        if self.heartbeat_every_rows <= 0:
            raise ValueError("heartbeat_every_rows must be > 0")

        previous: Shard | None = None
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        output_rows_since_hb = 0
        inflight: list[Shard] = []

        def _source_rows():
            nonlocal previous, claimed
            while True:
                shard = self.ledger.claim(previous=previous)
                if shard is None:
                    break
                claimed += 1
                inflight.append(shard)
                yield from self.pipeline.source.read_shard(shard)
                previous = shard

        while True:
            try:
                for _ in self.pipeline.execute_rows(_source_rows()):
                    output_rows += 1
                    output_rows_since_hb += 1
                    if output_rows_since_hb % self.heartbeat_every_rows == 0:
                        for shard in inflight:
                            self.ledger.heartbeat(shard)

                for shard in inflight:
                    self.ledger.heartbeat(shard)
                    self.ledger.complete(shard)
                    completed += 1
                inflight.clear()
                break
            except Exception as e:
                for shard in inflight:
                    self.ledger.fail(shard, str(e))
                    failed += 1
                inflight.clear()
                previous = None
                break

        return WorkerRunStats(
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )


__all__ = ["Worker", "WorkerRunStats"]
