from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Protocol

from refiner.ledger import BaseLedger
from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class WorkerRunStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    output_rows: int = 0


class WorkerObserver(Protocol):
    def on_worker_start(self, *, rank: int) -> None: ...

    def on_shard_start(self, shard: Shard) -> None: ...

    def on_shard_finish(
        self, shard: Shard, *, status: str, error: str | None = None
    ) -> None: ...

    def on_worker_finish(self, *, status: str, error: str | None = None) -> None: ...


class Worker:
    def __init__(
        self,
        rank: int,
        ledger: BaseLedger,
        pipeline: RefinerPipeline,
        *,
        heartbeat_every_rows: int = 4096,
        observer: WorkerObserver | None = None,
    ):
        self.rank = rank
        self.ledger = ledger
        self.pipeline = pipeline
        self.heartbeat_every_rows = heartbeat_every_rows
        self.observer = observer

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
        failed_error: str | None = None
        observer = self.observer

        def _notify(fn, *args, **kwargs) -> None:
            if observer is None:
                return
            try:
                fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                print(
                    f"[refiner] observer hook failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

        if observer is not None:
            _notify(observer.on_worker_start, rank=self.rank)

        def _source_rows():
            nonlocal previous, claimed
            while True:
                shard = self.ledger.claim(previous=previous)
                if shard is None:
                    break
                claimed += 1
                inflight.append(shard)
                if observer is not None:
                    _notify(observer.on_shard_start, shard)
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
                    if observer is not None:
                        _notify(
                            observer.on_shard_finish,
                            shard,
                            status="completed",
                            error=None,
                        )
                    completed += 1
                inflight.clear()
                break
            except Exception as e:
                failed_error = str(e)
                for shard in inflight:
                    self.ledger.fail(shard, str(e))
                    if observer is not None:
                        _notify(
                            observer.on_shard_finish,
                            shard,
                            status="failed",
                            error=str(e),
                        )
                    failed += 1
                inflight.clear()
                previous = None
                break

        if observer is not None:
            _notify(
                observer.on_worker_finish,
                status="failed" if failed_error is not None else "completed",
                error=failed_error,
            )

        return WorkerRunStats(
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )


__all__ = ["Worker", "WorkerRunStats", "WorkerObserver"]
