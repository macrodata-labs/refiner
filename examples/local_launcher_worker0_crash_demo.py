from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator

import refiner as mdr
from refiner.readers import DictRow


def _parse_worker_rank_from_argv() -> int:
    args = sys.argv
    try:
        rank_idx = args.index("--rank")
    except ValueError:
        return -1
    if rank_idx + 1 >= len(args):
        return -1
    try:
        return int(args[rank_idx + 1])
    except ValueError:
        return -1


_WORKER_RANK_BY_PID: dict[int, int] = {}
_SEEN_SHARDS: set[str] = set()


def _current_worker_rank() -> int:
    # Resolve rank at call-time in each subprocess; avoid parent-process value capture.
    pid = os.getpid()
    cached = _WORKER_RANK_BY_PID.get(pid)
    if cached is not None:
        return cached
    rank = _parse_worker_rank_from_argv()
    _WORKER_RANK_BY_PID[pid] = rank
    return rank


class CrashDemoReader(mdr.BaseReader):
    """In-memory reader with multiple tiny shards so workers claim repeatedly."""

    def __init__(self, *, num_shards: int = 12):
        self._shards = [
            mdr.Shard(path=f"demo-shard-{i}", start=0, end=1) for i in range(num_shards)
        ]

    def list_shards(self) -> list[mdr.Shard]:
        return list(self._shards)

    def read_shard(self, shard: mdr.Shard) -> Iterator[mdr.Row]:
        # Keep worker 1 slower so worker 0 is likely to claim multiple shards.
        worker_rank = _current_worker_rank()
        if worker_rank == 1:
            time.sleep(0.15)
        else:
            time.sleep(0.02)
        yield DictRow({"value": 1})


def crash_worker_zero_after_first_shard(row: mdr.Row) -> mdr.Row:
    shard_id = str(row["shard_id"])
    worker_rank = _current_worker_rank()

    # First row from second distinct shard on worker 0 => kill process.
    if worker_rank == 0 and shard_id not in _SEEN_SHARDS and len(_SEEN_SHARDS) >= 1:
        print(
            f"[demo] worker rank=0 crashing after first successful shard; next shard={shard_id}",
            flush=True,
        )
        os._exit(86)

    _SEEN_SHARDS.add(shard_id)
    return row.update(value=int(row["value"]) + 1)


def main() -> None:
    pipeline = mdr.RefinerPipeline(CrashDemoReader()).map(
        crash_worker_zero_after_first_shard
    )

    print("Launching local Refiner job with 2 workers...")
    print("Expected behavior: worker rank=0 exits after finishing its first shard.")

    t0 = time.time()
    try:
        stats = pipeline.launch_local(
            name="local-launcher-worker0-crash-demo",
            num_workers=2,
        )
    except RuntimeError as e:
        dt = time.time() - t0
        print("\nLaunch failed as expected")
        print(f"error       : {e}")
        print(f"elapsed_s   : {dt:.2f}")
        return

    dt = time.time() - t0
    print("\nLaunch unexpectedly completed")
    print(f"run_id      : {stats.run_id}")
    print(f"workers     : {stats.workers}")
    print(f"claimed     : {stats.claimed}")
    print(f"completed   : {stats.completed}")
    print(f"failed      : {stats.failed}")
    print(f"output_rows : {stats.output_rows}")
    print(f"elapsed_s   : {dt:.2f}")


if __name__ == "__main__":
    main()
