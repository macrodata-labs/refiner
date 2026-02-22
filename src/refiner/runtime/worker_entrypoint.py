from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

import cloudpickle

from refiner.ledger import FsLedger
from refiner.runtime.worker import Worker


def _set_cpu_affinity(cpu_ids: list[int]) -> None:
    if not cpu_ids:
        return
    if not hasattr(os, "sched_setaffinity"):
        warnings.warn(
            "cpus_per_worker requested but os.sched_setaffinity is not available; running without CPU pinning",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    try:
        os.sched_setaffinity(0, set(cpu_ids))
    except Exception as e:
        warnings.warn(
            f"Failed to set CPU affinity ({e}); running without CPU pinning",
            RuntimeWarning,
            stacklevel=2,
        )


def _parse_cpu_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def _write_stats(path: str, payload: dict[str, int | str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refiner local worker entrypoint")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--heartbeat-every-rows", type=int, required=True)
    parser.add_argument("--pipeline-payload", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--cpu-ids", type=str, default="")
    args = parser.parse_args()

    try:
        cpu_ids = _parse_cpu_ids(args.cpu_ids)
        if cpu_ids:
            _set_cpu_affinity(cpu_ids)

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)

        ledger = FsLedger(run_id=args.run_id, worker_id=args.rank, workdir=args.workdir)
        stats = Worker(
            rank=args.rank,
            ledger=ledger,
            pipeline=pipeline,
            heartbeat_every_rows=args.heartbeat_every_rows,
        ).run()
        _write_stats(
            args.stats_path,
            {
                "claimed": stats.claimed,
                "completed": stats.completed,
                "failed": stats.failed,
                "output_rows": stats.output_rows,
            },
        )
        return 0
    except Exception as e:
        _write_stats(args.stats_path, {"error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
