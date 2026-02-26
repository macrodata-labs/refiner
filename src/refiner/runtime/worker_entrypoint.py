from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle

from refiner.ledger import FsLedger
from refiner.platform import CredentialsError, MacrodataClient, current_api_key
from refiner.runtime.cpu import set_cpu_affinity
from refiner.runtime.observer import WorkerLifecycleObserver, WorkerObserverContext
from refiner.runtime.worker import Worker


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
    parser.add_argument("--job-id", type=str, default="")
    parser.add_argument("--stage-id", type=str, default="")
    parser.add_argument("--worker-id", type=str, default="")
    args = parser.parse_args()

    try:
        cpu_ids = _parse_cpu_ids(args.cpu_ids)
        if cpu_ids:
            set_cpu_affinity(cpu_ids)

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)

        ledger = FsLedger(run_id=args.run_id, worker_id=args.rank, workdir=args.workdir)
        observer = None
        if args.job_id and args.stage_id and args.worker_id:
            try:
                observer = WorkerLifecycleObserver(
                    client=MacrodataClient(api_key=current_api_key()),
                    context=WorkerObserverContext(
                        job_id=args.job_id,
                        stage_id=args.stage_id,
                        worker_id=args.worker_id,
                    ),
                )
            except CredentialsError:
                observer = None
        stats = Worker(
            rank=args.rank,
            ledger=ledger,
            pipeline=pipeline,
            heartbeat_every_rows=args.heartbeat_every_rows,
            observer=observer,
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
