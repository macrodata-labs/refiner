from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cloudpickle

from refiner.ledger import CloudLedger, FsLedger
from refiner.platform import CredentialsError, MacrodataClient
from refiner.runtime.cpu import set_cpu_affinity
from refiner.runtime.memory import set_memory_soft_limit_mb
from refiner.runtime.worker import Worker, WorkerLifecycleContext


def _parse_cpu_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def _write_stats(path: str, payload: dict[str, int | str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refiner runtime worker entrypoint")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--heartbeat-every-rows", type=int, required=True)
    parser.add_argument("--pipeline-payload", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--cpu-ids", type=str, default="")
    parser.add_argument("--mem-mb-per-worker", type=int, default=0)
    parser.add_argument("--stage-id", type=str, default="")
    parser.add_argument("--worker-id", type=str, default="")
    parser.add_argument(
        "--ledger-backend",
        type=str,
        choices=("fs", "cloud"),
        default=os.environ.get("REFINER_LEDGER_BACKEND", "fs"),
    )
    args = parser.parse_args()

    try:
        cpu_ids = _parse_cpu_ids(args.cpu_ids)
        if cpu_ids:
            set_cpu_affinity(cpu_ids)
        if args.mem_mb_per_worker > 0:
            set_memory_soft_limit_mb(args.mem_mb_per_worker)

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)

        if args.ledger_backend == "cloud":
            cloud_api_key = os.environ.get("REFINER_CLOUD_RUNTIME_TOKEN", "").strip()
            if not args.job_id or not args.stage_id or not cloud_api_key:
                raise ValueError(
                    "cloud ledger requires --job-id, --stage-id, and REFINER_CLOUD_RUNTIME_TOKEN"
                )
            ledger = CloudLedger(
                job_id=args.job_id,
                worker_id=args.rank,
                stage_id=args.stage_id,
                api_key=cloud_api_key,
            )
        else:
            ledger = FsLedger(
                job_id=args.job_id, worker_id=args.rank, workdir=args.workdir
            )
        lifecycle_client = None
        lifecycle_context = None
        if args.job_id and args.stage_id and args.worker_id:
            try:
                lifecycle_client = MacrodataClient()
                lifecycle_context = WorkerLifecycleContext(
                    job_id=args.job_id,
                    stage_id=args.stage_id,
                    worker_id=args.worker_id,
                )
            except CredentialsError:
                lifecycle_client = None
                lifecycle_context = None
        stats = Worker(
            rank=args.rank,
            ledger=ledger,
            pipeline=pipeline,
            heartbeat_every_rows=args.heartbeat_every_rows,
            lifecycle_client=lifecycle_client,
            lifecycle_context=lifecycle_context,
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
