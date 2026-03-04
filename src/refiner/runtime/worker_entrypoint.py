from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import cloudpickle
from loguru import logger

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
    parser.add_argument("--worker-name", type=str, default="")
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
            if not args.job_id or not args.stage_id:
                raise ValueError("cloud ledger requires --job-id and --stage-id")
            try:
                ledger = CloudLedger(
                    job_id=args.job_id,
                    worker_id=args.rank,
                    stage_id=args.stage_id,
                )
            except CredentialsError as e:
                raise ValueError(
                    "cloud ledger requires Macrodata authentication. "
                    "Run `macrodata login` or set MACRODATA_API_KEY."
                ) from e
        else:
            ledger = FsLedger(
                job_id=args.job_id, worker_id=args.rank, workdir=args.workdir
            )
        lifecycle_client = None
        worker_name = args.worker_name or f"worker-{args.rank}"
        lifecycle_context = WorkerLifecycleContext(
            job_id=args.job_id,
            stage_id=args.stage_id,
            worker_id="",
            worker_name=worker_name,
        )
        if args.job_id and args.stage_id:
            try:
                lifecycle_client = MacrodataClient()
                try:
                    host = socket.gethostname()
                except Exception:
                    host = None
                started_resp = lifecycle_client.report_worker_started(
                    job_id=args.job_id,
                    stage_id=args.stage_id,
                    host=host,
                    worker_name=worker_name,
                )
                reported_worker_id = started_resp.get("worker_id")
                if not isinstance(reported_worker_id, str) or not reported_worker_id:
                    raise RuntimeError("workers/start response missing worker_id")
                lifecycle_context = WorkerLifecycleContext(
                    job_id=args.job_id,
                    stage_id=args.stage_id,
                    worker_id=reported_worker_id,
                    worker_name=worker_name,
                )
            except CredentialsError:
                lifecycle_client = None
            except Exception as e:
                logger.warning(
                    "lifecycle worker start failed: {}: {}",
                    type(e).__name__,
                    e,
                )
                lifecycle_client = None
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
