from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cloudpickle
from loguru import logger

from refiner.platform.client.api import MacrodataClient
from refiner.run import RunHandle
from refiner.worker.resources.cpu import set_cpu_affinity
from refiner.worker.resources.memory import set_memory_soft_limit_mb
from refiner.worker.runner import Worker


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
    parser.add_argument("--heartbeat-interval-seconds", type=int, required=True)
    parser.add_argument("--pipeline-payload", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--cpu-ids", type=str, default="")
    parser.add_argument("--mem-mb-per-worker", type=int, default=0)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--worker-name", type=str, default="")
    parser.add_argument(
        "--runtime-backend",
        type=str,
        choices=("auto", "platform", "file"),
        default=os.environ.get("REFINER_RUNTIME_BACKEND", "auto"),
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

        worker_name = args.worker_name or f"worker-{args.rank}"
        run_handle = RunHandle(
            job_id=args.job_id,
            stage_index=args.stage_index,
            worker_name=worker_name,
        )

        if args.runtime_backend != "file":
            try:
                client = MacrodataClient()
                run_handle = RunHandle(
                    job_id=args.job_id,
                    stage_index=args.stage_index,
                    client=client,
                    worker_name=worker_name,
                )
            except Exception as e:
                if args.runtime_backend == "platform":
                    raise
                logger.warning(
                    "platform runtime unavailable (falling back to file runtime): {}: {}",
                    type(e).__name__,
                    e,
                )

        stats = Worker(
            rank=args.rank,
            pipeline=pipeline,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            run_handle=run_handle,
            local_workdir=args.workdir,
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
