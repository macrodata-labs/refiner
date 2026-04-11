from __future__ import annotations

import argparse
import json
import os
import warnings

import cloudpickle
from loguru import logger

from refiner.platform.client.http import MacrodataApiError
from refiner.platform.client.api import MacrodataClient
from refiner.worker.context import RunHandle
from refiner.worker.resources.cpu import parse_cpu_ids, set_cpu_affinity
from refiner.worker.resources.gpu import parse_gpu_ids, set_visible_gpu_ids
from refiner.worker.runner import Worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Refiner runtime worker entrypoint")
    parser.add_argument("--pipeline-payload", type=str, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument(
        "--runtime-backend",
        type=str,
        choices=("auto", "platform", "file"),
        default=os.environ.get("REFINER_RUNTIME_BACKEND", "auto"),
    )
    parser.add_argument("--worker-name", type=str, default="worker")
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=30)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--cpu-ids", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="")
    args = parser.parse_args()

    try:
        cpu_ids = parse_cpu_ids(args.cpu_ids)
        if cpu_ids:
            set_cpu_affinity(cpu_ids)
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if gpu_ids:
            set_visible_gpu_ids(gpu_ids)
        elif args.gpu_ids.strip():
            warnings.warn(
                "gpu-ids argument did not resolve to any visible devices",
                RuntimeWarning,
                stacklevel=2,
            )

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)

        run_handle = RunHandle(
            job_id=args.job_id,
            stage_index=args.stage_index,
            worker_name=args.worker_name,
        )

        if args.runtime_backend != "file":
            try:
                client = MacrodataClient()
                run_handle = RunHandle(
                    job_id=args.job_id,
                    stage_index=args.stage_index,
                    worker_name=args.worker_name,
                    client=client,
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
            pipeline=pipeline,
            run_handle=run_handle,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            local_workdir=args.workdir,
        ).run()
        print(
            json.dumps(
                {
                    "claimed": stats.claimed,
                    "completed": stats.completed,
                    "failed": stats.failed,
                    "output_rows": stats.output_rows,
                },
                sort_keys=True,
            )
        )
        return 0
    except MacrodataApiError as e:
        message = str(e).strip() or type(e).__name__
        if e.status == 409:
            logger.info("worker entrypoint exiting cleanly: {}", message)
            print(json.dumps({"skipped": message}, sort_keys=True))
            return 0
        logger.exception("worker entrypoint failed: {}", message)
        print(json.dumps({"error": message}, sort_keys=True))
        return 1
    except Exception as e:
        message = str(e).strip() or type(e).__name__
        logger.exception("worker entrypoint failed: {}", message)
        print(json.dumps({"error": message}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
