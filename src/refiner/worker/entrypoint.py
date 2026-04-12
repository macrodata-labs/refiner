from __future__ import annotations

import argparse
import json
import os
import socket

import cloudpickle

from refiner.platform.client.api import MacrodataApiError, MacrodataClient
from refiner.worker.context import RunHandle, logger
from refiner.worker.lifecycle import PlatformRuntimeLifecycle
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
        choices=("platform",),
        default=os.environ.get("REFINER_RUNTIME_BACKEND", "platform"),
    )
    parser.add_argument("--worker-name", type=str, default="worker")
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=30)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--cpu-ids", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="")
    args = parser.parse_args()

    try:
        # temp compatibility for the cloud
        if args.workdir:
            os.environ["REFINER_WORKDIR"] = args.workdir

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if gpu_ids:
            set_visible_gpu_ids(gpu_ids)

        client = MacrodataClient()
        try:
            host = socket.gethostname()
        except Exception:
            host = None
        started_resp = client.report_worker_started(
            job_id=args.job_id,
            stage_index=args.stage_index,
            host=host,
            worker_name=args.worker_name,
        )
        run_handle = RunHandle(
            job_id=args.job_id,
            stage_index=args.stage_index,
            worker_id=started_resp.worker_id,
            worker_name=args.worker_name,
            client=client,
        )
        runtime_lifecycle = PlatformRuntimeLifecycle(run=run_handle)

        stats = Worker(
            pipeline=pipeline,
            run_handle=run_handle,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            runtime_lifecycle=runtime_lifecycle,
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
