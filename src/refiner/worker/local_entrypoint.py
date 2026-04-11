from __future__ import annotations

import argparse
import json

import cloudpickle

from refiner.pipeline.data.shard import Shard
from refiner.worker.context import RunHandle, logger
from refiner.worker.lifecycle import LocalRuntimeLifecycle
from refiner.worker.resources.cpu import parse_cpu_ids, set_cpu_affinity
from refiner.worker.runner import Worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Refiner local worker entrypoint")
    parser.add_argument("--pipeline-payload", type=str, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--worker-name", type=str, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--rundir", type=str, required=True)
    parser.add_argument("--cpu-ids", type=str, default="")
    args = parser.parse_args()

    payload = {
        "worker_id": args.worker_id,
        "claimed": 0,
        "completed": 0,
        "failed": 0,
        "output_rows": 0,
        "error": None,
    }
    try:
        cpu_ids = tuple(parse_cpu_ids(args.cpu_ids))
        if cpu_ids:
            set_cpu_affinity(list(cpu_ids))

        with open(args.pipeline_payload, "rb") as f:
            pipeline = cloudpickle.load(f)

        with open(
            f"{args.rundir}/stage-{args.stage_index}/assignments/worker-{args.worker_id}.json",
            encoding="utf-8",
        ) as handle:
            shard_payload = json.load(handle)
        if not isinstance(shard_payload, list):
            raise ValueError("local worker shards payload must be a list")

        run_handle = RunHandle(
            job_id=args.job_id,
            stage_index=args.stage_index,
            worker_name=args.worker_name,
            worker_id=args.worker_id,
            client=None,
        )
        runtime_lifecycle = LocalRuntimeLifecycle(
            run=run_handle,
            rundir=args.rundir,
            assigned_shards=[
                Shard.from_dict(item)
                for item in shard_payload
                if isinstance(item, dict)
            ],
        )
        stats = Worker(
            pipeline=pipeline,
            run_handle=run_handle,
            runtime_lifecycle=runtime_lifecycle,
        ).run()
        payload.update(
            {
                "claimed": stats.claimed,
                "completed": stats.completed,
                "failed": stats.failed,
                "output_rows": stats.output_rows,
            }
        )
    except Exception as e:
        message = str(e).strip() or type(e).__name__
        logger.exception("local worker entrypoint failed: {}", message)
        payload["failed"] = 1
        payload["error"] = message
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
