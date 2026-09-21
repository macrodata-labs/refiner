from __future__ import annotations

from argparse import Namespace

from refiner.cli.common import create_client
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import _run_job_command


def _render_cancel(payload: dict[str, object]) -> int:
    job_id = payload.get("jobId", payload.get("job_id"))
    requested = payload.get("requestedOperations", payload.get("requested_operations"))
    canceled = payload.get("canceledOperations", payload.get("canceled_operations"))
    failed = payload.get("failedOperations", payload.get("failed_operations"))
    print(
        "Canceled:"
        f" {_safe_text(job_id)}"
        f"  Requested: {_safe_text(requested)}"
        f"  Canceled: {_safe_text(canceled)}"
        f"  Failed: {_safe_text(failed)}"
    )
    return 0


def cmd_jobs_cancel(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: create_client().cli_cancel_job(job_id=args.job_id),
        renderer=_render_cancel,
    )


def _render_scale(payload: dict[str, object]) -> int:
    job_id = payload.get("job_id", payload.get("jobId"))
    stage_index = payload.get("stage_index", payload.get("stageIndex"))
    previous = payload.get(
        "previous_desired_workers", payload.get("previousDesiredWorkers")
    )
    desired = payload.get("desired_workers", payload.get("desiredWorkers"))
    active = payload.get("active_workers", payload.get("activeWorkers"))
    starting = payload.get("starting_workers", payload.get("startingWorkers"))
    draining = payload.get("draining_workers", payload.get("drainingWorkers"))
    print(f"Job:       {_safe_text(job_id)}")
    print(f"Stage:     {_safe_text(stage_index)}")
    print(f"Workers:   {_safe_text(previous)} → {_safe_text(desired)} desired")
    print(f"Active:    {_safe_text(active)}")
    print(f"Starting:  {_safe_text(starting)}")
    print(f"Draining:  {_safe_text(draining)}")
    if isinstance(draining, int) and draining > 0:
        print("\nWorkers will exit after finishing their current shard.")
    return 0


def cmd_jobs_scale(args: Namespace) -> int:
    if args.workers < 1:
        print("Error: --workers must be a positive integer")
        return 2
    if args.stage is not None and args.stage < 0:
        print("Error: --stage must be non-negative")
        return 2
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: create_client().cli_scale_job_workers(
            job_id=args.job_id,
            workers=args.workers,
            stage_index=args.stage,
        ),
        renderer=_render_scale,
    )
