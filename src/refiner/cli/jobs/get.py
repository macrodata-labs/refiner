from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.job_utils import format_ts as _format_ts
from refiner.cli.job_utils import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _executor_text,
    _print_table,
    _progress_text,
    _run_job_command,
    _step_summary_text,
)


def _render_job(payload: dict[str, Any]) -> int:
    job = payload.get("job")
    if not isinstance(job, dict):
        print("Job details unavailable.", file=sys.stderr)
        return 1

    print(f"Job: {_safe_text(job.get('name'))} ({_safe_text(job.get('id'))})")
    print(
        "Status:"
        f" {_safe_text(job.get('status'))}"
        f"  Kind: {_executor_text(job.get('executorKind'))}"
        f"  Progress: {_progress_text(job.get('progress'))}"
    )
    print(
        "Created:"
        f" {_format_ts(job.get('createdAt'))}"
        f"  Started: {_format_ts(job.get('startedAt'))}"
        f"  Ended: {_format_ts(job.get('endedAt'))}"
    )
    started_by_email = job.get("startedByEmail")
    started_by_username = job.get("startedByUsername")
    if isinstance(started_by_email, str) and started_by_email:
        if isinstance(started_by_username, str) and started_by_username:
            print(
                f"Started By: {_safe_text(f'{started_by_username} ({started_by_email})')}"
            )
        else:
            print(f"Started By: {_safe_text(started_by_email)}")
    elif isinstance(started_by_username, str) and started_by_username:
        print(f"Started By: {_safe_text(started_by_username)}")
    print(
        "Workers:"
        f" {_safe_text(job.get('runningWorkers'))}/{_safe_text(job.get('totalWorkers'))}"
        f"  Cost: {_safe_text(job.get('currentCostUsd'))}"
    )
    print(
        "Manifest:"
        f" {_safe_text(job.get('manifestAvailable'))}"
        f"  Logs: {_safe_text(job.get('logsAvailable'))}"
        f"  Metrics: {_safe_text(job.get('metricsAvailable'))}"
    )
    if isinstance(job.get("error"), str) and job["error"]:
        print(f"Error: {_safe_text(job.get('error'))}")

    stages = job.get("stages")
    if isinstance(stages, list) and stages:
        print("\nStages")
        rows = [["Idx", "Status", "Shards", "Workers", "Name"]]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            rows.append(
                [
                    _safe_text(stage.get("index")),
                    _safe_text(stage.get("status")),
                    f"{_safe_text(stage.get('shardDone'))}/{_safe_text(stage.get('shardTotal'))}",
                    (
                        f"{_safe_text(stage.get('runningWorkers'))}"
                        f"/{_safe_text(stage.get('completedWorkers'))}"
                        f"/{_safe_text(stage.get('totalWorkers'))}"
                    ),
                    _safe_text(stage.get("name")),
                ]
            )
        _print_table(rows)
        step_rows = [["Stage", "Step", "Type", "Name", "Summary"]]
        has_steps = False
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_steps = stage.get("steps")
            if not isinstance(stage_steps, list):
                continue
            for step in stage_steps:
                if not isinstance(step, dict):
                    continue
                has_steps = True
                step_rows.append(
                    [
                        _safe_text(stage.get("index")),
                        _safe_text(step.get("index")),
                        _safe_text(step.get("type")),
                        _safe_text(step.get("name")),
                        _step_summary_text(step.get("args")),
                    ]
                )
        if has_steps:
            print("\nSteps")
            _print_table(step_rows)
    return 0


def cmd_jobs_get(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job(job_id=args.job_id),
        renderer=_render_job,
    )
