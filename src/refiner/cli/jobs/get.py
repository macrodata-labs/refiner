from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _dim_text,
    _error_text,
    _executor_text,
    _kind_text,
    _label_text,
    _print_table,
    _run_job_command,
    _section_text,
    _status_text,
    _started_by_text,
    _timestamp_text,
    _value_text,
)
from refiner.cli.ui.terminal import stdout_is_interactive
from refiner.job_urls import build_job_tracking_url

_COST_COLOR = "\x1b[1;38;5;255m"
_ANSI_RESET = "\x1b[0m"
_URL_COLOR = "\x1b[4;38;5;117m"


def _step_summary_text(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return "-"
    parts: list[str] = []
    for key in sorted(args.keys())[:3]:
        value = args.get(key)
        if key == "__meta":
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
            if key == "fn" and ("\n" in text or len(text) > 48):
                parts.append("fn=<code>")
            elif len(text) > 48:
                parts.append(f"{key}=...")
            else:
                parts.append(f"{key}={text}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        elif isinstance(value, dict):
            parts.append(f"{key}={{...}}")
    return _safe_text(", ".join(parts[:3]) if parts else "{...}")


def _availability_text(job: dict[str, Any]) -> str:
    available: list[str] = []
    if job.get("manifestAvailable") is True:
        available.append("manifest")
    if job.get("logsAvailable") is True:
        available.append("logs")
    if job.get("metricsAvailable") is True:
        available.append("metrics")
    return ", ".join(available) if available else "-"


def _cost_text(value: Any) -> str:
    text = _safe_text(value)
    if text in {"", "-"}:
        return "-"
    try:
        float(text)
    except (TypeError, ValueError):
        return text
    if text.startswith("$"):
        amount = text
    else:
        amount = f"${text}"
    if not stdout_is_interactive():
        return amount
    return f"{_COST_COLOR}{amount}{_ANSI_RESET}"


def _url_text(value: str) -> str:
    if not stdout_is_interactive() or value == "-":
        return value
    return f"{_URL_COLOR}{value}{_ANSI_RESET}"


def _tracking_url(job: dict[str, Any]) -> str:
    client = _client()
    if not hasattr(client, "base_url"):
        return "-"
    workspace_slug = job.get("workspaceSlug")
    built = build_job_tracking_url(
        client=client,
        job_id=_safe_text(job.get("id")),
        workspace_slug=workspace_slug if isinstance(workspace_slug, str) else None,
    )
    return _safe_text(built) if built else "-"


def _stage_runtime_value(runtime_config: Any, key: str) -> str:
    if not isinstance(runtime_config, dict):
        return "-"
    return _safe_text(runtime_config.get(key))


def _stage_gpu_text(runtime_config: Any) -> str:
    if not isinstance(runtime_config, dict):
        return "-"
    count = _safe_text(runtime_config.get("gpuCount"))
    gpu_type = _safe_text(runtime_config.get("gpuType"))
    if count in {"", "-"}:
        return "-"
    if gpu_type in {"", "-", "None"}:
        return count
    return f"{count} {gpu_type}"


def _render_job(payload: dict[str, Any]) -> int:
    job = payload.get("job")
    if not isinstance(job, dict):
        print("Job details unavailable.", file=sys.stderr)
        return 1

    tracking_url = _tracking_url(job)
    print(
        f"{_label_text('Job')}: {_value_text(job.get('name'))}"
        f"  {_label_text('ID')}: {_value_text(job.get('id'))}"
        f"  {_label_text('URL')}: {_url_text(tracking_url)}"
    )
    print(
        f"{_label_text('Status')}:"
        f" {_status_text(job.get('status'))}"
        f"  {_label_text('Kind')}: {_kind_text(_executor_text(job.get('executorKind')))}"
        f"  {_label_text('Cost')}: {_cost_text(job.get('currentCostUsd'))}"
    )
    print(
        f"{_label_text('Created')}:"
        f" {_timestamp_text(_format_ts(job.get('createdAt')))}"
        f"  {_label_text('Started')}: {_timestamp_text(_format_ts(job.get('startedAt')))}"
        f"  {_label_text('Ended')}: {_timestamp_text(_format_ts(job.get('endedAt')))}"
    )
    rundir = _safe_text(job.get("rundir"))
    if _executor_text(job.get("executorKind")) == "local" and rundir != "-":
        print(f"{_label_text('Rundir')}: {_value_text(rundir)}")
    started_by = _started_by_text(job)
    if started_by != "-":
        print(f"{_label_text('Created By')}: {_value_text(started_by)}")
    print(f"{_label_text('Available')}: {_availability_text(job)}")
    if isinstance(job.get("error"), str) and job["error"]:
        print(f"{_label_text('Error')}: {_error_text(job.get('error'))}")

    stages = job.get("stages")
    if isinstance(stages, list) and stages:
        print(f"\n{_section_text('Stages')}")
        rows = [
            [
                _dim_text("Idx"),
                _dim_text("Name"),
                _dim_text("Status"),
                _dim_text("Shards"),
                _dim_text("Workers"),
                _dim_text("Req"),
                _dim_text("CPU"),
                _dim_text("Memory"),
                _dim_text("GPU"),
            ]
        ]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            runtime_config = stage.get("runtimeConfig")
            rows.append(
                [
                    _safe_text(stage.get("index")),
                    _safe_text(stage.get("name")),
                    _status_text(stage.get("status")),
                    f"{_safe_text(stage.get('shardDone'))}/{_safe_text(stage.get('shardTotal'))}",
                    (
                        f"run={_safe_text(stage.get('runningWorkers'))} "
                        f"done={_safe_text(stage.get('completedWorkers'))} "
                        f"tot={_safe_text(stage.get('totalWorkers'))}"
                    ),
                    _stage_runtime_value(runtime_config, "requestedNumWorkers"),
                    _stage_runtime_value(runtime_config, "cpuCores"),
                    _stage_runtime_value(runtime_config, "memoryMb"),
                    _stage_gpu_text(runtime_config),
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
            print(f"\n{_section_text('Steps')}")
            step_rows[0] = [_dim_text(cell) for cell in step_rows[0]]
            _print_table(step_rows)
    return 0


def cmd_jobs_get(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job(job_id=args.job_id),
        renderer=_render_job,
    )
