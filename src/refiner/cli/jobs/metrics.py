from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import _client, _print_table, _run_job_command

_MAX_METRICS_WORKER_IDS = 50


def _metric_details_text(metric: dict[str, Any]) -> str:
    kind = metric.get("metricKind")
    if kind == "counter":
        if "total" not in metric:
            return _safe_text(f"unit={metric.get('unit')}")
        return _safe_text(
            f"total={metric.get('total')} rate={metric.get('rateSinceStart')} per_worker={metric.get('perWorker')}"
        )
    if kind == "gauge":
        if "avgAllTime" not in metric:
            return _safe_text(
                f"kind={metric.get('kind') or '-'} unit={metric.get('unit')}"
            )
        return _safe_text(
            f"avg_all={metric.get('avgAllTime')} avg_5m={metric.get('avgLast5m')} max_5m={metric.get('maxLast5m')}"
        )
    if kind == "histogram":
        if "average" not in metric:
            return _safe_text(f"per={metric.get('per')} unit={metric.get('unit')}")
        return _safe_text(
            f"avg={metric.get('average')} total={metric.get('total')} min={metric.get('min')} max={metric.get('max')} count={metric.get('count')}"
        )
    return "-"


def _render_metrics(payload: dict[str, Any]) -> int:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        print("Metrics unavailable.", file=sys.stderr)
        return 1
    if not steps:
        print("No step metrics found.")
        return 0

    print(f"Job: {_safe_text(payload.get('jobId'))}")
    print(f"Stage: {_safe_text(payload.get('stageIndex'))}")
    detail_level = _safe_text(payload.get("detailLevel"))
    if detail_level != "-":
        print(f"Detail: {detail_level}")
    for step in steps:
        if not isinstance(step, dict):
            continue
        print(
            "\n"
            f"Step {_safe_text(step.get('stepIndex'))}: "
            f"{_safe_text(step.get('name'))} "
            f"({_safe_text(step.get('type'))})"
        )
        metrics = step.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            print("No metrics.")
            continue
        rows = [["Kind", "Label", "Details"]]
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            rows.append(
                [
                    _safe_text(metric.get("metricKind")),
                    _safe_text(metric.get("label")),
                    _metric_details_text(metric),
                ]
            )
        _print_table(rows)
    return 0


def _render_resource_metrics(payload: dict[str, Any]) -> int:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        print("Metrics unavailable.", file=sys.stderr)
        return 1

    resources = metrics.get("resources")
    if not isinstance(resources, list) or not resources:
        print("No resource metrics found.")
        return 0

    latest = resources[-1]
    if not isinstance(latest, dict):
        print("No resource metrics found.")
        return 0

    print(f"Job: {_safe_text(payload.get('jobId'))}")
    print(f"Stage: {_safe_text(payload.get('stageIndex'))}")
    print(f"Range: {_safe_text(payload.get('range'))}")
    print(f"Latest sample: {_format_ts(latest.get('t'))}")
    print(
        f"CPU: {_safe_text(latest.get('cpuUsage'))} / {_safe_text(latest.get('cpuQuota'))}"
        f"  Memory: {_safe_text(latest.get('memoryUsage'))} / {_safe_text(latest.get('memoryLimit'))}"
    )
    print(
        f"Network In: {_safe_text(latest.get('networkInMb'))} MB"
        f"  Network Out: {_safe_text(latest.get('networkOutMb'))} MB"
    )
    print(f"Samples: {len(resources)}")
    return 0


def cmd_jobs_metrics(args: Namespace) -> int:
    metric_labels = list(dict.fromkeys(args.metric))
    if metric_labels and args.step is None:
        print("--metric requires --step.", file=sys.stderr)
        return 1
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job_step_metrics(
            job_id=args.job_id,
            stage_index=args.stage_index,
            step_index=args.step,
            metric_labels=metric_labels,
        ),
        renderer=_render_metrics,
    )


def cmd_jobs_resource_metrics(args: Namespace) -> int:
    worker_ids = list(dict.fromkeys(args.worker_id))
    if len(worker_ids) > _MAX_METRICS_WORKER_IDS:
        print(
            f"Too many --worker-id values; maximum is {_MAX_METRICS_WORKER_IDS}.",
            file=sys.stderr,
        )
        return 1
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job_metrics(
            job_id=args.job_id,
            range_value=args.range,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            bucket_count=args.bucket_count,
            stage_index=args.stage_index,
            worker_ids=worker_ids,
        ),
        renderer=_render_resource_metrics,
    )
