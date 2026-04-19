from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _dim_text,
    _label_text,
    _print_table,
    _run_job_command,
    _section_text,
    _timestamp_text,
    _value_text,
)

_MAX_METRICS_WORKER_IDS = 50
_MIN_RESOURCE_METRICS_BUCKETS = 20
_MAX_RESOURCE_METRICS_BUCKETS = 240


def _render_inventory_table(metrics: list[dict[str, Any]]) -> None:
    rows = [[_dim_text("Label"), _dim_text("Kind"), _dim_text("Details")]]
    for metric in metrics:
        rows.append(
            [
                _value_text(metric.get("label")),
                _safe_text(metric.get("metricKind")),
                _metric_details_text(metric),
            ]
        )
    _print_table(rows)


def _print_metric_field(label: str, value: Any) -> None:
    print(f"{_label_text(label)}: {_value_text(value)}")


def _ranking_key(metric: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _safe_text(metric.get("metricKind")),
        _safe_text(metric.get("label")),
        _safe_text(metric.get("kind")),
    )


def _ranking_title(metric: dict[str, Any], sort: str | None) -> str:
    direction = "↑" if sort == "asc" else "↓"
    metric_kind = _safe_text(metric.get("metricKind"))
    if metric_kind == "counter":
        return f"Workers (Rate {direction})"
    if metric_kind == "gauge":
        return f"Workers (Avg 5m {direction})"
    return "Workers"


def _render_metric_rankings(
    rankings: list[dict[str, Any]], metric: dict[str, Any], sort: str | None
) -> None:
    ranking = next(
        (item for item in rankings if _ranking_key(item) == _ranking_key(metric)),
        None,
    )
    if not isinstance(ranking, dict):
        return
    workers = ranking.get("workers")
    if not isinstance(workers, list) or not workers:
        return
    print()
    print(_section_text(_ranking_title(metric, sort)))
    kind = _safe_text(metric.get("metricKind"))
    if kind == "counter":
        rows = [[_dim_text("Worker"), _dim_text("Total"), _dim_text("Rate / sec")]]
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            rows.append(
                [
                    _value_text(worker.get("workerId")),
                    _safe_text(worker.get("total")),
                    _safe_text(worker.get("ratePerSec")),
                ]
            )
        _print_table(rows)
        return
    if kind == "gauge":
        rows = [
            [
                _dim_text("Worker"),
                _dim_text("Avg 5m"),
                _dim_text("Max 5m"),
            ]
        ]
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            rows.append(
                [
                    _value_text(worker.get("workerId")),
                    _safe_text(worker.get("avgLast5m")),
                    _safe_text(worker.get("maxLast5m")),
                ]
            )
        _print_table(rows)


def _render_value_metrics(
    metrics: list[dict[str, Any]],
    rankings: list[dict[str, Any]] | None = None,
    sort: str | None = None,
) -> None:
    rendered_any = False
    for metric in metrics:
        if rendered_any:
            print()
        rendered_any = True
        print(
            f"{_section_text(_safe_text(metric.get('label')))} "
            f"({_dim_text(metric.get('metricKind'))})"
        )
        kind = metric.get("metricKind")
        if kind == "counter":
            _print_metric_field("Total", metric.get("total"))
            _print_metric_field("Rate (lifetime)", metric.get("rateSinceStart"))
            _print_metric_field("Per Worker (lifetime)", metric.get("perWorker"))
            unit = _safe_text(metric.get("unit"))
            if unit != "-":
                _print_metric_field("Unit", unit)
            if rankings:
                _render_metric_rankings(rankings, metric, sort)
            continue
        if kind == "histogram":
            _print_metric_field("Total", metric.get("total"))
            _print_metric_field("Samples", metric.get("count"))
            _print_metric_field("Per", metric.get("per"))
            _print_metric_field("Unit", metric.get("unit"))
            _print_metric_field("Min", metric.get("min"))
            _print_metric_field("Mean", metric.get("average"))
            _print_metric_field("Max", metric.get("max"))
            continue
        if kind == "gauge":
            gauge_kind = _safe_text(metric.get("kind"))
            if gauge_kind != "-":
                _print_metric_field("Gauge Kind", gauge_kind)
            _print_metric_field("Unit", metric.get("unit"))
            _print_metric_field("Avg lifetime", metric.get("avgAllTime"))
            _print_metric_field("Avg 5m", metric.get("avgLast5m"))
            _print_metric_field("Max 5m", metric.get("maxLast5m"))
            if rankings:
                _render_metric_rankings(rankings, metric, sort)
            continue
        _print_metric_field("Details", _metric_details_text(metric))


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


def _render_metrics(payload: dict[str, Any], sort: str | None = None) -> int:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        print("Metrics unavailable.", file=sys.stderr)
        return 1
    if not steps:
        print("No step metrics found.")
        return 0

    print(
        f"{_label_text('Job')}: {_value_text(payload.get('jobId'))}"
        f"  {_label_text('Stage')}: {_value_text(payload.get('stageIndex'))}"
    )
    detail_level = _safe_text(payload.get("detailLevel"))
    if detail_level != "-":
        print(f"{_label_text('Detail')}: {_value_text(detail_level)}")
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_title = _section_text(f"Step {_safe_text(step.get('stepIndex'))}")
        print(
            "\n"
            f"{step_title}: "
            f"{_value_text(step.get('name'))} "
            f"({_dim_text(step.get('type'))})"
        )
        metrics = step.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            print("No metrics.")
            continue
        normalized_metrics = [metric for metric in metrics if isinstance(metric, dict)]
        rankings = step.get("rankings")
        normalized_rankings = (
            [ranking for ranking in rankings if isinstance(ranking, dict)]
            if isinstance(rankings, list)
            else None
        )
        if detail_level == "values":
            _render_value_metrics(normalized_metrics, normalized_rankings, sort)
        else:
            _render_inventory_table(normalized_metrics)
    if detail_level == "inventory":
        print(
            "\n"
            f"{_dim_text('Tip')}: "
            f"{_value_text('rerun with --step <index> --metric <label> to fetch metric values')}"
        )
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

    print(
        f"{_label_text('Job')}: {_value_text(payload.get('jobId'))}"
        f"  {_label_text('Stage')}: {_value_text(payload.get('stageIndex'))}"
    )
    print(f"{_label_text('Range')}: {_value_text(payload.get('range'))}")
    print(
        f"{_label_text('Latest sample')}: {_timestamp_text(_format_ts(latest.get('t')))}"
    )
    print(
        f"{_label_text('CPU')}: {_value_text(latest.get('cpuUsage'))} / {_value_text(latest.get('cpuQuota'))}"
        f"  {_label_text('Memory')}: {_value_text(latest.get('memoryUsage'))} / {_value_text(latest.get('memoryLimit'))}"
    )
    print(
        f"{_label_text('Network In')}: {_value_text(latest.get('networkInMb'))} MB"
        f"  {_label_text('Network Out')}: {_value_text(latest.get('networkOutMb'))} MB"
    )
    print(f"{_label_text('Samples')}: {_value_text(len(resources))}")
    return 0


def cmd_jobs_metrics(args: Namespace) -> int:
    metric_labels = list(dict.fromkeys(args.metric))
    worker_ids = list(dict.fromkeys(args.worker))
    if metric_labels and args.step is None:
        print("--metric requires --step.", file=sys.stderr)
        return 1
    if worker_ids and args.step is None:
        print("--worker requires --step.", file=sys.stderr)
        return 1
    if worker_ids and not metric_labels:
        print("--worker requires --metric.", file=sys.stderr)
        return 1
    if args.workers and args.step is None:
        print("--workers requires --step.", file=sys.stderr)
        return 1
    if args.workers and not metric_labels:
        print("--workers requires --metric.", file=sys.stderr)
        return 1
    if args.asc and args.desc:
        print("--asc and --desc are mutually exclusive.", file=sys.stderr)
        return 1
    if (args.asc or args.desc) and not args.workers:
        print("--asc/--desc require --workers.", file=sys.stderr)
        return 1
    if len(worker_ids) > _MAX_METRICS_WORKER_IDS:
        print(
            f"Too many --worker values; maximum is {_MAX_METRICS_WORKER_IDS}.",
            file=sys.stderr,
        )
        return 1
    sort = "asc" if args.asc else "desc" if args.desc else None
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job_step_metrics(
            job_id=args.job_id,
            stage_index=args.stage_index,
            step_index=args.step,
            metric_labels=metric_labels,
            workers=args.workers or None,
            worker_ids=worker_ids,
            sort=sort,
        ),
        renderer=lambda payload: _render_metrics(payload, sort),
    )


def cmd_jobs_resource_metrics(args: Namespace) -> int:
    worker_ids = list(dict.fromkeys(args.worker_id))
    if len(worker_ids) > _MAX_METRICS_WORKER_IDS:
        print(
            f"Too many --worker-id values; maximum is {_MAX_METRICS_WORKER_IDS}.",
            file=sys.stderr,
        )
        return 1
    if args.bucket_count is not None and not (
        _MIN_RESOURCE_METRICS_BUCKETS
        <= args.bucket_count
        <= _MAX_RESOURCE_METRICS_BUCKETS
    ):
        print(
            "--bucket-count must be between "
            f"{_MIN_RESOURCE_METRICS_BUCKETS} and {_MAX_RESOURCE_METRICS_BUCKETS}.",
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
