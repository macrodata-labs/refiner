from __future__ import annotations

import argparse

from refiner.cli.jobs import (
    cmd_jobs_attach,
    cmd_jobs_cancel,
    cmd_jobs_get,
    cmd_jobs_list,
    cmd_jobs_logs,
    cmd_jobs_manifest,
    cmd_jobs_metrics,
    cmd_jobs_resource_metrics,
    cmd_jobs_workers,
)


def register_jobs_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    jobs = subparsers.add_parser("jobs", help="Inspect Macrodata jobs")

    def _show_jobs_help(_args: argparse.Namespace) -> int:
        _ = _args
        jobs.print_help()
        return 0

    jobs.set_defaults(handler=_show_jobs_help)
    jobs_subparsers = jobs.add_subparsers(dest="jobs_command")

    jobs_list = jobs_subparsers.add_parser(
        "list", help="List jobs in the current workspace"
    )
    jobs_list.add_argument("--status", help="Filter by job status")
    jobs_list.add_argument(
        "--kind", choices=("local", "cloud"), help="Filter by executor kind"
    )
    jobs_list.add_argument(
        "--limit", type=int, default=20, help="Maximum jobs to return"
    )
    jobs_list.add_argument(
        "--me",
        action="store_true",
        help="Only include jobs started by the authenticated user",
    )
    jobs_list.add_argument(
        "--cursor", help="Pagination cursor from a previous response"
    )
    jobs_list.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_list.set_defaults(handler=cmd_jobs_list)

    jobs_get = jobs_subparsers.add_parser("get", help="Get job summary")
    jobs_get.add_argument("job_id", help="Job identifier")
    jobs_get.add_argument("--json", action="store_true", help="Print raw JSON response")
    jobs_get.set_defaults(handler=cmd_jobs_get)

    jobs_attach = jobs_subparsers.add_parser(
        "attach", help="Attach to a running cloud job"
    )
    jobs_attach.add_argument("job_id", help="Job identifier")
    jobs_attach.set_defaults(handler=cmd_jobs_attach)

    jobs_manifest = jobs_subparsers.add_parser("manifest", help="Get job manifest")
    jobs_manifest.add_argument("job_id", help="Job identifier")
    jobs_manifest.add_argument(
        "--show-deps",
        action="store_true",
        help="Show dependencies from the manifest",
    )
    jobs_manifest.add_argument(
        "--show-code",
        action="store_true",
        help="Show captured script metadata from the manifest",
    )
    jobs_manifest.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_manifest.set_defaults(handler=cmd_jobs_manifest)

    jobs_workers = jobs_subparsers.add_parser("workers", help="List job workers")
    jobs_workers.add_argument("job_id", help="Job identifier")
    jobs_workers.add_argument("--stage", type=int, help="Filter by stage index")
    jobs_workers.add_argument(
        "--limit", type=int, default=20, help="Maximum workers to return"
    )
    jobs_workers.add_argument(
        "--cursor", help="Pagination cursor from a previous response"
    )
    jobs_workers.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_workers.set_defaults(handler=cmd_jobs_workers)

    jobs_logs = jobs_subparsers.add_parser("logs", help="Fetch cloud job logs")
    jobs_logs.add_argument("job_id", help="Job identifier")
    jobs_logs.add_argument("--stage", type=int, help="Filter by stage index")
    jobs_logs.add_argument("--worker", help="Filter by worker ID")
    jobs_logs.add_argument(
        "--source-type", choices=("worker", "service"), help="Filter by log source type"
    )
    jobs_logs.add_argument("--source-name", help="Filter by log source name")
    jobs_logs.add_argument(
        "--severity", choices=("info", "warning", "error"), help="Filter by severity"
    )
    jobs_logs.add_argument("--search", help="Case-insensitive substring filter")
    jobs_logs.add_argument("--start-ms", type=int, help="Window start time in epoch ms")
    jobs_logs.add_argument("--end-ms", type=int, help="Window end time in epoch ms")
    jobs_logs.add_argument(
        "--cursor", help="Pagination cursor from a previous response"
    )
    jobs_logs.add_argument("--limit", type=int, help="Maximum log entries")
    jobs_logs.add_argument(
        "--follow",
        action="store_true",
        help="Poll continuously for new log entries",
    )
    jobs_logs.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_logs.set_defaults(handler=cmd_jobs_logs)

    jobs_metrics = jobs_subparsers.add_parser(
        "metrics", help="Fetch cloud step metrics for a stage"
    )
    jobs_metrics.add_argument("job_id", help="Job identifier")
    jobs_metrics.add_argument("stage_index", type=int, help="Stage index")
    jobs_metrics.add_argument("--step", type=int, help="Filter to one step index")
    jobs_metrics.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric label to fetch for the selected step; may be repeated",
    )
    jobs_metrics.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_metrics.set_defaults(handler=cmd_jobs_metrics)

    jobs_resource_metrics = jobs_subparsers.add_parser(
        "resource-metrics", help="Fetch cloud resource metrics for a stage"
    )
    jobs_resource_metrics.add_argument("job_id", help="Job identifier")
    jobs_resource_metrics.add_argument("stage_index", type=int, help="Stage index")
    jobs_resource_metrics.add_argument(
        "--range",
        choices=("5m", "15m", "1h", "4h", "6h", "24h", "7d"),
        default="1h",
        help="Metrics range",
    )
    jobs_resource_metrics.add_argument(
        "--worker-id",
        action="append",
        default=[],
        help="Filter by worker ID; may be repeated",
    )
    jobs_resource_metrics.add_argument(
        "--start-ms", type=int, help="Window start time in epoch ms"
    )
    jobs_resource_metrics.add_argument(
        "--end-ms", type=int, help="Window end time in epoch ms"
    )
    jobs_resource_metrics.add_argument(
        "--bucket-count", type=int, help="Requested number of metric buckets"
    )
    jobs_resource_metrics.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_resource_metrics.set_defaults(handler=cmd_jobs_resource_metrics)

    jobs_cancel = jobs_subparsers.add_parser("cancel", help="Cancel a cloud job")
    jobs_cancel.add_argument("job_id", help="Job identifier")
    jobs_cancel.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_cancel.set_defaults(handler=cmd_jobs_cancel)
