from __future__ import annotations

import argparse
import sys

from refiner.cli.auth import cmd_login, cmd_logout, cmd_whoami
from refiner.cli.jobs import (
    cmd_jobs_cancel,
    cmd_jobs_get,
    cmd_jobs_list,
    cmd_jobs_logs,
    cmd_jobs_manifest,
    cmd_jobs_metrics,
    cmd_jobs_workers,
)
from refiner.cli.run import cmd_run


def _show_parser_help(parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macrodata", description="Macrodata CLI")
    subparsers = parser.add_subparsers(dest="command")

    login = subparsers.add_parser(
        "login", help="Store and validate a Macrodata API key"
    )
    login.add_argument("--token", help="Macrodata API key (md_...)")
    login.add_argument(
        "--token-stdin",
        action="store_true",
        help="Read Macrodata API key from stdin",
    )
    login.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the banner and key-creation link prompt",
    )
    login.set_defaults(handler=cmd_login)

    whoami = subparsers.add_parser(
        "whoami", help="Validate local credentials and show identity"
    )
    whoami.set_defaults(handler=cmd_whoami)

    logout = subparsers.add_parser("logout", help="Remove local Macrodata credentials")
    logout.set_defaults(handler=cmd_logout)

    run = subparsers.add_parser(
        "run",
        help="Run a Macrodata Refiner pipeline script",
    )
    run.add_argument(
        "--logs",
        choices=("all", "none", "one", "errors"),
        default=None,
        help="Override local live log display mode via REFINER_LOCAL_LOGS",
    )
    run.add_argument("script", help="Python script to execute")
    run.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the script",
    )
    run.set_defaults(handler=cmd_run)

    jobs = subparsers.add_parser("jobs", help="Inspect Macrodata jobs")
    jobs.set_defaults(handler=lambda _args: _show_parser_help(jobs))
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
    jobs_list.add_argument("--cursor", help="Opaque pagination cursor")
    jobs_list.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_list.set_defaults(handler=cmd_jobs_list)

    jobs_get = jobs_subparsers.add_parser("get", help="Get job summary")
    jobs_get.add_argument("job_id", help="Job identifier")
    jobs_get.add_argument("--json", action="store_true", help="Print raw JSON response")
    jobs_get.set_defaults(handler=cmd_jobs_get)

    jobs_manifest = jobs_subparsers.add_parser("manifest", help="Get job manifest")
    jobs_manifest.add_argument("job_id", help="Job identifier")
    jobs_manifest.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_manifest.set_defaults(handler=cmd_jobs_manifest)

    jobs_workers = jobs_subparsers.add_parser("workers", help="List job workers")
    jobs_workers.add_argument("job_id", help="Job identifier")
    jobs_workers.add_argument("--stage", type=int, help="Filter by stage index")
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
    jobs_logs.add_argument("--limit", type=int, default=100, help="Maximum log entries")
    jobs_logs.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_logs.set_defaults(handler=cmd_jobs_logs)

    jobs_metrics = jobs_subparsers.add_parser("metrics", help="Fetch cloud job metrics")
    jobs_metrics.add_argument("job_id", help="Job identifier")
    jobs_metrics.add_argument(
        "--range",
        choices=("5m", "15m", "1h", "4h", "6h", "24h", "7d"),
        default="1h",
        help="Metrics range",
    )
    jobs_metrics.add_argument("--stage", type=int, help="Filter by stage index")
    jobs_metrics.add_argument(
        "--worker-id",
        action="append",
        default=[],
        help="Filter by worker ID; may be repeated",
    )
    jobs_metrics.add_argument(
        "--start-ms", type=int, help="Window start time in epoch ms"
    )
    jobs_metrics.add_argument("--end-ms", type=int, help="Window end time in epoch ms")
    jobs_metrics.add_argument(
        "--bucket-count", type=int, help="Requested number of metric buckets"
    )
    jobs_metrics.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_metrics.set_defaults(handler=cmd_jobs_metrics)

    jobs_cancel = jobs_subparsers.add_parser("cancel", help="Cancel a cloud job")
    jobs_cancel.add_argument("job_id", help="Job identifier")
    jobs_cancel.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    jobs_cancel.set_defaults(handler=cmd_jobs_cancel)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
