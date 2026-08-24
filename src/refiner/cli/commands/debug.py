from __future__ import annotations

import argparse


def register_debug_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    debug = subparsers.add_parser("debug", help="Control a retained cloud debug worker")
    debug_subparsers = debug.add_subparsers(dest="debug_command", required=True)

    status = debug_subparsers.add_parser("status", help="Show debug worker status")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_status")

    run = debug_subparsers.add_parser("run", help="Run the current pipeline once")
    run.add_argument("job_id")
    run.add_argument("--max-shards", type=int)
    run.add_argument("--timeout", type=int, default=3600)
    run.add_argument(
        "--profile",
        action="store_true",
        help="Record this attempt with py-spy and download its flamegraph",
    )
    run.add_argument(
        "--profile-output",
        default="refiner-debug-profile.svg",
        help="Path for the downloaded flamegraph",
    )
    run.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_run")

    profile = debug_subparsers.add_parser(
        "profile", help="Download the latest recorded flamegraph"
    )
    profile.add_argument("job_id")
    profile.add_argument("--output", default="refiner-debug-profile.svg")
    profile.set_defaults(
        handler_module="refiner.cli.debug", handler="cmd_debug_profile"
    )

    execute = debug_subparsers.add_parser(
        "exec", help="Execute argv in the debug worker"
    )
    execute.add_argument("job_id")
    execute.add_argument("--workdir")
    execute.add_argument("--timeout", type=int, default=300)
    execute.add_argument("exec_command", nargs=argparse.REMAINDER)
    execute.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_exec")

    stop = debug_subparsers.add_parser("stop", help="Close the debug session")
    stop.add_argument("job_id")
    stop.add_argument("--json", action="store_true")
    stop.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_stop")

    sync = debug_subparsers.add_parser("sync", help="Synchronize local source")
    sync.add_argument("job_id")
    sync.add_argument("path", nargs="?", default=".")
    sync.add_argument("--json", action="store_true")
    sync.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_sync")

    doctor = debug_subparsers.add_parser(
        "doctor", help="Inspect the exact worker environment"
    )
    doctor.add_argument("job_id")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug_doctor")
