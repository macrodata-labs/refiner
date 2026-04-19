from __future__ import annotations

import argparse

from refiner.cli.run.command import cmd_run


def register_run_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = subparsers.add_parser(
        "run",
        help="Run a Macrodata Refiner pipeline script",
    )
    attach_mode = run.add_mutually_exclusive_group()
    attach_mode.add_argument(
        "--attach",
        action="store_true",
        help="Force attached mode for cloud launches",
    )
    attach_mode.add_argument(
        "--detach",
        action="store_true",
        help="Force detached mode for cloud launches",
    )
    run.add_argument(
        "--logs",
        choices=("all", "none", "one", "errors"),
        default=None,
        help="Override attached live log display mode via REFINER_LOGS",
    )
    run.add_argument("script", help="Python script to execute")
    run.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the script",
    )
    run.set_defaults(handler=cmd_run)
