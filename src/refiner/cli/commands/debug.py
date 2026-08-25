from __future__ import annotations

import argparse


def register_debug_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    debug = subparsers.add_parser(
        "debug",
        help="Create and control a retained cloud debug worker",
        add_help=False,
    )
    debug.add_argument("-h", "--help", dest="debug_help", action="store_true")
    debug.add_argument("debug_args", nargs=argparse.REMAINDER)
    debug.set_defaults(handler_module="refiner.cli.debug", handler="cmd_debug")
