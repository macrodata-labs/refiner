from __future__ import annotations

import argparse
import sys

from refiner.cli.commands.auth import register_auth_commands
from refiner.cli.commands.jobs import register_jobs_command
from refiner.cli.commands.run import register_run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macrodata", description="Macrodata CLI")
    subparsers = parser.add_subparsers(dest="command")
    register_auth_commands(subparsers)
    register_run_command(subparsers)
    register_jobs_command(subparsers)

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
