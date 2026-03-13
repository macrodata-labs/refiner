from __future__ import annotations

import argparse
import sys

from refiner.cli.auth import cmd_login, cmd_logout, cmd_whoami


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
