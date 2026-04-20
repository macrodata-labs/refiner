from __future__ import annotations

import argparse

from refiner.cli.auth import cmd_login, cmd_logout, cmd_whoami


def register_auth_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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
