from __future__ import annotations

import argparse

from refiner.cli.secrets import cmd_secrets_list, cmd_secrets_remove, cmd_secrets_set


def register_secrets_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    secrets = subparsers.add_parser("secrets", help="Manage workspace secrets")

    def _show_secrets_help(_args: argparse.Namespace) -> int:
        _ = _args
        secrets.print_help()
        return 0

    secrets.set_defaults(handler=_show_secrets_help)
    secrets_subparsers = secrets.add_subparsers(dest="secrets_command")

    secrets_list = secrets_subparsers.add_parser(
        "list", help="List workspace secret names"
    )
    secrets_list.add_argument("--env", default=None, help="Secret environment to list")
    secrets_list.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    secrets_list.set_defaults(handler=cmd_secrets_list)

    secrets_set = secrets_subparsers.add_parser(
        "set", help="Add or replace a workspace secret"
    )
    secrets_set.add_argument("name", help="Secret name")
    secrets_set.add_argument("--env", default="default", help="Secret environment")
    secrets_set.add_argument(
        "--value-stdin",
        action="store_true",
        help="Read the secret value from stdin",
    )
    secrets_set.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    secrets_set.set_defaults(handler=cmd_secrets_set)

    secrets_remove = secrets_subparsers.add_parser(
        "remove", aliases=["delete"], help="Remove a workspace secret"
    )
    secrets_remove.add_argument("name", help="Secret name")
    secrets_remove.add_argument("--env", default="default", help="Secret environment")
    secrets_remove.add_argument(
        "--json", action="store_true", help="Print raw JSON response"
    )
    secrets_remove.set_defaults(handler=cmd_secrets_remove)
