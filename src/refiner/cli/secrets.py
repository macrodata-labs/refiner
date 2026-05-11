from __future__ import annotations

import argparse
import getpass
import sys

from refiner.cli.common import create_client
from refiner.cli.common import dim_text
from refiner.cli.common import handle_error
from refiner.cli.common import print_json
from refiner.cli.common import print_table
from refiner.cli.common import safe_text
from refiner.cli.ui import stdin_is_interactive
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError

SECRET_VALUE_MAX_CHARS = 32768


def _read_stdin_value() -> str:
    value = sys.stdin.read(SECRET_VALUE_MAX_CHARS + 2)
    value = value.removesuffix("\n").removesuffix("\r")
    if len(value) > SECRET_VALUE_MAX_CHARS:
        raise RuntimeError("Secret value is too large")
    if value:
        return value
    raise RuntimeError("No secret value provided on stdin")


def _read_secret_value(args: argparse.Namespace) -> str:
    if args.value_stdin:
        return _read_stdin_value()

    if not stdin_is_interactive():
        return _read_stdin_value()

    value = getpass.getpass(f"Secret value for {args.name}: ")
    if not value:
        raise RuntimeError("No secret value provided")
    return value


def cmd_secrets_list(args: argparse.Namespace) -> int:
    try:
        payload = create_client().cli_list_secrets(env=args.env)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return handle_error(err)

    if args.json:
        return print_json(payload)

    secrets = payload.get("secrets")
    if not isinstance(secrets, list) or not secrets:
        print("No secrets found.")
        return 0

    rows = [[dim_text("Env"), dim_text("Name"), dim_text("Updated")]]
    for secret in secrets:
        if not isinstance(secret, dict):
            continue
        rows.append(
            [
                safe_text(secret.get("env")),
                safe_text(secret.get("name")),
                safe_text(secret.get("updatedAt")),
            ]
        )
    print_table(rows)
    return 0


def cmd_secrets_set(args: argparse.Namespace) -> int:
    try:
        value = _read_secret_value(args)
        payload = create_client().cli_set_secret(
            name=args.name, env=args.env, value=value
        )
    except (MacrodataApiError, MacrodataCredentialsError, RuntimeError) as err:
        return handle_error(err)

    if args.json:
        return print_json(payload)

    secret_metadata = payload.get("secret")
    if isinstance(secret_metadata, dict):
        print(
            f"Saved secret {safe_text(secret_metadata.get('env'))}/"
            f"{safe_text(secret_metadata.get('name'))}."
        )
    else:
        print("Saved secret.")
    return 0


def cmd_secrets_remove(args: argparse.Namespace) -> int:
    try:
        payload = create_client().cli_delete_secret(name=args.name, env=args.env)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return handle_error(err)

    if args.json:
        return print_json(payload)

    print(f"Removed secret {safe_text(args.env)}/{safe_text(args.name)}.")
    return 0
