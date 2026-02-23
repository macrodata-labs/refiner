from __future__ import annotations

import argparse
import getpass
import sys
from typing import Any

from ..platform.auth import (
    CredentialsError,
    clear_api_key,
    credentials_path,
    current_api_key,
    save_api_key,
)
from ..platform.config import resolve_platform_base_url
from ..platform.http import MacrodataApiError, verify_api_key
from .ui import display_identity, print_banner

_TOKEN_SETTINGS_SUFFIX = "/settings/tokens"


def _token_settings_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{_TOKEN_SETTINGS_SUFFIX}"


def _read_token(args: argparse.Namespace) -> str:
    if args.token and args.token.strip():
        return args.token.strip()

    read_from_stdin = args.token_stdin or not sys.stdin.isatty()
    if read_from_stdin:
        token = sys.stdin.read().strip()
        if token:
            return token
        raise RuntimeError("No API key provided on stdin")

    if not args.quiet:
        base_url = resolve_platform_base_url()
        print_banner()
        print("Create an API key here:")
        print(f"  {_token_settings_url(base_url)}")
        print("")
        if credentials_path().exists():
            print("A local API key is already stored.")
            print("Run `macrodata whoami` to see who you are logged in as.")
            print("Continuing will overwrite the stored key.")
            print("")

    token = getpass.getpass("Paste your Macrodata API key (ing_...): ").strip()
    if not token:
        raise RuntimeError("No API key provided")
    return token


def _extract_user(payload: dict[str, Any]) -> dict[str, object]:
    user = payload.get("user")
    return user if isinstance(user, dict) else {}


def cmd_login(args: argparse.Namespace) -> int:
    base_url = resolve_platform_base_url()
    try:
        token = _read_token(args)
        payload = verify_api_key(base_url=base_url, api_key=token)
        user = _extract_user(payload)
        path = save_api_key(token)
    except CredentialsError as err:
        print(f"Credential storage error: {err}", file=sys.stderr)
        return 1
    except MacrodataApiError as err:
        if err.status == 401:
            print("Invalid API key.", file=sys.stderr)
            print(
                f"Create or inspect your key: {_token_settings_url(base_url)}",
                file=sys.stderr,
            )
            return 1
        print(
            f"Failed to validate API key via {base_url}/api/me: {err}", file=sys.stderr
        )
        return 1
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    print(f"Logged in as {display_identity(user)}")
    print(f"API key name: {payload.get('name')}")
    print(f"Credentials saved to {path}")
    return 0


def cmd_whoami(_: argparse.Namespace) -> int:
    base_url = resolve_platform_base_url()
    try:
        token = current_api_key()
        payload = verify_api_key(base_url=base_url, api_key=token)
    except CredentialsError as err:
        print(f"{err}. Run `macrodata login`.", file=sys.stderr)
        return 1
    except MacrodataApiError as err:
        if err.status == 401:
            print(
                "Stored API key is invalid or expired. Run `macrodata login`.",
                file=sys.stderr,
            )
            return 1
        print(f"Failed to verify API key via {base_url}/api/me: {err}", file=sys.stderr)
        return 1

    user = _extract_user(payload)
    print(f"Logged in as {display_identity(user)}")
    print(f"API key name: {payload.get('name')}")
    return 0


def cmd_logout(_: argparse.Namespace) -> int:
    deleted = clear_api_key()
    if deleted:
        print("Logged out. Local credentials removed.")
    else:
        print("No local credentials found.")
    return 0
