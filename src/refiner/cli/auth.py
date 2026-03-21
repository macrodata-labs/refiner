from __future__ import annotations

import argparse
import getpass
import sys

from refiner.platform.auth import (
    CredentialsError,
    clear_api_key,
    credentials_path,
    current_api_key,
    save_api_key,
)
from refiner.platform.client import (
    MacrodataApiError,
    VerifyApiKeyResponse,
    resolve_platform_base_url,
    sanitize_terminal_text,
    verify_api_key,
)
from refiner.cli.ui import display_identity, print_banner

_TOKEN_SETTINGS_SUFFIX = "/settings/api-keys"


def stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:  # pragma: no cover
        return False


def _token_settings_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{_TOKEN_SETTINGS_SUFFIX}"


def _read_token(args: argparse.Namespace) -> str:
    if args.token and args.token.strip():
        return args.token.strip()

    read_from_stdin = args.token_stdin or not stdin_is_interactive()
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

    token = getpass.getpass("Paste your Macrodata API key (md_...): ").strip()
    if not token:
        raise RuntimeError("No API key provided")
    return token


def _workspace_display(identity: VerifyApiKeyResponse) -> str | None:
    workspace = identity.workspace
    if workspace is None:
        return None
    name = sanitize_terminal_text(workspace.name).strip()
    slug = sanitize_terminal_text(workspace.slug).strip()
    if not name and not slug:
        return None
    if name and slug:
        return f"{name} ({slug})"
    return name or slug


def _api_key_name_display(identity: VerifyApiKeyResponse) -> str:
    return sanitize_terminal_text(identity.name).strip()


def cmd_login(args: argparse.Namespace) -> int:
    base_url = resolve_platform_base_url()
    try:
        token = _read_token(args)
        payload = verify_api_key(base_url=base_url, api_key=token)
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

    print(f"Logged in as {display_identity(payload.user)}")
    print(f"API key name: {_api_key_name_display(payload)}")
    workspace = _workspace_display(payload)
    if workspace:
        print(f"Workspace: {workspace}")
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

    print(f"Logged in as {display_identity(payload.user)}")
    print(f"API key name: {_api_key_name_display(payload)}")
    workspace = _workspace_display(payload)
    if workspace:
        print(f"Workspace: {workspace}")
    return 0


def cmd_logout(_: argparse.Namespace) -> int:
    deleted = clear_api_key()
    if deleted:
        print("Logged out. Local credentials removed.")
    else:
        print("No local credentials found.")
    return 0
