from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _handle_error,
    _label_text,
    _print_json,
    _section_text,
    _value_text,
)
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError


def _sanitize_script_text(value: str) -> str:
    return "".join(
        ch
        for ch in value
        if ch in "\n\t"
        or (ord(ch) >= 0x20 and ch != "\x7f" and not (0x80 <= ord(ch) <= 0x9F))
    )


def _requirements_text(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    lines: list[str] = []
    for dependency in value:
        if isinstance(dependency, dict):
            lines.append(
                f"{_safe_text(dependency.get('name'))}=={_safe_text(dependency.get('version'))}"
            )
    return "\n".join(lines)


def _filtered_manifest_payload(
    payload: dict[str, Any], *, show_deps: bool, show_code: bool
) -> dict[str, Any]:
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        return payload

    filtered_manifest = dict(manifest)

    dependencies_text = _requirements_text(manifest.get("dependencies"))
    if dependencies_text is not None:
        filtered_manifest["dependencyCount"] = len(manifest["dependencies"])
        if show_deps:
            filtered_manifest["dependencies"] = dependencies_text
        else:
            filtered_manifest.pop("dependencies", None)

    script = manifest.get("script")
    if isinstance(script, dict):
        filtered_script = dict(script)
        text = filtered_script.get("text")
        if isinstance(text, str):
            if show_code:
                filtered_script["text"] = _sanitize_script_text(text)
            else:
                filtered_script.pop("text", None)
        filtered_manifest["script"] = filtered_script

    return {**payload, "manifest": filtered_manifest}


def _render_manifest(
    payload: dict[str, Any],
    *,
    show_deps: bool,
    show_code: bool,
) -> int:
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        print("Manifest unavailable.", file=sys.stderr)
        return 1
    environment = manifest.get("environment")
    print(_section_text("Runtime"))
    if isinstance(environment, dict):
        print(
            f"{_label_text('Python')}: {_value_text(environment.get('python_version'))}"
        )
        print(
            f"{_label_text('Refiner')}: {_value_text(environment.get('refiner_version'))}"
        )
        print(f"{_label_text('Platform')}: {_value_text(environment.get('platform'))}")
    else:
        print("-")
    dependencies = manifest.get("dependencies")
    if isinstance(dependencies, list):
        count = len(dependencies)
        noun = "dependency" if count == 1 else "dependencies"
        if show_deps:
            print(
                f"\n{_section_text('Dependencies')}: {_value_text(f'{count} {noun}')}"
            )
            dependencies_text = _requirements_text(dependencies)
            if dependencies_text:
                print()
                print(dependencies_text)
            else:
                print("-")
        else:
            print(
                f"\n{_section_text('Dependencies')}:"
                f" {_value_text(f'{count} {noun}')}"
                " (rerun with --deps)"
            )
    else:
        print(f"\n{_section_text('Dependencies')}: -")

    script = manifest.get("script")
    if isinstance(script, dict):
        path_text = _safe_text(script.get("path"))
        sha_text = _safe_text(script.get("sha256"))
        if show_code:
            print(f"\n{_section_text('Code')}")
            print(f"{_label_text('Path')}: {_value_text(path_text)}")
            print(f"{_label_text('SHA256')}: {_value_text(sha_text)}")
            if isinstance(script.get("text"), str) and script["text"]:
                safe_script_text = _sanitize_script_text(script["text"])
                print(f"{_label_text('Source')}:")
                print()
                print(safe_script_text)
            else:
                print(f"{_label_text('Source')}: -")
        else:
            print(f"\n{_section_text('Code')}")
            print(f"{_label_text('Path')}: {_value_text(path_text)}")
            print(f"{_label_text('SHA256')}: {_value_text(sha_text)}")
            print(f"{_label_text('Source')}: (rerun with --code)")
    else:
        print(f"\n{_section_text('Code')}: -")
    return 0


def cmd_jobs_manifest(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_manifest(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    filtered_payload = _filtered_manifest_payload(
        payload,
        show_deps=args.deps,
        show_code=args.code,
    )
    if args.json:
        return _print_json(filtered_payload)
    return _render_manifest(
        payload,
        show_deps=args.deps,
        show_code=args.code,
    )
