from __future__ import annotations

from argparse import Namespace
import sys
from typing import Any

from refiner.cli.job_utils import safe_text as _safe_text
from refiner.cli.jobs.common import _client, _run_job_command


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
    print("Runtime")
    if isinstance(environment, dict):
        print(f"Python: {_safe_text(environment.get('python_version'))}")
        print(f"Refiner: {_safe_text(environment.get('refiner_version'))}")
        print(f"Platform: {_safe_text(environment.get('platform'))}")
    else:
        print("-")
    if show_deps:
        dependencies = manifest.get("dependencies")
        print("\nDependencies")
        if isinstance(dependencies, list) and dependencies:
            for dependency in dependencies:
                if isinstance(dependency, dict):
                    print(
                        f"{_safe_text(dependency.get('name'))}=={_safe_text(dependency.get('version'))}"
                    )
        else:
            print("-")
    if show_code:
        script = manifest.get("script")
        print("\nCode")
        if isinstance(script, dict):
            print(f"Path: {_safe_text(script.get('path'))}")
            print(f"SHA256: {_safe_text(script.get('sha256'))}")
            if isinstance(script.get("text"), str) and script["text"]:
                safe_script_text = "".join(
                    ch
                    for ch in script["text"]
                    if ch in "\n\t"
                    or (
                        ord(ch) >= 0x20
                        and ch != "\x7f"
                        and not (0x80 <= ord(ch) <= 0x9F)
                    )
                )
                print("\n" + safe_script_text)
        else:
            print("-")
    return 0


def cmd_jobs_manifest(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job_manifest(job_id=args.job_id),
        renderer=lambda payload: _render_manifest(
            payload,
            show_deps=args.show_deps,
            show_code=args.show_code,
        ),
    )
