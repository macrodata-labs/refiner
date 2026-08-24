from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from refiner.platform.client import MacrodataClient
from refiner.cli.debug_sync import build_source_archive


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_exec_result(payload: dict[str, Any]) -> int:
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if isinstance(stdout, str) and stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if isinstance(stderr, str) and stderr:
        print(
            stderr,
            end="" if stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    return_code = payload.get("exit_code")
    return return_code if isinstance(return_code, int) else 1


def _save_profile(payload: dict[str, Any], output: str) -> Path:
    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise RuntimeError("cloud debug profile response did not contain an SVG")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(profile, encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def cmd_debug_status(args: argparse.Namespace) -> int:
    payload = MacrodataClient().cloud_debug_status(job_id=args.job_id)
    if args.json:
        _print_json(payload)
    else:
        print(f"{payload.get('status', 'unknown')}\t{args.job_id}")
    return 0


def cmd_debug_run(args: argparse.Namespace) -> int:
    if args.max_shards is not None and args.max_shards <= 0:
        raise SystemExit("--max-shards must be greater than zero")
    client = MacrodataClient()
    profile = bool(getattr(args, "profile", False))
    payload = client.cloud_debug_run(
        job_id=args.job_id,
        max_shards=args.max_shards,
        timeout_secs=args.timeout,
        profile=profile,
    )
    return_code = _emit_exec_result(payload)
    if profile:
        profile_payload = client.cloud_debug_profile(job_id=args.job_id)
        output_path = _save_profile(profile_payload, args.profile_output)
        print(f"Profile saved to {output_path}")
    return return_code


def cmd_debug_profile(args: argparse.Namespace) -> int:
    payload = MacrodataClient().cloud_debug_profile(job_id=args.job_id)
    output_path = _save_profile(payload, args.output)
    print(f"Profile saved to {output_path}")
    return 0


def cmd_debug_exec(args: argparse.Namespace) -> int:
    command = list(args.exec_command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("debug exec requires a command after --")
    payload = MacrodataClient().cloud_debug_exec(
        job_id=args.job_id,
        command=command,
        workdir=args.workdir,
        timeout_secs=args.timeout,
    )
    return _emit_exec_result(payload)


def cmd_debug_stop(args: argparse.Namespace) -> int:
    payload = MacrodataClient().cloud_debug_stop(job_id=args.job_id)
    if args.json:
        _print_json(payload)
    else:
        print(f"Debug session closed: {args.job_id}")
    return 0


def cmd_debug_sync(args: argparse.Namespace) -> int:
    archive = build_source_archive(args.path)
    payload = MacrodataClient().cloud_debug_sync(
        job_id=args.job_id,
        archive=archive.payload,
        sha256=archive.sha256,
    )
    if args.json:
        _print_json(payload)
    else:
        print(
            f"Synced {payload.get('files', archive.file_count)} files "
            f"({archive.sha256[:12]}) to {args.job_id}"
        )
    return 0


def cmd_debug_doctor(args: argparse.Namespace) -> int:
    payload = MacrodataClient().cloud_debug_doctor(job_id=args.job_id)
    _print_json(payload)
    return 0
