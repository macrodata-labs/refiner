from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from refiner.platform.client import MacrodataClient


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
    payload = MacrodataClient().cloud_debug_run(
        job_id=args.job_id,
        max_shards=args.max_shards,
        timeout_secs=args.timeout,
    )
    return _emit_exec_result(payload)


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
