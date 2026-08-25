from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from refiner.cli.debug_sessions import (
    DebugSessionRecord,
    find_session,
    new_session_record,
    remove_session,
    save_session,
)
from refiner.cli.debug_sync import (
    build_debug_sync_bundle,
    find_project_root,
    pickle_project_modules_by_value,
)
from refiner.cli.run.command import cmd_run
from refiner.launchers.cloud import CloudLauncher
from refiner.launchers.cloud_debug_capture import capture_cloud_launches
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError, MacrodataClient


_COMMANDS = frozenset(
    {"create", "status", "run", "profile", "exec", "stop", "sync", "doctor"}
)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_exec_result(payload: dict[str, Any]) -> int:
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if isinstance(stdout, str) and stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if isinstance(stderr, str) and stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    return_code = payload.get("exit_code")
    return return_code if isinstance(return_code, int) else 1


def _save_profile(payload: dict[str, Any], output: str) -> Path:
    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise RuntimeError("cloud debug profile response did not contain an SVG")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(profile)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def _add_target(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("pipeline", nargs=None if required else "?")
    parser.add_argument("--job", dest="job_id")


def _debug_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macrodata debug")
    subparsers = parser.add_subparsers(dest="debug_command", required=True)

    create = subparsers.add_parser("create", help="Create a retained debug session")
    create.add_argument("pipeline")
    create.add_argument("--startup-timeout", type=int, default=1200)

    status = subparsers.add_parser("status", help="Show debug worker status")
    _add_target(status)
    status.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="Run the synchronized pipeline once")
    _add_target(run)
    run.add_argument("--max-shards", type=int)
    run.add_argument("--timeout", type=int, default=3600)
    run.add_argument("--profile", action="store_true")
    run.add_argument("--profile-output", default="refiner-debug-profile.svg")

    profile = subparsers.add_parser(
        "profile", help="Download the latest recorded flamegraph"
    )
    _add_target(profile)
    profile.add_argument("--output", default="refiner-debug-profile.svg")

    execute = subparsers.add_parser("exec", help="Execute argv in the debug worker")
    _add_target(execute)
    execute.add_argument("--workdir")
    execute.add_argument("--timeout", type=int, default=1200)
    execute.set_defaults(exec_command=[])

    stop = subparsers.add_parser("stop", help="Close the debug session")
    _add_target(stop)
    stop.add_argument("--json", action="store_true")

    sync = subparsers.add_parser(
        "sync", help="Synchronize source, pipeline, and private shards"
    )
    _add_target(sync, required=True)
    sync.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser(
        "doctor", help="Inspect the exact worker environment"
    )
    _add_target(doctor)
    doctor.add_argument("--json", action="store_true")
    return parser


def _parse_debug_args(raw_args: list[str]) -> argparse.Namespace:
    if not raw_args:
        _debug_parser().print_help()
        raise SystemExit(0)
    if raw_args in (["--help"], ["-h"]):
        return _debug_parser().parse_args(raw_args)
    normalized = raw_args if raw_args[0] in _COMMANDS else ["create", *raw_args]
    script_args: list[str] = []
    exec_command: list[str] | None = None
    if normalized[0] in {"create", "sync"} and "--" in normalized:
        separator = normalized.index("--")
        script_args = normalized[separator + 1 :]
        normalized = normalized[:separator]
    elif normalized[0] == "exec" and "--" in normalized:
        separator = normalized.index("--")
        exec_command = normalized[separator + 1 :]
        normalized = normalized[:separator]
    parsed = _debug_parser().parse_args(normalized)
    if parsed.debug_command in {"create", "sync"}:
        parsed.script_args = script_args
    elif parsed.debug_command == "exec" and exec_command is not None:
        parsed.exec_command = exec_command
    return parsed


def _normalized_script_args(values: list[str]) -> list[str]:
    return values[1:] if values[:1] == ["--"] else values


def _capture_launcher(script: str, script_args: list[str]) -> CloudLauncher:
    with capture_cloud_launches() as capture:
        return_code = cmd_run(
            argparse.Namespace(
                script=script,
                script_args=script_args,
                attach=False,
                detach=False,
                logs=None,
            )
        )
    if return_code != 0:
        raise SystemExit(return_code)
    return capture.single()


def _record_for_target(
    args: argparse.Namespace, client: MacrodataClient
) -> DebugSessionRecord | None:
    pipeline = getattr(args, "pipeline", None)
    return find_session(script=pipeline, client=client) if pipeline else None


def _job_for_target(
    args: argparse.Namespace, client: MacrodataClient
) -> tuple[str, DebugSessionRecord | None]:
    explicit_job = getattr(args, "job_id", None)
    if explicit_job:
        return explicit_job, None
    record = _record_for_target(args, client)
    if record is None:
        if getattr(args, "pipeline", None):
            raise SystemExit(
                "No debug session is remembered for this pipeline. "
                "Run `macrodata debug PIPELINE.py` first or pass --job JOB_ID."
            )
        raise SystemExit("Provide a pipeline path or --job JOB_ID.")
    return record.job_id, record


def _sync_launcher(
    *,
    launcher: CloudLauncher,
    pipeline: str,
    job_id: str,
    project_root: str | Path,
    client: MacrodataClient,
) -> dict[str, Any]:
    with pickle_project_modules_by_value(project_root):
        prepared = launcher.prepare_debug_sync()
    bundle = build_debug_sync_bundle(
        source_root=project_root,
        pipeline_payload=prepared.pipeline_payload,
        pipeline_sha256=prepared.pipeline_sha256,
        allocation_fingerprint=prepared.allocation_fingerprint,
    )
    payload = client.cloud_debug_sync(
        job_id=job_id,
        bundle=bundle.payload,
        sha256=bundle.sha256,
        allocation_fingerprint=prepared.allocation_fingerprint,
    )
    payload.setdefault("files", bundle.file_count)
    payload.setdefault("source_sha256", bundle.source_sha256)
    payload.setdefault("pipeline_sha256", bundle.pipeline_sha256)
    payload.setdefault("project_root", str(Path(project_root).resolve()))
    payload.setdefault("pipeline", str(Path(pipeline).resolve()))
    return payload


def _wait_until_ready(
    *, client: MacrodataClient, job_id: str, timeout_secs: int
) -> dict[str, Any]:
    if timeout_secs <= 0:
        raise SystemExit("--startup-timeout must be greater than zero")
    deadline = time.monotonic() + timeout_secs
    last_status = ""
    while True:
        payload = client.cloud_debug_status(job_id=job_id)
        status = str(payload.get("status", "unknown"))
        if status != last_status:
            print(f"Debug worker: {status}")
            last_status = status
        if status == "ready":
            return payload
        if status in {"failed", "stopped"}:
            detail = payload.get("error") or payload.get("operation_status") or status
            raise RuntimeError(f"debug worker did not start: {detail}")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"debug worker was not ready within {timeout_secs} seconds; "
                f"continue with `macrodata debug status --job {job_id}`"
            )
        time.sleep(2)


def _cmd_create(args: argparse.Namespace) -> int:
    client = MacrodataClient()
    script = Path(args.pipeline).expanduser().resolve()
    existing = find_session(script=script, client=client)
    if existing is not None:
        status = client.cloud_debug_status(job_id=existing.job_id).get("status")
        if status not in {"failed", "stopped"}:
            raise SystemExit(
                f"A debug session already exists for {script}: {existing.job_id}. "
                f"Use `macrodata debug sync {args.pipeline}` or stop it first."
            )
        remove_session(script=script, client=client)
    script_args = _normalized_script_args(list(args.script_args))
    launcher = _capture_launcher(str(script), script_args)
    project_root = find_project_root(script)
    with pickle_project_modules_by_value(project_root):
        result = launcher.launch_debug()
    record = new_session_record(
        script=script,
        project_root=project_root,
        script_args=script_args,
        job_id=result.job_id,
        client=client,
    )
    save_session(record)
    print(f"Debug session {result.job_id} remembered for {script}")
    _wait_until_ready(
        client=client,
        job_id=result.job_id,
        timeout_secs=args.startup_timeout,
    )
    payload = _sync_launcher(
        launcher=launcher,
        pipeline=str(script),
        job_id=result.job_id,
        project_root=project_root,
        client=client,
    )
    print(
        f"Ready: synchronized {payload.get('files', 0)} files and "
        f"{payload.get('shards', 0)} private shards"
    )
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    client = MacrodataClient()
    job_id, record = _job_for_target(args, client)
    script_args = _normalized_script_args(list(args.script_args))
    if not script_args and record is not None:
        script_args = record.script_args
    launcher = _capture_launcher(args.pipeline, script_args)
    project_root = (
        record.project_root if record is not None else find_project_root(args.pipeline)
    )
    payload = _sync_launcher(
        launcher=launcher,
        pipeline=args.pipeline,
        job_id=job_id,
        project_root=project_root,
        client=client,
    )
    if args.json:
        _print_json(payload)
    else:
        print(
            f"Synced {payload.get('files', 0)} files, pipeline "
            f"{str(payload.get('pipeline_sha256', ''))[:12]}, and "
            f"{payload.get('shards', 0)} private shards to {job_id}"
        )
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.debug_command == "create":
        return _cmd_create(args)
    if args.debug_command == "sync":
        return _cmd_sync(args)

    client = MacrodataClient()
    job_id, record = _job_for_target(args, client)
    if args.debug_command == "status":
        payload = client.cloud_debug_status(job_id=job_id)
        if args.json:
            _print_json(payload)
        else:
            print(f"{payload.get('status', 'unknown')}\t{job_id}")
        return 0
    if args.debug_command == "doctor":
        _print_json(client.cloud_debug_doctor(job_id=job_id))
        return 0
    if args.debug_command == "run":
        if args.max_shards is not None and args.max_shards <= 0:
            raise SystemExit("--max-shards must be greater than zero")
        payload = client.cloud_debug_run(
            job_id=job_id,
            max_shards=args.max_shards,
            timeout_secs=args.timeout,
            profile=args.profile,
        )
        return_code = _emit_exec_result(payload)
        if args.profile:
            profile = client.cloud_debug_profile(job_id=job_id)
            output_path = _save_profile(profile, args.profile_output)
            print(f"Profile saved to {output_path}")
        return return_code
    if args.debug_command == "profile":
        payload = client.cloud_debug_profile(job_id=job_id)
        output_path = _save_profile(payload, args.output)
        print(f"Profile saved to {output_path}")
        return 0
    if args.debug_command == "exec":
        command = list(args.exec_command)
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            raise SystemExit("debug exec requires a command after --")
        return _emit_exec_result(
            client.cloud_debug_exec(
                job_id=job_id,
                command=command,
                workdir=args.workdir,
                timeout_secs=args.timeout,
            )
        )
    if args.debug_command == "stop":
        payload = client.cloud_debug_stop(job_id=job_id)
        if record is not None:
            remove_session(script=record.script_path, client=client)
        if args.json:
            _print_json(payload)
        else:
            print(f"Debug session closed: {job_id}")
        return 0
    raise AssertionError("unreachable")


def cmd_debug(args: argparse.Namespace) -> int:
    try:
        if getattr(args, "debug_help", False):
            _debug_parser().print_help()
            return 0
        return _dispatch(_parse_debug_args(list(args.debug_args)))
    except (
        MacrodataApiError,
        MacrodataCredentialsError,
        RuntimeError,
        ValueError,
    ) as error:
        print(error, file=sys.stderr)
        return 1


__all__ = ["cmd_debug"]
