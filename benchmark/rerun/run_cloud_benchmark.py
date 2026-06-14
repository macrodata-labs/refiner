from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsspec import url_to_fs

import refiner as mdr
from refiner.pipeline.data.row import DictRow, Row
from refiner.platform.client import MacrodataClient

DEFAULT_INPUTS = tuple(
    f"s3://macrodata-rerun-format-tests/dominique-sample/episode-{index}__base.rrd"
    for index in range(10)
)
DEFAULT_OUTPUT_ROOT = "s3://macrodata-rerun-format-tests/refiner-rerun-benchmark"
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CASES = ("recording-summary", "robotics-summary", "rrd-copy")
AWS_SECRET_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
)
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}


@dataclass(slots=True)
class StageResult:
    index: int
    name: str
    status: str
    n_shards: int | None
    shard_done: int | None
    shard_total: int | None
    requested_workers: int | None
    cpu_cores: int | None
    memory_mb: int | None
    duration_s: float | None
    metrics: dict[str, dict[str, float | int | str | None]]


@dataclass(slots=True)
class CaseResult:
    case: str
    iteration: int
    job_id: str
    status: str
    job_error: str | None
    started_at_utc: str
    finished_at_utc: str
    input_count: int
    planned_shards: int | None
    planning_warning: str | None
    output_root: str
    cloud_wall_time_s: float | None
    queue_time_s: float | None
    stage_results: list[StageResult]
    output_file_count: int | None
    output_size_bytes: int | None
    output_inspection_error: str | None
    python_version: str
    platform: str
    git_ref: str
    package_versions: dict[str, str]


def summarize_recording(row: Row) -> DictRow:
    recording = row["rerun"]
    table_summaries = {
        name: {
            "rows": table.table.num_rows,
            "columns": table.table.num_columns,
            "bytes": int(table.table.nbytes),
        }
        for name, table in recording.tables.items()
    }
    static = recording.static.table if recording.static is not None else None
    return DictRow(
        {
            "episode_id": row["episode_id"],
            "table_count": len(table_summaries),
            "tables": table_summaries,
            "static_columns": static.num_columns if static is not None else 0,
            "static_bytes": int(static.nbytes) if static is not None else 0,
            "application_id": recording.application_id,
            "recording_id": recording.recording_id,
        },
        shard_id=row.shard_id,
    )


def summarize_robotics(row: Row) -> DictRow:
    table = row["frames"].table
    action = table.column("action") if "action" in table.column_names else None
    state = (
        table.column("observation.state")
        if "observation.state" in table.column_names
        else None
    )
    return DictRow(
        {
            "episode_id": row["episode_id"],
            "num_frames": table.num_rows,
            "frame_columns": table.column_names,
            "action_type": (
                str(table.schema.field("action").type) if action is not None else None
            ),
            "state_type": (
                str(table.schema.field("observation.state").type)
                if state is not None
                else None
            ),
            "first_action_width": (
                len(action[0].as_py() or [])
                if action is not None and table.num_rows
                else 0
            ),
            "first_state_width": (
                len(state[0].as_py() or [])
                if state is not None and table.num_rows
                else 0
            ),
        },
        shard_id=row.shard_id,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Macrodata Cloud benchmarks for Rerun reader/writer paths."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help=(
            "Input RRD file, directory, or glob. Repeat for multiple inputs. "
            "Defaults to the ten Dominique sample base RRDs."
        ),
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=DEFAULT_CASES,
        help=(
            "Benchmark case to run. Repeat for multiple cases. Defaults to all "
            f"cases: {', '.join(DEFAULT_CASES)}."
        ),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--mem-mb-per-worker", type=int, default=8192)
    parser.add_argument("--timeline", default="frame")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--secret-env", default="researcher")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-token")
    parser.add_argument("--poll-interval-s", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=60.0 * 60.0)
    parser.add_argument(
        "--aws-profile",
        help=(
            "Optional AWS profile used by local output inspection after the "
            "cloud job completes."
        ),
    )
    parser.add_argument(
        "--skip-output-inspection",
        action="store_true",
        help="Do not inspect output object counts/sizes from the submitting machine.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later cases after a cloud job fails.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    return {
        "macrodata-refiner": _package_version("macrodata-refiner"),
        "rerun-sdk": _package_version("rerun-sdk"),
        "datafusion": _package_version("datafusion"),
        "pyarrow": _package_version("pyarrow"),
        "s3fs": _package_version("s3fs"),
    }


def _git_ref() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sanitize_segment(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    if not sanitized:
        raise ValueError("path segment cannot be empty")
    return sanitized


def _output_for_case(
    *,
    output_root: str,
    run_token: str,
    case: str,
    iteration: int,
) -> str:
    return "/".join(
        [
            output_root.rstrip("/"),
            _sanitize_segment(run_token),
            _sanitize_segment(case),
            f"iteration-{iteration:02d}",
        ]
    )


def _build_pipeline(
    *,
    case: str,
    inputs: Sequence[str],
    output: str,
    timeline: str,
    fps: float,
) -> mdr.RefinerPipeline:
    if case == "recording-summary":
        return (
            mdr.read_rerun(inputs, output="recording", timelines=(timeline,))
            .map(summarize_recording)
            .write_jsonl(output)
        )
    if case == "robotics-summary":
        return (
            mdr.read_rerun(
                inputs,
                output="robotics",
                contents=("/action/**", "/observation/state/**"),
                timelines=(timeline,),
                include_recording=False,
                fps=fps,
            )
            .map(summarize_robotics)
            .write_jsonl(output)
        )
    if case == "rrd-copy":
        return mdr.read_rerun(
            inputs,
            output="recording",
            materialize_tables=False,
        ).write_rerun(output)
    raise ValueError(f"Unsupported benchmark case: {case}")


def _wait_for_job(
    client: MacrodataClient,
    *,
    job_id: str,
    poll_interval_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        payload = client.cli_get_job(job_id=job_id)
        status = str(payload.get("status") or "")
        if status in TERMINAL_STATUSES:
            return payload
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for cloud job {job_id}")
        time.sleep(max(1.0, poll_interval_s))


def _duration_s(started_ms: Any, ended_ms: Any) -> float | None:
    if not isinstance(started_ms, (int, float)) or not isinstance(
        ended_ms, (int, float)
    ):
        return None
    return max(0.0, (float(ended_ms) - float(started_ms)) / 1000.0)


def _metric_values(
    client: MacrodataClient,
    *,
    job_id: str,
    stage_index: int,
    step_index: int,
    labels: Sequence[str],
) -> dict[str, dict[str, float | int | str | None]]:
    payload = client.cli_get_job_step_metrics(
        job_id=job_id,
        stage_index=stage_index,
        step_index=step_index,
        metric_labels=list(labels),
    )
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return {}
    metrics = steps[0].get("metrics")
    if not isinstance(metrics, list):
        return {}
    out: dict[str, dict[str, float | int | str | None]] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        label = metric.get("label")
        if not isinstance(label, str):
            continue
        out[label] = {
            "total": metric.get("total"),
            "rate_since_start": metric.get("rateSinceStart"),
            "per_worker": metric.get("perWorker"),
            "unit": metric.get("unit"),
        }
    return out


def _stage_results(client: MacrodataClient, job: dict[str, Any]) -> list[StageResult]:
    job_id = str(job["id"])
    stages = job.get("stages")
    if not isinstance(stages, list):
        return []
    out: list[StageResult] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_index = int(stage.get("index", 0))
        metrics: dict[str, dict[str, float | int | str | None]] = {}
        steps = stage.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_index = step.get("index")
                if not isinstance(step_index, int):
                    continue
                metrics.update(
                    _metric_values(
                        client,
                        job_id=job_id,
                        stage_index=stage_index,
                        step_index=step_index,
                        labels=(
                            "rows_read",
                            "rows_processed",
                            "rows_written",
                            "files_written",
                        ),
                    )
                )
        runtime = stage.get("runtimeConfig")
        runtime = runtime if isinstance(runtime, dict) else {}
        out.append(
            StageResult(
                index=stage_index,
                name=str(stage.get("name") or ""),
                status=str(stage.get("status") or ""),
                n_shards=_optional_int(stage.get("nShards")),
                shard_done=_optional_int(stage.get("shardDone")),
                shard_total=_optional_int(stage.get("shardTotal")),
                requested_workers=_optional_int(runtime.get("requestedNumWorkers")),
                cpu_cores=_optional_int(runtime.get("cpuCores")),
                memory_mb=_optional_int(runtime.get("memoryMb")),
                duration_s=_duration_s(stage.get("startedAt"), stage.get("endedAt")),
                metrics=metrics,
            )
        )
    return out


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _inspect_output(path: str) -> tuple[int | None, int | None, str | None]:
    try:
        fs, fs_path = url_to_fs(path)
        if not fs.exists(fs_path):
            return 0, 0, None
        total_size = 0
        total_files = 0
        for child in fs.find(fs_path):
            info = fs.info(child)
            if info.get("type") == "directory":
                continue
            total_files += 1
            total_size += int(info.get("size", 0))
        return total_files, total_size, None
    except Exception as err:
        return None, None, str(err)


def _planned_shard_count(
    pipeline: mdr.RefinerPipeline,
    *,
    requested_workers: int,
) -> tuple[int | None, str | None]:
    try:
        planned_shards = len(pipeline.list_shards())
    except Exception as err:
        return None, f"could not inspect planned shards before launch: {err}"
    if planned_shards < requested_workers:
        return (
            planned_shards,
            "planned Rerun shards are fewer than requested workers; file-atomic "
            "RRD sharding may underutilize cloud workers",
        )
    return planned_shards, None


def _run_case(
    *,
    args: argparse.Namespace,
    client: MacrodataClient,
    case: str,
    iteration: int,
    inputs: Sequence[str],
    git_ref: str,
    run_token: str,
) -> CaseResult:
    output = _output_for_case(
        output_root=args.output_root,
        run_token=run_token,
        case=case,
        iteration=iteration,
    )
    pipeline = _build_pipeline(
        case=case,
        inputs=inputs,
        output=output,
        timeline=args.timeline,
        fps=args.fps,
    )
    planned_shards, planning_warning = _planned_shard_count(
        pipeline,
        requested_workers=args.num_workers,
    )
    if planning_warning is not None:
        print(f"Warning: {case}: {planning_warning}", file=sys.stderr, flush=True)
    started_at = _utc_now()
    os.environ.setdefault("REFINER_ATTACH", "detach")
    launch = pipeline.launch_cloud(
        name=f"rerun-benchmark-{case}-{iteration:02d}-{git_ref[:8]}",
        num_workers=args.num_workers,
        cpus_per_worker=args.cpus_per_worker,
        mem_mb_per_worker=args.mem_mb_per_worker,
        secrets=mdr.Secrets.env(name=args.secret_env, keys=AWS_SECRET_KEYS),
    )
    job = _wait_for_job(
        client,
        job_id=launch.job_id,
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.timeout_s,
    )
    finished_at = _utc_now()
    output_file_count: int | None = None
    output_size_bytes: int | None = None
    output_error: str | None = None
    if not args.skip_output_inspection:
        output_file_count, output_size_bytes, output_error = _inspect_output(output)

    return CaseResult(
        case=case,
        iteration=iteration,
        job_id=launch.job_id,
        status=str(job.get("status") or ""),
        job_error=job.get("error") if isinstance(job.get("error"), str) else None,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        input_count=len(inputs),
        planned_shards=planned_shards,
        planning_warning=planning_warning,
        output_root=output,
        cloud_wall_time_s=_duration_s(job.get("startedAt"), job.get("endedAt")),
        queue_time_s=_duration_s(job.get("createdAt"), job.get("startedAt")),
        stage_results=_stage_results(client, job),
        output_file_count=output_file_count,
        output_size_bytes=output_size_bytes,
        output_inspection_error=output_error,
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        git_ref=git_ref,
        package_versions=_package_versions(),
    )


def _write_result(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_summary(
    *,
    args: argparse.Namespace,
    run_token: str,
    git_ref: str,
    inputs: Sequence[str],
    cases: Sequence[str],
    results: Sequence[CaseResult],
) -> Path:
    summary = {
        "run_token": run_token,
        "git_ref": git_ref,
        "inputs": list(inputs),
        "cases": list(cases),
        "iterations": args.iterations,
        "results": [asdict(result) for result in results],
    }
    summary_path = args.artifacts_dir / run_token / "summary.json"
    _write_result(summary_path, summary)
    return summary_path


def main() -> int:
    args = _parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.aws_profile:
        os.environ["AWS_PROFILE"] = args.aws_profile
    inputs = tuple(args.inputs or DEFAULT_INPUTS)
    cases = tuple(args.cases or DEFAULT_CASES)
    git_ref = _git_ref()
    run_token = args.run_token or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{git_ref[:8]}"
    )
    client = MacrodataClient()
    results: list[CaseResult] = []
    for iteration in range(args.iterations):
        for case in cases:
            print(f"Running {case} iteration {iteration}...", flush=True)
            result = _run_case(
                args=args,
                client=client,
                case=case,
                iteration=iteration,
                inputs=inputs,
                git_ref=git_ref,
                run_token=run_token,
            )
            results.append(result)
            result_path = (
                args.artifacts_dir / run_token / f"{case}-{iteration:02d}.json"
            )
            _write_result(result_path, asdict(result))
            print(
                f"Finished {case} iteration {iteration}: "
                f"{result.status} job={result.job_id} "
                f"cloud_wall_time_s={result.cloud_wall_time_s}",
                flush=True,
            )
            if result.status != "completed" and not args.continue_on_failure:
                summary_path = _write_summary(
                    args=args,
                    run_token=run_token,
                    git_ref=git_ref,
                    inputs=inputs,
                    cases=cases,
                    results=results,
                )
                print(f"Summary written to {summary_path}")
                return 1

    summary_path = _write_summary(
        args=args,
        run_token=run_token,
        git_ref=git_ref,
        inputs=inputs,
        cases=cases,
        results=results,
    )
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
