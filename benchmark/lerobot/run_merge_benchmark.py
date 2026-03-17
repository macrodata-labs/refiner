from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import get_token
from refiner.io import DataFolder


DEFAULT_REPO_IDS = (
    "macrodata/aloha_static_battery_ep000_004",
    "macrodata/aloha_static_battery_ep005_009",
)
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPO_ROOT = Path(__file__).resolve().parents[2]
_CASE_COMMAND = "__run_case__"


@dataclass(slots=True)
class CaseResult:
    implementation: str
    iteration: int
    started_at_utc: str
    finished_at_utc: str
    repo_ids: list[str]
    hf_home: str
    output_root: str
    wall_time_s: float
    merged_episode_count: int
    merged_frame_count: int
    python_version: str
    platform: str
    package_versions: dict[str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LeRobot dataset merge speed for the official lerobot "
            "implementation versus Refiner."
        )
    )
    parser.add_argument(
        "--repo-id",
        dest="repo_ids",
        action="append",
        help=(
            "Input LeRobot dataset repo ID. Repeat the flag for multiple datasets. "
            f"Defaults to {', '.join(DEFAULT_REPO_IDS)}."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of cold-cache iterations to run for each implementation.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "official", "refiner"),
        default="both",
        help="Which implementation(s) to benchmark.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory where per-run outputs, cache dirs, and summary JSON are stored.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Local worker count for the Refiner write_lerobot launch.",
    )
    parser.add_argument(
        "--upload-hf-bucket",
        help=(
            "Optional Hugging Face bucket prefix used for benchmark outputs. "
            "Accepted forms are 'owner/bucket[/prefix]' or "
            "'hf://buckets/owner/bucket[/prefix]'. The benchmark derives a "
            "unique per-run subdirectory for each implementation and iteration."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for isolated case subprocesses.",
    )
    parser.add_argument(_CASE_COMMAND, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--impl", help=argparse.SUPPRESS)
    parser.add_argument("--iteration-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-token", help=argparse.SUPPRESS)
    return parser.parse_args()


def _prepare_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prepare_absent_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _configure_isolated_hf_cache(hf_home: Path) -> None:
    auth_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    _prepare_empty_dir(hf_home)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_home / "assets")
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    if auth_token:
        os.environ["HF_TOKEN"] = auth_token
        (hf_home / "token").write_text(auth_token, encoding="utf-8")


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {
        "huggingface-hub": _package_version("huggingface-hub"),
        "pyarrow": _package_version("pyarrow"),
        "av": _package_version("av"),
        "macrodata-refiner": _package_version("macrodata-refiner"),
        "lerobot": _package_version("lerobot"),
    }
    return versions


def _normalize_hf_bucket_prefix(value: str) -> str:
    bucket_prefix = value.removeprefix("hf://buckets/").strip().strip("/")
    parts = [part for part in bucket_prefix.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "--upload-hf-bucket must be 'owner/bucket[/prefix]' or "
            "'hf://buckets/owner/bucket[/prefix]'"
        )
    return f"hf://buckets/{'/'.join(parts)}"


def _sanitize_bucket_segment(value: str) -> str:
    lowered = value.lower()
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", lowered).strip("-")
    if not sanitized:
        raise ValueError("upload bucket segment cannot be empty")
    return sanitized


def _bucket_id_from_uri(bucket_uri: str) -> str:
    relative = bucket_uri.removeprefix("hf://buckets/").strip("/")
    parts = [part for part in relative.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid Hugging Face bucket URI: {bucket_uri!r}")
    return f"{parts[0]}/{parts[1]}"


def _build_bucket_output_root(
    *,
    bucket_prefix: str,
    run_token: str,
    implementation: str,
    iteration: int,
) -> str:
    run_segment = _sanitize_bucket_segment(run_token)
    impl_segment = _sanitize_bucket_segment(implementation)
    iteration_segment = _sanitize_bucket_segment(f"iteration-{iteration:02d}")
    return "/".join(
        [
            bucket_prefix.rstrip("/"),
            run_segment,
            impl_segment,
            iteration_segment,
        ]
    )


def _read_lerobot_totals(output_root: str | Path) -> tuple[int, int]:
    folder = DataFolder.resolve(str(output_root))
    with folder.open("meta/info.json", mode="rt", encoding="utf-8") as src:
        info = json.load(src)
    return (
        int(info.get("total_episodes", 0)),
        int(info.get("total_frames", 0)),
    )


def _run_official_case(
    *,
    repo_ids: list[str],
    run_dir: Path,
    iteration: int,
    output_root: str | None,
) -> CaseResult:
    from importlib import import_module

    from huggingface_hub import create_bucket, sync_bucket

    merge_datasets = import_module("lerobot.datasets.dataset_tools").merge_datasets
    LeRobotDataset = import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset

    hf_home = run_dir / "hf-home"
    local_output_root = run_dir / "merged-official"
    _prepare_absent_path(local_output_root)

    started_at = datetime.now(timezone.utc)
    total_start = perf_counter()

    datasets = [
        LeRobotDataset(repo_id=repo_id, download_videos=True) for repo_id in repo_ids
    ]
    merge_datasets(
        datasets=datasets,
        output_repo_id="benchmark-merged-official",
        output_dir=local_output_root,
    )
    if output_root is not None:
        create_bucket(_bucket_id_from_uri(output_root), exist_ok=True)
        sync_bucket(
            source=str(local_output_root),
            dest=output_root,
            delete=True,
            quiet=True,
        )

    wall_time_s = perf_counter() - total_start
    finished_at = datetime.now(timezone.utc)
    final_output_root = (
        output_root if output_root is not None else str(local_output_root)
    )
    merged_episode_count, merged_frame_count = _read_lerobot_totals(local_output_root)

    return CaseResult(
        implementation="official",
        iteration=iteration,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        repo_ids=repo_ids,
        hf_home=str(hf_home),
        output_root=final_output_root,
        wall_time_s=wall_time_s,
        merged_episode_count=merged_episode_count,
        merged_frame_count=merged_frame_count,
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_package_versions(),
    )


def _run_refiner_case(
    *,
    repo_ids: list[str],
    run_dir: Path,
    iteration: int,
    num_workers: int,
    output_root: str | Path,
) -> CaseResult:
    import refiner as mdr
    from refiner.pipeline.utils.cache.decoder_cache import reset_video_decoder_cache
    from refiner.pipeline.utils.cache.file_cache import reset_media_cache

    hf_home = run_dir / "hf-home"
    workdir = run_dir / "workdir"
    if isinstance(output_root, Path):
        _prepare_empty_dir(output_root)
    _prepare_empty_dir(workdir)
    reset_media_cache()
    reset_video_decoder_cache()

    started_at = datetime.now(timezone.utc)
    total_start = perf_counter()

    launch_stats = (
        mdr.read_lerobot([f"hf://datasets/{repo_id}" for repo_id in repo_ids])
        .write_lerobot(str(output_root))
        .launch_local(
            name=f"lerobot-benchmark-merge-{iteration}",
            num_workers=num_workers,
            workdir=str(workdir),
        )
    )
    if int(launch_stats.failed) != 0:
        raise RuntimeError(
            f"Refiner merge failed with {launch_stats.failed} failed shards"
        )

    wall_time_s = perf_counter() - total_start
    finished_at = datetime.now(timezone.utc)
    merged_episode_count, merged_frame_count = _read_lerobot_totals(output_root)

    return CaseResult(
        implementation="refiner",
        iteration=iteration,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        repo_ids=repo_ids,
        hf_home=str(hf_home),
        output_root=str(output_root),
        wall_time_s=wall_time_s,
        merged_episode_count=merged_episode_count,
        merged_frame_count=merged_frame_count,
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_package_versions(),
    )


def _run_case_worker(args: argparse.Namespace) -> int:
    if args.impl is None or args.iteration_index is None:
        raise ValueError("worker mode requires --impl and --iteration-index")
    if args.run_dir is None or args.result_path is None:
        raise ValueError("worker mode requires --run-dir and --result-path")

    repo_ids = list(args.repo_ids or DEFAULT_REPO_IDS)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_isolated_hf_cache(args.run_dir / "hf-home")
    upload_bucket_prefix = (
        _normalize_hf_bucket_prefix(args.upload_hf_bucket)
        if args.upload_hf_bucket is not None
        else None
    )
    upload_output_root = (
        _build_bucket_output_root(
            bucket_prefix=upload_bucket_prefix,
            run_token=args.run_token or "benchmark",
            implementation=str(args.impl),
            iteration=int(args.iteration_index),
        )
        if upload_bucket_prefix is not None
        else None
    )

    print(f"upload_output_root: {upload_output_root}")

    if args.impl == "official":
        result = _run_official_case(
            repo_ids=repo_ids,
            run_dir=args.run_dir,
            iteration=args.iteration_index,
            output_root=upload_output_root,
        )
    elif args.impl == "refiner":
        result = _run_refiner_case(
            repo_ids=repo_ids,
            run_dir=args.run_dir,
            iteration=args.iteration_index,
            num_workers=args.num_workers,
            output_root=upload_output_root or (args.run_dir / "merged-refiner"),
        )
    else:
        raise ValueError(f"Unknown implementation: {args.impl!r}")

    args.result_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "implementation": result.implementation,
                "iteration": result.iteration,
                "wall_time_s": round(result.wall_time_s, 3),
                "merged_episode_count": result.merged_episode_count,
                "merged_frame_count": result.merged_frame_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _subprocess_case(
    *,
    implementation: str,
    iteration: int,
    repo_ids: list[str],
    artifacts_dir: Path,
    python_executable: str,
    num_workers: int,
    upload_hf_bucket: str | None,
    run_token: str,
) -> CaseResult:
    case_dir = artifacts_dir / f"iteration-{iteration:02d}" / implementation
    result_path = case_dir / "result.json"
    cmd = [
        python_executable,
        str(Path(__file__).resolve()),
        _CASE_COMMAND,
        "--impl",
        implementation,
        "--iteration-index",
        str(iteration),
        "--run-dir",
        str(case_dir),
        "--result-path",
        str(result_path),
        "--num-workers",
        str(num_workers),
        "--run-token",
        run_token,
    ]
    if upload_hf_bucket is not None:
        cmd.extend(["--upload-hf-bucket", upload_hf_bucket])
    for repo_id in repo_ids:
        cmd.extend(["--repo-id", repo_id])

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark case failed for {implementation} iteration {iteration}"
        )

    return CaseResult(**json.loads(result_path.read_text(encoding="utf-8")))


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_impl: dict[str, list[CaseResult]] = {}
    for result in results:
        by_impl.setdefault(result.implementation, []).append(result)

    summary_impls: dict[str, Any] = {}
    for implementation, items in by_impl.items():
        wall_times = [item.wall_time_s for item in items]
        summary_impls[implementation] = {
            "iterations": len(items),
            "wall_time_s": {
                "min": min(wall_times),
                "max": max(wall_times),
                "mean": statistics.fmean(wall_times),
            },
            "merged_episode_count_mean": statistics.fmean(
                item.merged_episode_count for item in items
            ),
            "merged_frame_count_mean": statistics.fmean(
                item.merged_frame_count for item in items
            ),
        }

    comparison: dict[str, Any] = {}
    official = summary_impls.get("official")
    refiner = summary_impls.get("refiner")
    if official is not None and refiner is not None:
        official_mean = float(official["wall_time_s"]["mean"])
        refiner_mean = float(refiner["wall_time_s"]["mean"])
        comparison = {
            "official_over_refiner_wall_time_ratio": (
                official_mean / refiner_mean if refiner_mean > 0 else None
            ),
            "refiner_over_official_wall_time_ratio": (
                refiner_mean / official_mean if official_mean > 0 else None
            ),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
        "summary_by_implementation": summary_impls,
        "comparison": comparison,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    summary_by_impl = summary["summary_by_implementation"]
    print("\nImplementation summary")
    for implementation in ("official", "refiner"):
        item = summary_by_impl.get(implementation)
        if item is None:
            continue
        wall = item["wall_time_s"]
        print(
            f"- {implementation}: mean={wall['mean']:.3f}s "
            f"min={wall['min']:.3f}s max={wall['max']:.3f}s"
        )
        print(f"  mean_merged_episode_count={item['merged_episode_count_mean']:.0f}")
        print(f"  mean_merged_frame_count={item['merged_frame_count_mean']:.0f}")

    comparison = summary.get("comparison", {})
    if comparison:
        if comparison.get("refiner_over_official_wall_time_ratio") is not None:
            print(
                "- refiner_over_official_wall_time_ratio="
                f"{comparison['refiner_over_official_wall_time_ratio']:.3f}"
            )
        if comparison.get("official_over_refiner_wall_time_ratio") is not None:
            print(
                "- official_over_refiner_wall_time_ratio="
                f"{comparison['official_over_refiner_wall_time_ratio']:.3f}"
            )


def main() -> int:
    args = _parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")

    if args.__dict__.get(_CASE_COMMAND):
        return _run_case_worker(args)

    repo_ids = list(args.repo_ids or DEFAULT_REPO_IDS)
    implementations = ["official", "refiner"] if args.mode == "both" else [args.mode]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_dir = args.artifacts_dir / timestamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for iteration in range(1, args.iterations + 1):
        for implementation in implementations:
            print(
                f"Running {implementation} iteration {iteration}/{args.iterations} "
                f"with cold HF cache"
            )
            result = _subprocess_case(
                implementation=implementation,
                iteration=iteration,
                repo_ids=repo_ids,
                artifacts_dir=artifacts_dir,
                python_executable=args.python,
                num_workers=args.num_workers,
                upload_hf_bucket=args.upload_hf_bucket,
                run_token=timestamp,
            )
            results.append(result)

    summary = _summarize(results)
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_summary(summary)
    print(f"\nSummary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
