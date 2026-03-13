from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import refiner as mdr


def _default_output_root(source_root: str) -> str:
    source_dataset = Path(source_root.rstrip("/")).name
    return f"hf://datasets/hynky/{source_dataset}-refined"


def _build_hf_storage_options() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"token": token}


def _parse_hf_dataset_repo_id(root: str) -> str | None:
    prefix = "hf://datasets/"
    if not root.startswith(prefix):
        return None
    repo_and_path = root[len(prefix) :].strip("/")
    if not repo_and_path:
        return None
    parts = repo_and_path.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _ensure_hf_dataset_repo_exists(root: str) -> None:
    repo_id = _parse_hf_dataset_repo_id(root)
    if repo_id is None:
        return
    from huggingface_hub import HfApi

    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _summarize_episode(row: mdr.Row) -> dict[str, Any]:
    video_columns = [k for k, v in row.items() if isinstance(v, mdr.Video)]
    summary: dict[str, Any] = {
        "episode_index": row.get("episode_index"),
        "task": row.get("task"),
        "tasks": row.get("tasks"),
        "num_frames": len(row.get("frames", [])),
        "video_columns": video_columns,
        "metadata_keys": sorted((row.get("metadata") or {}).keys()),
    }

    if video_columns:
        first_key = video_columns[0]
        video = row[first_key]
        if isinstance(video, mdr.Video):
            summary["first_video_key"] = first_key
            summary["first_video_uri"] = video.uri
            summary["first_video_has_local_path"] = video.media.local_path is not None
            summary["first_video_bytes"] = (
                None
                if video.media.bytes_cache is None
                else len(video.media.bytes_cache)
            )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read LeRobot episodes from Hugging Face with Refiner."
    )
    parser.add_argument(
        "--source-root",
        default="hf://datasets/lerobot/aloha_static_battery",
        help=(
            "LeRobot dataset root path. Defaults to HF URI. "
            "You can also pass a local directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "HF or local output root for LeRobot writer. "
            "Defaults to hf://datasets/hynky/<dataset>-refined."
        ),
    )
    parser.add_argument(
        "--take",
        type=int,
        default=0,
        help="Maximum episodes to process. Use 0 for no limit.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of local workers for the write stage.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="LeRobot writer chunk size.",
    )
    parser.add_argument(
        "--data-files-size-mb",
        type=int,
        default=100,
        help="Max LeRobot data parquet file size in MB.",
    )
    parser.add_argument(
        "--video-files-size-mb",
        type=int,
        default=200,
        help="Max LeRobot video file size in MB.",
    )
    parser.add_argument(
        "--lease-max-in-flight",
        type=int,
        default=16,
        help="Concurrent in-flight media lease/download operations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = args.output_root.strip()
    if not output_root:
        output_root = _default_output_root(args.source_root)
    _ensure_hf_dataset_repo_exists(output_root)

    pipeline = mdr.read_lerobot(
        args.source_root,
        decode=False,
        limit=args.take if args.take > 0 else None,
    )

    writer = pipeline.write_lerobot(
        root=output_root,
        overwrite=True,
        chunk_size=args.chunk_size,
        data_files_size_in_mb=args.data_files_size_mb,
        video_files_size_in_mb=args.video_files_size_mb,
        lease_max_in_flight=args.lease_max_in_flight,
        storage_options=_build_hf_storage_options(),
    )

    t0 = time.time()
    write_stats = writer.launch_local(
        name="lerobot-reader-demo",
        num_workers=args.num_workers,
    )
    dt = time.time() - t0

    sample = {
        "claimed": write_stats.claimed,
        "failed": write_stats.failed,
        "elapsed_s": f"{dt:.2f}",
        "workers": write_stats.workers,
    }
    summary = {**sample, "output_root": output_root}
    if write_stats.failed:
        summary["status"] = "failed"
    sys.stdout.write(f"{summary}\n")


if __name__ == "__main__":
    main()
