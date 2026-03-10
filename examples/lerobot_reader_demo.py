from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import refiner as mdr


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
        "--root",
        default="hf://datasets/lerobot/aloha_static_battery",
        help=(
            "LeRobot dataset root path. Defaults to HF URI. "
            "You can also pass a local directory."
        ),
    )
    parser.add_argument(
        "--take",
        type=int,
        default=3,
        help="Number of episodes to print.",
    )
    parser.add_argument(
        "--hydrate-columns",
        nargs="*",
        default=("observation.images.cam_left_wrist",),
        help=(
            "Optional columns to hydrate into bytes (for example "
            "'observation.images.top')."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = mdr.read_lerobot(
        args.root,
        decode=False,
    )

    if args.hydrate_columns:
        for column in args.hydrate_columns:
            pipeline = pipeline.map_async(
                mdr.hydrate_media(column, mode="file"),
                max_in_flight=16,
            )

    pipeline = pipeline.map(_summarize_episode)
    rows = pipeline.take(args.take)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
