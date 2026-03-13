from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import refiner as mdr


def _drop_video_columns(row: mdr.Row) -> mdr.Row:
    video_columns = [k for k, v in row.items() if isinstance(v, mdr.Video)]
    if not video_columns:
        return row
    return row.drop(*video_columns)


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
        default=(),
        help=(
            "Optional columns to hydrate into bytes (for example "
            "'observation.images.top')."
        ),
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        default=None,
        help=(
            "Optional output root for LeRobot writer mode. "
            "When set, runs launch_local and writes dataset artifacts there."
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=600,
        help=(
            "Hard timeout per local worker process in write mode. "
            "Prevents indefinite waits."
        ),
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help=(
            "Include video transcoding in write mode. "
            "Disabled by default to keep demo runtime bounded."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_limit = args.take if args.output_root else None
    pipeline = mdr.read_lerobot(args.root, decode=False, limit=read_limit)

    # Hydration is optional; write mode does not require it for LeRobot writer inputs.
    if args.hydrate_columns:
        for column in args.hydrate_columns:
            pipeline = pipeline.map_async(
                mdr.hydrate_media(column),
                max_in_flight=16,
            )

    if args.output_root:
        output_root = str(Path(args.output_root).expanduser().resolve())
        write_pipeline = pipeline
        if not args.include_videos:
            write_pipeline = write_pipeline.map(_drop_video_columns)
        stats = (
            write_pipeline.write_lerobot(output_root, overwrite=True)
            .launch_local(
                name="lerobot-reader-demo",
                num_workers=2,
            )
        )
        print(
            json.dumps(
                {
                    "mode": "write_lerobot",
                    "output_root": output_root,
                    "job_id": stats.job_id,
                    "workers": stats.workers,
                    "claimed": stats.claimed,
                    "completed": stats.completed,
                    "failed": stats.failed,
                    "output_rows": stats.output_rows,
                    "include_videos": bool(args.include_videos),
                },
                indent=2,
            )
        )
        return

    rows = [_summarize_episode(row) for row in pipeline.take(args.take)]
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
