from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        default=None,
        help="Number of episodes to print.",
    )
    parser.add_argument(
        "--hydrate-columns",
        nargs="*",
        default=(),
        help="Optional video columns to decode in-process.",
    )
    parser.add_argument(
        "--output-dir",
        default="./tmp",
        help="Output directory for the written LeRobot dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pipeline = mdr.read_lerobot(args.root, limit=args.take)

    if args.hydrate_columns:
        for column in args.hydrate_columns:
            pipeline = pipeline.map_async(
                mdr.hydrate_media(column),
                max_in_flight=16,
            )

    stats = pipeline.write_lerobot(str(output_dir), overwrite=True).launch_local(
        name="lerobot-reader-demo",
        num_workers=1,
        workdir=str(output_dir / "workdir"),
    )

    preview_limit = args.take if args.take is not None else 3
    rows = [
        _summarize_episode(row)
        for row in mdr.read_lerobot(str(output_dir), limit=preview_limit).take(
            preview_limit
        )
    ]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "launch_stats": {
                    "job_id": stats.job_id,
                    "workers": stats.workers,
                    "claimed": stats.claimed,
                    "completed": stats.completed,
                    "failed": stats.failed,
                    "output_rows": stats.output_rows,
                },
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
