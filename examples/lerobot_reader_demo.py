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
            summary["first_video_frame_count"] = getattr(
                video.media,
                "frame_count",
                None,
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
        default=None,
        help="Number of episodes to print.",
    )
    parser.add_argument(
        "--hydrate-columns",
        nargs="*",
        default=(),
        help=(
            "Optional video columns to decode in-process (for example "
            "'observation.images.top')."
        ),
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        default="./lerobot-reader-demo",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_limit = args.take
    pipeline = mdr.read_lerobot(args.root, limit=read_limit)

    # Hydration is optional; write mode does not require it for LeRobot writer inputs.
    if args.hydrate_columns:
        for column in args.hydrate_columns:
            pipeline = pipeline.map_async(
                mdr.hydrate_media(column, decode=True),
                max_in_flight=16,
            )

    if args.output_root:
        output_root = args.output_root
        stats = pipeline.write_lerobot(output_root, overwrite=True).launch_local(
            name="lerobot-reader-demo",
            num_workers=1,
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
                },
                indent=2,
            )
        )
        return

    rows = [_summarize_episode(row) for row in pipeline.take(args.take)]
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
