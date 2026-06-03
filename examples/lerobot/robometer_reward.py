from __future__ import annotations

import os
from datetime import datetime, timezone

import refiner as mdr

INPUT_DATASET = os.environ.get(
    "ROBOMETER_INPUT_DATASET",
    "hf://datasets/lerobot/libero",
)
OUTPUT_ROOT = os.environ.get(
    "ROBOMETER_OUTPUT_ROOT",
    "hf://buckets/macrodata/test_bucket/libero-robometer-reward",
)
VIDEO_KEY = os.environ.get("ROBOMETER_VIDEO_KEY") or None
TASK = os.environ.get("ROBOMETER_TASK") or "complete the robot manipulation task"
MAX_FRAMES = int(os.environ.get("ROBOMETER_MAX_FRAMES", "8"))
MAX_IN_FLIGHT = int(os.environ.get("ROBOMETER_MAX_IN_FLIGHT", "512"))
NUM_WORKERS = int(os.environ.get("ROBOMETER_NUM_WORKERS", "1"))
MEM_MB_PER_WORKER = int(os.environ.get("ROBOMETER_MEM_MB_PER_WORKER", "4096"))


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = f"{OUTPUT_ROOT}/{stamp}"

    (
        mdr.read_lerobot(INPUT_DATASET)
        .map_async(
            mdr.robotics.reward_score(
                model="robometer/Robometer-4B",
                video_key=VIDEO_KEY,
                task=TASK,
                max_frames=MAX_FRAMES,
                max_concurrent_requests=MAX_IN_FLIGHT,
            ),
            max_in_flight=MAX_IN_FLIGHT,
            preserve_order=False,
        )
        .write_lerobot(output)
        .launch_cloud(
            name="robometer-reward",
            num_workers=NUM_WORKERS,
            cpus_per_worker=1,
            mem_mb_per_worker=MEM_MB_PER_WORKER,
            extra_dependencies=("macrodata-refiner[hf,video]",),
            secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        )
    )


if __name__ == "__main__":
    main()
