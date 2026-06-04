from __future__ import annotations

import refiner as mdr

INPUT_DATASET = "hf://datasets/macrodata/aloha_static_battery_ep005_009"
OUTPUT_DATASET = "hf://buckets/macrodata/test_bucket/aloha_motion"


if __name__ == "__main__":
    (
        mdr.read_lerobot(INPUT_DATASET)
        .map(
            # this will compute inactive frames, drop them from the parquet rows and shift video file timestamp markers
            mdr.robot.motion_trim(
                threshold=0.001,
            )
        )
        # the writer will only copy the video sections within the new markers to the output dataset. video stats will automatically be recomputed for the episodes that were trimmed (the other ones can reuse the previous ones)
        .write_lerobot(OUTPUT_DATASET)
        .launch_cloud(
            name="motion_trim-robot",
            num_workers=1,
            mem_mb_per_worker=1024 * 2,
            secrets={
                "HF_TOKEN": "---",
            },
        )
    )
