from __future__ import annotations

import refiner as mdr

MOTION_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_low",
    "observation.images.cam_right_wrist",
)
INPUT_DATASET = "hf://datasets/macrodata/aloha_static_battery_ep005_009"
OUTPUT_DATASET = "hf://buckets/macrodata/test_bucket/aloha_motion"


def _invalidate_trimmed_video_stats(row):
    return row.update({f"stats/{key}": None for key in MOTION_VIDEO_KEYS if key in row})


if __name__ == "__main__":
    (
        mdr.read_lerobot(INPUT_DATASET)
        .map(
            mdr.robotics.motion_trim(
                threshold=0.001,
                pad_frames=5,
            )
        )
        .map(_invalidate_trimmed_video_stats)
        .write_lerobot(OUTPUT_DATASET)
        .launch_local(name="motion_trim", num_workers=1)
    )
