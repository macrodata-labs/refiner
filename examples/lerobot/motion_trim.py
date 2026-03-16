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


if __name__ == "__main__":
    (
        mdr.read_lerobot(INPUT_DATASET)
        .map_async(mdr.hydrate_video(*MOTION_VIDEO_KEYS), max_in_flight=8)
        .map(
            mdr.robotics.motion_trim(
                threshold=0.001,
                pad_frames=5,
            )
        )
        .write_lerobot(OUTPUT_DATASET)
        .launch_local(name="motion_trim", num_workers=1)
    )
