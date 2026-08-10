from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (
        mdr.read_lerobot(
            "hf://datasets/nvidia/LIBERO_LeRobot_v3/libero_90",
            num_shards=5,
        )
        .map_async(
            mdr.robotics.reward_score(
                model="robometer/Robometer-4B",
                video_key="observation.images.image",
                # Usually you will have task in the row itself use lambda row: row.task or "; ".join(row.tasks) if them
                task="complete the robot manipulation task",
                # This is robometer default, we don't recommend going above 16
                max_frames=8,
                max_concurrent_requests=256,
            ),
            max_in_flight=256,
        )
        .write_lerobot(
            f"hf://buckets/macrodata/test_bucket/libero-robometer-reward/{stamp}"
        )
        .launch_local(
            name="robometer-reward",
            num_workers=1,
        )
    )


if __name__ == "__main__":
    main()
