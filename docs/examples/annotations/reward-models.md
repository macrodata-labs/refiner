---
title: "Score Rewards"
description: "Add progress and success scores to episode videos"
---

# Score Rewards

This example uses [Robometer](https://arxiv.org/abs/2603.02115) to compute reward scores over episodes in Libero.
Each episode will be annotated with `reward_score`, `robometer_success`, and `reward_frames` columns, signaling the reward, tasks completion state for each frame in
`reward_frames`.

```python
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
        .filter(lambda row: row.episode_index == 0)
        .map_async(
            mdr.robotics.reward_score(
                model="robometer/Robometer-4B",
                video_key=None,
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
        .launch_cloud(
            name="robometer-reward",
            num_workers=1,
            cpus_per_worker=1,
            mem_mb_per_worker=4096,
            extra_dependencies=("macrodata-refiner[hf,video]",),
            secrets={"HF_TOKEN": None},
        )
    )


if __name__ == "__main__":
    main()
```

The output path includes a UTC timestamp so repeated runs do not overwrite prior
results. `HF_TOKEN` is loaded from the local submitter environment and mounted
as a worker secret so workers can read and write Hugging Face datasets and
buckets.

See [Reward Scoring](../../episode-operations/reward-scoring.md).
