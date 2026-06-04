---
title: "Score Rewards"
description: "Add progress and success scores to episode videos"
---

# Score Rewards

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/nvidia/LIBERO_LeRobot_v3/libero_90")
    .map_async(
        mdr.robotics.reward_score(
            model="robometer/Robometer-4B",
            video_key="observation.images.image",
            task="complete the robot manipulation task",
            max_frames=8,
            max_concurrent_requests=256,
        ),
        max_in_flight=256,
        preserve_order=False,
    )
    .write_lerobot("hf://buckets/acme-robotics/libero-robometer-reward")
)
```

See [Reward Scoring](../episode-operations/reward-scoring.md).
