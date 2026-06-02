---
title: "Score Rewards"
description: "Add progress and success scores to episode videos"
---

# Score Rewards

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/acme/raw-demos")
    .map_async(
        mdr.robotics.reward_score(
            model="robometer/Robometer-4B",
            video_key="observation.images.top",
            task=lambda row: "; ".join(row.tasks),
            max_frames=8,
        ),
        max_in_flight=32,
    )
    .write_lerobot("hf://buckets/acme-robotics/demos-with-rewards")
)
```

See [Reward Scoring](../episode-operations/reward-scoring.md).
