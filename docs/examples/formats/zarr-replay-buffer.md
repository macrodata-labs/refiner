---
title: "Zarr replay buffer"
description: "Convert Zarr replay buffers into robot episode rows"
---

# Zarr replay buffer

Replay buffers often store frame arrays and cumulative episode boundaries.

```python
import refiner as mdr

pipeline = (
    mdr.read_zarr(
        "hf://datasets/acme/robot-buffer/buffer.zarr",
        arrays={
            "action": "data/action",
            "eef_pos": "data/eef_pos",
            "joint_pos": "data/joint_pos",
            "wrist": "data/wrist_rgb",
        },
        row_ends="meta/episode_ends",
        index_column="episode_id",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        action_key="action",
        state_key=("eef_pos", "joint_pos"),
        video_keys=("wrist",),
        fps=20.0,
    )
    .write_lerobot("hf://buckets/acme-robotics/zarr-converted")
)
```

Related: [Zarr Reader](../../reading-data/zarr.md),
[Frames and Videos](../../episode-data/frames-and-videos.md).
