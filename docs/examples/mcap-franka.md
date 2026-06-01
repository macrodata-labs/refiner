---
title: "MCAP Franka"
description: "Convert MCAP robot logs with video into LeRobot and Zarr"
---

# MCAP Franka

MCAP logs often contain robot state, actions, and camera topics at different
rates. Use a camera topic as `sync_primary` when each output row should
correspond to one video frame.

```python
import refiner as mdr

pipeline = (
    mdr.read_mcap(
        "hf://datasets/SLAI-scientific-embodied-2026/"
        "Franka1_mcap_short_task1_0520/"
        "task101-bag_20260517_195933/"
        "task101-bag_20260517_195933_0.mcap",
        fields={
            "state": "/joint_states.position",
            "action": "/joint_states.velocity",
        },
        videos={"front": "/cam1/realsense_camera/color/image_raw/compressed"},
        sync_primary="front",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key="action",
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
        robot_type="franka",
    )
)
```

Write LeRobot:

```python
pipeline.write_lerobot("hf://buckets/acme-robotics/slai-franka-lerobot")
```

Write Zarr:

```python
pipeline.write_zarr("hf://buckets/acme-robotics/slai-franka.zarr")
```

For H.264 packet topics, select the packet topic as a video source. Messages
must include `format="h264"` and base64 or bytes `data`.

```python
pipeline = (
    mdr.read_mcap(
        "run.mcap",
        fields={},
        videos={"front": "/camera/h264"},
        sync_primary="front",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key=None,
        action_key=None,
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
    .write_lerobot("hf://buckets/acme-robotics/h264-video-dataset")
)
```

Related: [MCAP Reader](../reading-data/mcap.md),
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
