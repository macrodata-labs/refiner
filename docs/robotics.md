---
title: "Robotics"
description: "Current robotics support in Refiner"
---

Refiner already includes robotics-specific support through the LeRobot reader, writer, and robotics transforms.

## Current Support

- `read_lerobot(...)`
- `write_lerobot(...)`
- robotics transforms under `mdr.robotics.*`

Current workflows include:

- reading LeRobot datasets from local paths or remote `fsspec` paths
- transforming episode-level rows
- writing LeRobot-compatible output datasets
- multistage writer flows for episode data, metadata reduction, and dataset finalization

## Motion Trimming Example

```python
import refiner as mdr

MOTION_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_low",
    "observation.images.cam_right_wrist",
)

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(lambda row: row.drop(*MOTION_VIDEO_KEYS))
    .map(
        mdr.robotics.motion_trim(
            threshold=0.001,
            pad_frames=5,
        )
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_motion")
    .launch_cloud(
        name="motion_trim",
        num_workers=1,
    )
)
```

## Notes

- robotics is the first modality with deeper built-in support today
- more multimodal primitives and modality-specific building blocks are expected to grow from here

## Related Pages

- [Readers and sharding](readers-and-sharding.md)
- [Launchers](launchers.md)
