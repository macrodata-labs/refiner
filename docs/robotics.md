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

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
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

`motion_trim(...)` assumes LeRobot episode rows:

- it expects a `LeRobotRow`
- it trims the episode frame table directly
- it updates video timestamps on the row itself
- when a video span changes, the corresponding `stats/<video_key>/...` fields are dropped so the writer recomputes them later

## LeRobot Performance Notes

Current LeRobot output is optimized for:

- incremental frame parquet writes
- asynchronous per-episode video preparation
- cheap metadata reduction after shard-local stage-1 output

Key decisions:

- remux is preferred when source packets and boundaries are compatible
- transcode is used when compatibility or stats recomputation requires decoded frames
- `max_video_prepare_in_flight` bounds concurrent episode video work inside one worker
- `transencoding_threads` is treated as a worker budget and divided across simultaneous video streams in the same row

The practical consequence is:

- frame-heavy no-video datasets mostly behave like a parquet writer
- video-heavy datasets are dominated by source probing, remux/transcode work, and the quality of source clip alignment

## Notes

- robotics is the first modality with deeper built-in support today
- more multimodal primitives and modality-specific building blocks are expected to grow from here

## Related Pages

- [Readers and sharding](readers-and-sharding.md)
- [Launchers](launchers.md)
- [Observability](observability.md)
