---
title: "Hand Tracking"
description: "Run batched hand tracking on robotics episode videos"
---

# Hand Tracking

`track_hands` returns a `batch_map` function that runs an ego-vision hand
tracking pipeline over episode videos.

```python
pipeline = (
    mdr.read_videos("/data/ego/*.mp4", file_path_column="video")
    .to_robot_rows(video_keys=("video",), fps=30.0)
    .batch_map(
        mdr.robotics.track_hands(video_key="video", output_key="hand_tracking"),
        batch_size=8,
    )
)
```

## Requirements

Install the hand-tracking extra:

```bash
pip install macrodata-refiner[hand_tracking]
```

Input rows must implement `RoboticsRow` and include the selected `video_key` in
`row.videos`.

## Output

The output key contains the hand-tracking result converted to a dictionary:

```python
row["hand_tracking"]
```

The exact shape is determined by the ego-vision package configuration.

## Related Pages

- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
- [Files and Videos](../reading-data/files-and-videos.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)

