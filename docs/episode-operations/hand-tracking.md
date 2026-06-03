---
title: "Hand Tracking"
description: "Run batched hand tracking on robotics episode videos"
---

# Hand Tracking

`track_hands` runs ego-vision hand tracking over episode videos and writes a
side-keyed hand-tracking payload back to each row.

As of right now, Refiner supports ego-vision's HaWoR-based hand tracking
pipeline for human egocentric videos. Inputs are robotics rows with a video
source; URL strings, local files, and video assets can be converted with
`to_robot_rows(video_keys=...)`.

```python
pipeline = (
    mdr.read_hf_dataset(
        "toloka/HomER",
        split="train",
        columns_to_read=("video_id", "video_url", "description"),
        dtypes={"video_url": mdr.datatype.video_path()},
    )
    .to_robot_rows(
        episode_id_key="video_id",
        task_key="description",
        fps=30.0,
        robot_type="human_hand_tracking",
        video_keys={"video": "video_url"},
    )
    .batch_map(
        mdr.robotics.track_hands(video_key="video", output_key="hand_tracking"),
        batch_size=2,
    )
    .write_lerobot("hf://buckets/acme-robotics/homer-hand-tracking")
)
```

## Requirements

Install the hand-tracking extra:

```bash
pip install macrodata-refiner[hand_tracking]
```

Input rows must implement `RoboticsRow` and include the required `video_key` in
`row.videos`. Pass `video_key` explicitly to `track_hands`.

## Output Payload

By default, the operation writes one episode-level column:

| Column | Meaning |
| --- | --- |
| `hand_tracking` | Ego-vision hand-tracking result dictionary. |

Customize the column name with `output_key`.

The payload contains:

| Field | Meaning |
| --- | --- |
| `episode_id` | Ego-vision episode identifier. |
| `camera_trajectory` | Per-frame world-from-camera transforms, shaped `[T, 4, 4]`. |
| `intrinsics` | Per-frame camera intrinsics, usually `[T, 3, 3]`. |
| `hands_camera` | Dict keyed by `left` and `right` with camera-space hand tracks. |
| `hands_world` | Dict keyed by `left` and `right` with world-space hand tracks. |
| `metadata` | Pipeline settings such as `vggt_seq_length` and `hawor_seq_length`. |

Each hand entry contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `joints_camera` | `[T, 21, 3]` | Camera-space hand joints. |
| `joints_world` | `[T, 21, 3]` | World-space hand joints. |
| `T_camera_wrist` | `[T, 4, 4]` | Camera-space wrist transform. |
| `T_world_wrist` | `[T, 4, 4]` | World-space wrist transform. |
| `mano_pose` | `[T, 96]` | HaWoR/HMR2 MANO pose in 6D rotation format. |
| `mano_shape` | `[T, 10]` | MANO shape/betas. |
| `mano_translation` | `[T, 3]` | Camera-space MANO translation, when available. |
| `confidence` | `[T]` | Per-frame hand confidence. |
| `infilled` | `[T]` | Boolean mask for frames filled by the infiller. |

## Metrics

The operation logs:

| Metric | Meaning |
| --- | --- |
| `egovision_frames_decoded` | Frames decoded and handed to ego-vision. |
| `frames_processed` | Hand-tracking frames emitted by ego-vision. |

## Action Conversion

The raw payload can be converted into training actions with ego-vision helpers:

```python
from egovision.pipelines import to_joint_actions, to_mano_actions

mano_actions = to_mano_actions(row["hand_tracking"])
joint_actions = to_joint_actions(row["hand_tracking"])
```

For a complete cloud example that writes LeRobot actions, see
[`examples/hand_tracking_lerobot.py`](../../examples/hand_tracking_lerobot.py).

## Related Pages

- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
- [Files and Videos](../reading-data/files-and-videos.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
