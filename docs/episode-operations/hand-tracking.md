---
title: "Hand tracking"
description: "Runs ego-centric hand tracking over inputed videos"
---

# Hand tracking

`track_hands` relies on the `ego-vision` package to run ego-centric hand
tracking over episode videos, with settings optimized for the HOT3D
hand-tracking benchmark. It writes a side-keyed hand-tracking payload back to
each row.

The HomER example below writes one JSONL row per input video with raw
hand-tracking annotations plus MANO and joint action conversions.

```python
def hand_tracking_annotation(row):
    return {
        "video_id": row["video_id"],
        "description": row["description"],
        "hand_tracking": row["hand_tracking"],
    }


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
        mdr.robotics.track_hands(
            video_key="video",
            output_key="hand_tracking",
        ),
        batch_size=2,
    )
    .map(hand_tracking_annotation)
    .write_jsonl("hf://buckets/acme-robotics/homer-hand-tracking")
    .launch_cloud(
        name="hand-tracking",
        num_workers=1,
        mem_mb_per_worker=32 * 1024,
        gpu=mdr.GPU(count=1, type="h100"),
        secrets=mdr.Secrets.env(keys=("HF_TOKEN",)),
    )
)
```

## Requirements

Install the hand-tracking extra. If you use the `read_hf_dataset(...)` example
on this page locally, install `datasets` too:

```bash
pip install macrodata-refiner[datasets,hand_tracking]
```

Input rows must implement `RoboticsRow` and include the required `video_key` in
`row.videos`. Pass `video_key` explicitly to `track_hands`.

## How Ego-Vision works

The default ego-vision stack is:

1. Decode the episode video and pass frames to ego-vision.
2. Run VGGT-Omega over the clip to estimate camera trajectory and intrinsics.
3. Run HaWoR hand detection, tracking, and MANO hand reconstruction.
4. Fuse camera-space hand predictions with the VGGT camera trajectory to produce
   world-space wrists and joints.
5. Run the configured hand cleanup/postprocessing, including the default
   infiller, and return dense per-frame arrays.

The infiller is a temporal hand-motion model used after HaWoR reconstruction. It
looks at the surrounding valid hand predictions and fills short gaps where the
detector/reconstructor missed a hand or produced an unusable frame. Filled
frames are marked in each hand's `infilled` mask, so downstream training code can
keep them, drop them, or weight them differently. The infiller does not create
new visual evidence; it makes the returned trajectory dense and smoother across
brief detection gaps.

VGGT is a sequence model. The default `vggt_seq_length` is `300`, which matches
the current maximum 10 second, 30 fps episode target. When longer videos are
chunked, frames at chunk boundaries can be less stable. Refiner does not
currently mask, smooth, or otherwise special-case those boundary frames. If you
train action models on chunked output, consider masking action deltas at VGGT
chunk boundaries, or using downstream smoothing that explicitly handles
boundaries.

HaWoR and the infiller/postprocessing path also use temporal context and windows.
As with VGGT, Refiner does not apply extra boundary-specific handling beyond
ego-vision's configured model and postprocessing behavior.

## Configuration

`track_hands` accepts an optional `egovision.HandTrackingConfig`. If omitted,
Refiner constructs the default ego-vision pipeline:

```python
import egovision

config = egovision.HandTrackingConfig(
    hand_reconstruction=egovision.HaworReconstructionConfig(),
)


def hand_tracking_annotation(row):
    return {
        "video_id": row["video_id"],
        "description": row["description"],
        "hand_tracking": row["hand_tracking"],
    }


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
        mdr.robotics.track_hands(
            video_key="video",
            output_key="hand_tracking",
            config=config,
        ),
        batch_size=2,
    )
    .map(hand_tracking_annotation)
    .write_jsonl("hf://buckets/acme-robotics/homer-hand-tracking")
    .launch_cloud(
        name="hand-tracking",
        num_workers=1,
        mem_mb_per_worker=32 * 1024,
        gpu=mdr.GPU(count=1, type="h100"),
        secrets=mdr.Secrets.env(keys=("HF_TOKEN",)),
    )
)
```

The important ego-vision defaults are:

```python
egovision.HandTrackingConfig(
    hand_reconstruction=egovision.HaworReconstructionConfig(
        chunk_size=16,
        batch_size=64,
        detector_batch_size=None,
        detector_conf=0.2,
        detector_mode="vendored_track",
        batch_across_episodes=True,
        compile_mode=None,
    ),
    camera_pose_estimator=egovision.VggtOmegaConfig(
        batch_size=1,
        image_resolution=512,
        preprocess_backend="torch",
        seq_length=300,
        camera_sample_fps=None,
        compile_mode=None,
    ),
    hand_fusion=egovision.HandFusionConfig(
        infiller_batch_size=64,
        translation_window=7,
        translation_alpha=0.675,
        translation_accel_weight=None,
        translation_jerk_weight=0.0,
        rotation_window=9,
        rotation_gamma=0.6,
        joint_window=11,
        joint_alpha=1.0,
        wrist_rotation_source="model",
        missing_frame_strategy="none",
        mano_pose_window=11,
        mano_pose_gamma=0.8,
        mano_shape_strategy="median",
    ),
    device="cuda",
    decode_backend="memmap",
    decode_chunk_size=128,
)
```

In Refiner, configure the operation itself with:

| Refiner parameter | Meaning |
| --- | --- |
| `video_key` | Required key in `row.videos` to process. |
| `output_key` | Output column for the ego-vision payload. Defaults to `hand_tracking`. |
| `config` | Optional `egovision.HandTrackingConfig`. Defaults to the HOT3D-tuned ego-vision stack above. |

## Output payload

By default, the operation writes one episode-level column:

| Column | Meaning |
| --- | --- |
| `hand_tracking` | Ego-vision hand-tracking result dictionary. |

Customize the column name with `output_key`.

The payload is a dictionary of NumPy arrays and scalars. It is dense over the
input timeline: both `left` and `right` hand entries are present, and arrays have
the same frame count `T` as the returned camera trajectory. Missing hand
predictions are represented with `NaN` geometry, `confidence == 0`, and
`infilled == False` unless the infiller filled that frame.

The top-level payload contains:

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

`hands_camera` is useful for image-space visualization and debugging.
`hands_world` is the usual source for action labels because it includes head/camera
motion through the VGGT world trajectory.

## Action conversion

The raw payload can be converted into training actions with ego-vision helpers:

```python
from egovision.pipelines import to_joint_actions, to_mano_actions

mano_actions = to_mano_actions(row["hand_tracking"])
joint_actions = to_joint_actions(row["hand_tracking"])
```

`to_mano_actions(...)` returns side-keyed actions for `left` and `right` hands:

| Field | Shape per side | Meaning |
| --- | --- | --- |
| `delta_T_world_wrist` | `[T - 1, 4, 4]` | Relative wrist transform in world coordinates. |
| `delta_mano_pose` | `[T - 1, 96]` | Relative MANO pose action in the model's 6D rotation representation. |
| `valid` | `[T - 1]` | Whether the action delta is usable. |

`to_joint_actions(...)` returns:

| Field | Shape per side | Meaning |
| --- | --- | --- |
| `delta_joints_world` | `[T - 1, 21, 3]` | Per-joint world-space deltas. |
| `valid` | `[T - 1]` | Whether the action delta is usable. |

MANO actions are compact relative to dense joints but still high-dimensional:
`delta_T_world_wrist` is 16 numbers and `delta_mano_pose` is 96 numbers per
hand. Joint actions are 63 numbers per hand. If you need a smaller control
space, fit PCA or another learned bottleneck over the action vectors after
filtering invalid frames.

For a complete cloud example that writes hand-tracking annotations and actions, see
[`examples/hand_tracking.py`](../../examples/hand_tracking.py).

## Throughput

On one H100, the current default stack runs around 10 fps for ego-vision
prediction. The full cloud stage, including model initialization, video
read/write, LeRobot conversion, and action conversion, was about 6-7 fps. You
can increase throughput by setting `EgoVisionConfig.vggt.camera_sample_fps` to
run VGGT on a lower frame rate and interpolate the camera trajectory back to the
full timeline.

## Related pages

- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
- [Files and Videos](../reading-data/files-and-videos.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
