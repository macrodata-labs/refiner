---
title: "Hand Tracking"
description: "Run batched hand tracking on robotics episode videos"
---

# Hand Tracking

`track_hands` runs ego-vision hand tracking over episode videos and writes a
side-keyed hand-tracking payload back to each row. The current default settings
were tuned against the HOT3D hand-tracking benchmark, so they are a strong
starting point for egocentric human videos but should still be validated on a
new camera domain.

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
        mdr.robotics.track_hands(
            video_key="video",
            output_key="hand_tracking",
        ),
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

## How Ego-Vision Works

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

pipeline = pipeline.batch_map(
    mdr.robotics.track_hands(
        video_key="video",
        output_key="hand_tracking",
        config=config,
    ),
    batch_size=2,
)
```

The important ego-vision defaults are:

```python
egovision.HandTrackingConfig(
    hand_reconstruction=egovision.HaworReconstructionConfig(
        checkpoint="hf://macrodata/egovision-safetensors/hawor.safetensors",
        detector_checkpoint="hf://macrodata/egovision-safetensors/yolo.safetensors",
        mano_right="hf://macrodata/egovision-safetensors/MANO_RIGHT.safetensors",
        mano_left="hf://macrodata/egovision-safetensors/MANO_LEFT.safetensors",
        chunk_size=16,
        batch_size=64,
        detector_batch_size=None,
        detector_conf=0.2,
        detector_mode="vendored_track",
        batch_across_episodes=True,
        compile_mode=None,
    ),
    camera_pose_estimator=egovision.VggtOmegaConfig(
        checkpoint="hf://macrodata/egovision-safetensors/vggt_omega_1b_512.safetensors",
        batch_size=1,
        image_resolution=512,
        preprocess_backend="torch",
        seq_length=300,
        camera_sample_fps=None,
        compile_mode=None,
    ),
    hand_fusion=egovision.HandFusionConfig(
        infiller_checkpoint="hf://macrodata/egovision-safetensors/infiller.safetensors",
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
| `batch_size` | `Pipeline.batch_map(...)` batch size. This controls how many episodes Refiner passes to one model call. |
| `gpu` | Cloud/local worker GPU. The current path is intended for one H100. |
| `mem_mb_per_worker` | Worker memory. Use at least `32768` MB for the default H100 hand-tracking stack. |

## Output Payload

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

For a complete cloud example that writes LeRobot actions, see
[`examples/hand_tracking_lerobot.py`](../../examples/hand_tracking_lerobot.py).

## Throughput

On one H100, the current default stack runs around 10 fps for ego-vision
prediction after model initialization on the first HomER episode we tested
(`4,834` frames). The full cloud stage, including model initialization, video
read/write, LeRobot conversion, and action conversion, was about 6-7 fps. Exact
throughput depends on clip length, resolution, worker cold-start cost, whether
model compile/cuda graph cost has been amortized, and how much video writing the
pipeline performs.

## Related Pages

- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
- [Files and Videos](../reading-data/files-and-videos.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
