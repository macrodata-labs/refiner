---
title: "Libero RLDS"
description: "Convert a TFDS/RLDS Libero dataset to LeRobot or Zarr"
---

# Libero RLDS

Libero-style RLDS datasets store each episode as a TFDS `steps` dataset. Keep
state and actions in `steps`, and lift camera streams into videos.

```python
import refiner as mdr

pipeline = (
    mdr.read_tfds(
        "hf://datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0",
        videos={
            "front": "steps/observation/image",
            "wrist": "steps/observation/wrist_image",
        },
        fps=10,
    )
    .to_robot_rows(
        nested_frames_key="steps",
        task_key="steps/language_instruction",
        action_key="action",
        state_key="observation/state",
        video_keys={
            "observation.images.front": "videos/front",
            "observation.images.wrist": "videos/wrist",
        },
        fps=10,
        robot_type="libero",
    )
)
```

Write LeRobot:

```python
pipeline.write_lerobot("hf://buckets/acme-robotics/libero-10-lerobot")
```

Write Zarr:

```python
pipeline.write_zarr(
    "hf://buckets/acme-robotics/libero-10.zarr",
    arrays={
        "data/action": "action",
        "data/state": "observation.state",
        "data/front": "observation.images.front",
        "data/wrist": "observation.images.wrist",
    },
)
```

Related: [TensorFlow Reader](../reading-data/tensorflow.md),
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
