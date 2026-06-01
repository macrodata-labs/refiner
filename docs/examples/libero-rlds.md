---
title: "Libero RLDS"
description: "Convert a TFDS/RLDS Libero dataset to LeRobot or Zarr"
---

# Libero RLDS

Libero-style RLDS datasets store each episode as a TFDS `steps` dataset. Keep
state and actions in `steps`, and lift camera streams into videos.

```python
from huggingface_hub import snapshot_download

import refiner as mdr

dataset_root = snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    allow_patterns="libero_10_no_noops/1.0.0/*",
)
dataset_dir = f"{dataset_root}/libero_10_no_noops/1.0.0"

pipeline = (
    mdr.read_tfds(
        dataset_dir,
        videos={
            "front": "steps/observation/image",
            "wrist": "steps/observation/wrist_image",
        },
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="steps",
        action_key="action",
        state_key="observation/state",
        video_keys={
            "observation.images.front": "videos/front",
            "observation.images.wrist": "videos/wrist",
        },
        fps=30,
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
