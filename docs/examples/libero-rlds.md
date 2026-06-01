---
title: "Libero RLDS"
description: "Convert a TFDS/RLDS Libero dataset to LeRobot or Zarr"
---

# Libero RLDS

Libero-style RLDS datasets store each episode as a TFDS `steps` dataset. Keep
state and actions in `steps`, and lift camera streams into videos.

```python
from datetime import datetime, timezone

import refiner as mdr

raw_datasets = [
    "hf://datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_goal_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_object_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
]
output_prefix = "hf://buckets/acme-robotics/libero-rlds"
fps = 10.0

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output = f"{output_prefix}/{stamp}-full-eval"

(
    mdr.read_tfds(
        raw_datasets,
        videos={
            "front": "steps/observation/image",
            "wrist": "steps/observation/wrist_image",
        },
        fps=fps,
    )
    .to_robot_rows(
        nested_frames_key="steps",
        episode_id_key="episode_metadata/file_path",
        task_key="steps/language_instruction",
        action_key="action",
        state_key="observation/state",
        video_keys={
            "observation.images.front": "videos/front",
            "observation.images.wrist": "videos/wrist",
        },
        fps=fps,
        robot_type="libero",
    )
    .write_lerobot(output, max_video_prepare_in_flight=2)
    .launch_cloud(
        name="libero-rlds-full-eval",
        num_workers=40,
        cpus_per_worker=1,
        mem_mb_per_worker=1024,
        extra_dependencies=(
            "av",
            "huggingface-hub>=1.4.1",
            "pillow",
            "tensorflow",
            "tensorflow-datasets",
        ),
        secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
    )
)
```

Related: [TensorFlow Reader](../reading-data/tensorflow.md),
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
