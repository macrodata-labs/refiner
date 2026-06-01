---
title: "Libero HDF5"
description: "Convert the LIBERO HDF5 eval datasets to LeRobot"
---

# Libero HDF5

Convert all four LIBERO HDF5 eval subsets to LeRobot on cloud workers:

```python
from datetime import datetime, timezone

import refiner as mdr

dataset_root = "hf://datasets/yifengzhu-hf/LIBERO-datasets"
output_prefix = "hf://buckets/acme-robotics/libero-hdf5"
eval_suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
fps = 10.0


stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output = f"{output_prefix}/{stamp}-full-eval"

(
    mdr.read_hdf5(
        [f"{dataset_root}/{suite}" for suite in eval_suites],
        groups="/data/demo_*",
        datasets={
            "raw_action": "actions",
            "observation.images.image": "obs/agentview_rgb",
            "observation.images.wrist_image": "obs/eye_in_hand_rgb",
            "ee_state": "obs/ee_states",
            "gripper_state": "obs/gripper_states",
        },
        file_path_column="file_path",
        group_path_column="hdf5_group",
        cache_remote_files=True,
    )
    .map(
        lambda row: row.update(
            task=str(row["file_path"])
            .rsplit("/", 1)[-1]
            .removesuffix(".hdf5")
            .removesuffix("_demo")
            .replace("_", " ")
        )
    )
    .to_robot_rows(
        task_key="task",
        fps=fps,
        robot_type="libero",
        action_key="raw_action",
        state_key=("ee_state", "gripper_state"),
        video_keys={
            "observation.images.image": "observation.images.image",
            "observation.images.wrist_image": "observation.images.wrist_image",
        },
    )
    .write_lerobot(output, max_video_prepare_in_flight=2)
    .launch_cloud(
        name="libero-hdf5-full-eval",
        num_workers=40,
        cpus_per_worker=1,
        mem_mb_per_worker=1024,
        extra_dependencies=("av", "h5py", "huggingface-hub>=1.4.1", "pillow"),
        secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
    )
)
```

Related: [HDF5 Reader](../reading-data/hdf5.md),
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
