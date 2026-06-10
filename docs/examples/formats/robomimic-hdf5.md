---
title: "robomimic HDF5"
description: "Convert robomimic-style grouped HDF5 demonstrations"
---

# robomimic HDF5

robomimic-style datasets often store many demonstrations in one file under
groups such as `/data/demo_*`.

```python
import refiner as mdr

pipeline = (
    mdr.read_hdf5(
        "/data/robomimic.hdf5",
        groups="/data/demo_*",
        datasets={
            "actions": "actions",
            "joint_pos": "obs/robot0_joint_pos",
            "joint_vel": "obs/robot0_joint_vel",
            "eef_pos": "obs/robot0_eef_pos",
            "gripper_qpos": "obs/robot0_gripper_qpos",
            "agentview": "obs/agentview_image",
            "wrist": "obs/robot0_eye_in_hand_image",
        },
        group_path_column="episode_id",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        action_key="actions",
        state_key=("joint_pos", "joint_vel", "eef_pos", "gripper_qpos"),
        video_keys=("agentview", "wrist"),
        fps=20.0,
    )
    .write_lerobot("hf://buckets/acme-robotics/robomimic-converted")
)
```

If some demos are missing optional fields, choose a `missing_policy` in
[`read_hdf5`](../../reading-data/hdf5.md).
