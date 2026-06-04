---
title: "HDF5 Reader"
description: "Read HDF5 robotics datasets and convert them into episode rows"
---

# HDF5 Reader

Use `read_hdf5` for robotics datasets stored as HDF5 files, including ALOHA-like
files and robomimic-style grouped demonstrations.

## One File Per Episode

ALOHA-style data often stores one episode per HDF5 file:

```python
import refiner as mdr

pipeline = (
    mdr.read_hdf5(
        "/data/aloha/*.hdf5",
        datasets={
            "action": "action",
            "qpos": "observations/qpos",
            "qvel": "observations/qvel",
            "cam_high": "observations/images/cam_high",
            "cam_left_wrist": "observations/images/cam_left_wrist",
        },
        file_path_column="episode_file",
    )
    .to_robot_rows(
        episode_id_key="episode_file",
        action_key="action",
        state_key="qpos",
        extra_observation_keys=("qvel",),
        video_keys=("cam_high", "cam_left_wrist"),
    )
)
```

`datasets` maps output column names to HDF5 dataset paths relative to the
matched group.

## Grouped Demonstrations

robomimic-style files often store many demos in one file:

```python
pipeline = (
    mdr.read_hdf5(
        "/data/robomimic.hdf5",
        groups="/data/demo_*",
        datasets={
            "actions": "actions",
            "joint_pos": "obs/robot0_joint_pos",
            "joint_vel": "obs/robot0_joint_vel",
            "agentview": "obs/agentview_image",
        },
        group_path_column="episode_id",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        action_key="actions",
        state_key=("joint_pos", "joint_vel"),
        video_keys=("agentview",),
    )
)
```

`groups` can be a single glob string or exact group paths.

## Missing Data Policy

```python
pipeline = mdr.read_hdf5(
    "/data/*.hdf5",
    datasets={"action": "action", "state": "state"},
    missing_policy="drop_row",
)
```

| Policy | Behavior |
| --- | --- |
| `"error"` | Raise when a selected dataset or attribute is missing. |
| `"drop_row"` | Skip rows with missing selected values. |
| `"set_null"` | Emit `None` for missing selected values. |

## Related Pages

- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
- [HDF5 Conversion Example](../examples/formats/aloha-hdf5.md)
- [Zarr Reader](zarr.md)

