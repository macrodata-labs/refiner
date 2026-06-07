---
title: "ALOHA HDF5"
description: "Convert ALOHA-style HDF5 episodes into LeRobot"
---

# ALOHA HDF5

ALOHA-style datasets often store one episode per HDF5 file.

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
        fps=50.0,
        robot_type="aloha",
    )
    .write_lerobot("hf://buckets/acme-robotics/aloha-converted")
)
```

Run a small local test first:

```python
print(pipeline.take(1)[0].num_frames)
```

Then launch:

```python
pipeline.launch_cloud(
    name="convert-aloha",
    num_workers=8,
    secrets={"HF_TOKEN": None},
)
```

Related: [HDF5 Reader](../../reading-data/hdf5.md),
[Converting to Robot Rows](../../episode-data/converting-to-robot-rows.md).
