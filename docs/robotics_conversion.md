# Robotics Conversion Examples

## ALOHA / ACT HDF5

ALOHA-style HDF5 datasets are usually one HDF5 file per episode, with
frame-aligned arrays under fixed paths:

- `/action`
- `/observations/qpos`
- `/observations/qvel`
- `/observations/images/<camera>`

```python
import refiner as mdr

pipeline = (
    mdr.read_hdf5(
        "/data/aloha/*.hdf5",
        groups="/",
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

## robomimic HDF5

robomimic-style HDF5 datasets usually store many demonstrations in one file.
Each demo group is one episode:

- `/data/demo_*/actions`
- `/data/demo_*/obs/<field>`
- `/data/demo_*/rewards`
- `/data/demo_*/dones`

`read_hdf5(groups="/data/demo_*")` emits one row per matched demo group, and
`datasets=` paths are relative to that demo group.

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
        attrs={"task": "task"},
        group_path_column="episode_id",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        task_key="task",
        action_key="actions",
        state_key=("joint_pos", "joint_vel", "eef_pos", "gripper_qpos"),
        video_keys=("agentview", "wrist"),
    )
)
```
