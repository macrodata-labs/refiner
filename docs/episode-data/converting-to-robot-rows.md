---
title: "Converting to Robot Rows"
description: "Map generic rows into Refiner's robotics episode view"
---

# Converting To Robot Rows

Use `to_robot_rows(...)` when a generic reader produces episode-shaped rows that
should be treated as robotics episodes.

## Basic Mapping

```python
pipeline = (
    mdr.read_hdf5(
        "/data/aloha/*.hdf5",
        datasets={
            "action": "action",
            "qpos": "observations/qpos",
            "front": "observations/images/front",
        },
        file_path_column="episode_file",
    )
    .to_robot_rows(
        episode_id_key="episode_file",
        action_key="action",
        state_key="qpos",
        video_keys=("front",),
        fps=30.0,
        robot_type="aloha",
    )
)
```

The input columns remain available by their original names. The robotics view
adds semantic properties such as `row.actions`, `row.states`, and `row.videos`.

## Key Mapping

| Option | Use it for |
| --- | --- |
| `episode_id_key` | Source column or slash path for episode identity. |
| `task_key` | Source column or slash path containing task text. Strings become `row.tasks = [task]`; string sequences are preserved. |
| `fps` or `fps_key` | Literal fps or source column for fps. |
| `robot_type` or `robot_type_key` | Literal robot type or source column. |
| `timestamp_key` | Frame-aligned timestamp column. If missing and `fps` is known, timestamps are generated from frame indices. |
| `action_key` | Frame-aligned action column. |
| `state_key` | One state column or several columns concatenated as state. |
| `extra_observation_keys` | Additional observations exposed by name. |
| `video_keys` | Video columns exposed through `row.videos`. |

## Multiple State Columns

```python
pipeline = source.to_robot_rows(
    action_key="actions",
    state_key=("joint_pos", "joint_vel", "eef_pos", "gripper_qpos"),
)
```

Tuple state keys are concatenated into `row.states` and exposed as
`observation.state`.

## Nested Frame Data

If a row contains a nested frame table, pass `nested_frames_key`:

```python
pipeline = source.to_robot_rows(
    nested_frames_key="frames",
    action_key="action",
    state_key="observation.state",
)
```

This is useful for custom readers that already group frames per episode.
For RLDS-style rows, `task_key` can point into the nested frame data, such as
`task_key="steps/language_instruction"`.

## Writing Converted Rows

Converted rows can be written as LeRobot:

```python
pipeline.write_lerobot("hf://buckets/acme-robotics/converted-dataset")
```

See [Writing LeRobot](../writing-data/lerobot.md).
