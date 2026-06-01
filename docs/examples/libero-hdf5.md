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

## How It Works

### Inputs

`eval_suites` points at the four LIBERO evaluation subsets under the Hugging Face
dataset root. Passing a list of folders to `read_hdf5(...)` lets Refiner shard
the conversion across files from all four subsets.

### HDF5 Layout

Each LIBERO file stores demonstrations under `/data/demo_*`. The reader emits
one row per matched demo group and loads the arrays needed for LeRobot:

| Output column | HDF5 dataset |
| --- | --- |
| `raw_action` | `actions` |
| `observation.images.image` | `obs/agentview_rgb` |
| `observation.images.wrist_image` | `obs/eye_in_hand_rgb` |
| `ee_state` | `obs/ee_states` |
| `gripper_state` | `obs/gripper_states` |

`file_path_column` and `group_path_column` keep the original file and demo group
on the row. `cache_remote_files=True` downloads each remote HDF5 file to worker
local storage while it is being read, which avoids repeated random reads against
the remote filesystem.

### Task Labels

LIBERO task text is encoded in the HDF5 filename. The `.map(...)` step derives a
readable task label by removing `.hdf5` and `_demo`, then replacing underscores
with spaces.

### Robot Rows

`to_robot_rows(...)` turns each HDF5 demo row into one robotics episode:

| Argument | Meaning |
| --- | --- |
| `task_key="task"` | Uses the derived task label. |
| `fps=fps` | Supplies the fixed LIBERO frame rate. |
| `robot_type="libero"` | Records the robot/dataset family in LeRobot metadata. |
| `action_key="raw_action"` | Uses the HDF5 action trajectory as frame actions. |
| `state_key=("ee_state", "gripper_state")` | Concatenates end-effector and gripper state into `observation.state`. |
| `video_keys=...` | Treats the two RGB frame arrays as LeRobot video streams. |

### Writing And Running

`write_lerobot(...)` writes LeRobot parquet metadata plus encoded videos. The
example bounds per-worker video preparation with `max_video_prepare_in_flight=2`
to keep memory use predictable while two camera streams are encoded.

`launch_cloud(...)` runs the conversion with 40 workers. `extra_dependencies`
installs the packages needed to read HDF5, access Hugging Face storage, and
encode videos. The `HF_TOKEN` secret is passed through so workers can read and
write Hugging Face paths.

Related: [HDF5 Reader](../reading-data/hdf5.md),
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
