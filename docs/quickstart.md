---
title: "Quickstart"
description: "Run a complete robotics data pipeline with Refiner"
---

# Quickstart

Refiner is an open-source library for building robotics data pipelines. A
[pipeline](running-pipelines/index.md) describes how to
[read data](reading-data/index.md),
[transform rows and episodes](transforms/index.md), and
[write the result](writing-data/index.md). You can
[process a wide range of formats](reading-data/index.md) out of the box, use
[models](inference/index.md) for labeling and scoring, and
[inspect pipelines locally](running-pipelines/in-process-debugging.md) while you
develop. When you do not want to manage infrastructure, run the same code with
[local workers](running-pipelines/local-launcher.md) or submit it to
the [Macrodata Cloud](running-pipelines/cloud-launcher.md).

Readers and writers are [sharded](reading-data/sharding.md), so most pipelines
do not need to download or materialize the entire dataset before doing useful
work. Refiner streams shards through the pipeline and writes outputs as they are
produced.

## Install

Install the Refiner package in the Python environment where you want to build or
run the pipeline:

```bash
pip install macrodata-refiner[hf,video]
```

The `hf` and `video` extras are optional, but the example below uses them for
[Hugging Face paths](reading-data/hugging-face.md) and
[video data](episode-data/frames-and-videos.md). To install every optional
dependency, use `pip install macrodata-refiner[all]`.

[Create an account](/auth/register), then authenticate once with the
[Macrodata CLI](cli/auth-and-run.md). This is optional for local development,
but it lets you keep track of local runs in your workspace. The same
credentials will also be used to submit cloud runs:

```bash
macrodata login
```

The CLI stores an [API key](platform/workspaces-and-api-keys.md) for you. You
can also set one directly with the `MACRODATA_API_KEY` environment variable.

## Example

Most Refiner pipelines follow the same shape:

```text
read data  →  transform  →  write result
```

For example:

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(lambda row: row.update(task="battery insertion"))
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_static_with_task")
)
```

This example reads a small
[Hugging Face dataset](reading-data/hugging-face.md), adds a task label to each
episode, and writes the result back to a Hugging Face bucket.

Each step returns a new pipeline value:

| Step | What it does |
| --- | --- |
| `read_lerobot(...)` | Reads robotics episodes with the [LeRobot reader](reading-data/lerobot.md). |
| `.map(...)` | Updates each [row or episode](episode-data/episode-rows.md) with a task label. |
| `.write_lerobot(...)` | Writes the transformed dataset with the [LeRobot writer](writing-data/lerobot.md). |

Nothing runs when you create the pipeline. Refiner executes it only when you
inspect rows with methods like `take()`, launch
[local workers](running-pipelines/local-launcher.md), or submit a
[cloud job](running-pipelines/cloud-launcher.md).

## Inspect a pipeline

Start by inspecting a small amount of data
[in process](running-pipelines/in-process-debugging.md). This block is
self-contained and inspects the pipeline before the writer:

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(lambda row: row.update(task="battery insertion"))
)

row = pipeline.take(1)

print(row[0])
```

Output:

```text
LeRobotRow
  episode_id: '0'
  num_frames: 600
  task: 'battery insertion'
  fps: 50
  robot_type: 'unknown'
  frame_data (row.to_frame_table()):
    actions (row.actions): float[600, 14]
    states (row.states): float[600, 14]
    timestamps (row.timestamps): float[600]
  videos (row.videos):
    observation.images.cam_high: video[480, 640, 3]@50fps
    observation.images.cam_left_wrist: video[480, 640, 3]@50fps
    observation.images.cam_low: video[480, 640, 3]@50fps
    observation.images.cam_right_wrist: video[480, 640, 3]@50fps
  stats: ['action', 'episode_index', 'frame_index', 'index', 'next.done', 'observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_low', '... +4 more']
```

`take()` executes lazily and stops after the requested number of rows, so it is
the fastest way to check [schemas](transforms/schemas-and-dtypes.md),
[media references](episode-data/frames-and-videos.md), and transform outputs
before launching a full job.

## Run locally

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(lambda row: row.update(task="battery insertion"))
    .write_lerobot("./aloha_static_with_task", max_video_prepare_in_flight=4)
)

pipeline.launch_local(
    name="aloha-task-local",
    num_workers=1,
)
```

[Local launch](running-pipelines/local-launcher.md) runs worker processes on
your machine. Use it when you want the same
[shard](reading-data/sharding.md) and worker behavior as a launched job without
using cloud resources. If you are [logged in](cli/auth-and-run.md), local runs
are also tracked in the platform interface.

## Run on the Macrodata Cloud

This example converts the public LIBERO spatial HDF5 subset to LeRobot using
cloud workers. It reads one demo group per row, derives the task label from the
filename, turns action/state/image arrays into robotics episodes, encodes the
two camera streams as videos, and writes a LeRobot dataset to your output
bucket.

```python
import refiner as mdr

dataset = "hf://datasets/yifengzhu-hf/LIBERO-datasets/libero_spatial"
# Replace this with your output bucket (HF, S3, GCP, etc).
output = "hf://buckets/macrodata/test_bucket/libero-spatial"

(
    mdr.read_hdf5(
        dataset,
        groups="/data/demo_*",
        datasets={
            "raw_action": "actions",
            "observation.images.image": "obs/agentview_rgb",
            "observation.images.wrist_image": "obs/eye_in_hand_rgb",
            "ee_state": "obs/ee_states",
            "gripper_state": "obs/gripper_states",
        },
        file_path_column="file_path",
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
        fps=10.0,
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
        name="libero-spatial-subset",
        num_workers=10,
        cpus_per_worker=1,
        mem_mb_per_worker=1024,
        # Replace this with a token that can write the output.
        secrets=mdr.Secrets.dict({"HF_TOKEN": "---"}),
    )
)
```

The input dataset is public, but the cloud workers need `HF_TOKEN` to write back
to your Hugging Face bucket. You can also safely store reusable secrets directly
on the platform and reference them with `mdr.Secrets.env(...)`. See
[Secrets and environment](platform/secrets-and-environment.md).

After submission, follow the run from [Jobs](/jobs). Once scheduled, this
example should only take a couple of minutes. The job page shows live status,
worker progress, logs, metrics, resource usage, and output links while the
conversion runs. You can inspect the same run from the terminal with the
[`macrodata jobs` CLI](cli/jobs-logs-and-metrics.md). Cloud jobs are billed for
the compute they actually use; see [Billing](platform/billing.md) or
[pricing](/pricing).

For the full four-suite LIBERO conversion, see
[Libero HDF5](examples/formats/libero-hdf5.md).

## Where to go next

- Learn the execution options in [Running Pipelines](running-pipelines/index.md).
- Learn readers in [Reading Data](reading-data/index.md).
- Learn the episode model in [Episode Data](episode-data/index.md).
- Learn common row operations in [Transforms](transforms/index.md).
- Learn LeRobot output details in [Writing LeRobot](writing-data/lerobot.md).
