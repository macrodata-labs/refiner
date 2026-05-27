---
title: "Quickstart"
description: "Run a complete robotics data pipeline with Refiner"
---

# Quickstart

This quickstart reads a LeRobot dataset, trims inactive frames, writes a new
LeRobot dataset, and shows how to run the same pipeline locally or on Macrodata
Cloud.

## Install

```bash
pip install macrodata-refiner[robotics]
```

For cloud runs, authenticate once:

```bash
macrodata login
```

You can also create an API key in Macrodata and set `MACRODATA_API_KEY`; see
[API Keys and Auth](platform/workspaces-and-api-keys.md).

## A Small Pipeline

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(mdr.robotics.motion_trim(threshold=0.001))
    .write_lerobot("hf://buckets/acme-robotics/aloha_static_trimmed")
)
```

This pipeline has three parts:

| Part | What it does |
| --- | --- |
| `read_lerobot(...)` | Reads one episode row at a time, with frame data and video references attached. |
| `.map(...)` | Applies a Python function to each episode. |
| `.write_lerobot(...)` | Writes episodes, frame parquet files, videos, metadata, tasks, and stats. |

Refiner pipelines are immutable. Every method returns a new pipeline value, so
you can build variants without mutating the original.

## Inspect Locally

Before launching a job, inspect a few rows in process:

```python
row = pipeline.take(1)[0]

print(row.episode_id)
print(row.num_frames)
print(list(row.videos))
```

`take()` executes lazily and stops after the requested number of rows. See
[In-Process Debugging](running-pipelines/in-process-debugging.md) for more
inspection patterns.

## Run Locally

```python
pipeline.launch_local(
    name="aloha-trim-local",
    num_workers=2,
)
```

Local launch runs worker processes on your machine. Use it when you want the
same shard and worker behavior as a launched job without using cloud resources.

## Run On Macrodata Cloud

```python
pipeline.launch_cloud(
    name="aloha-trim",
    num_workers=8,
    secrets={"HF_TOKEN": None},
)
```

`None` means "read this value from my local environment at submission time".
If you store secrets in the platform, use `mdr.Secrets.env(...)`; see
[Secrets and Environment](platform/secrets-and-environment.md).

## Where To Go Next

- Learn the execution options in [Running Pipelines](running-pipelines/index.md).
- Learn readers in [Reading Data](reading-data/index.md).
- Learn the episode model in [Episode Data](episode-data/index.md).
- Learn common row operations in [Transforms](transforms/index.md).
- Learn LeRobot output details in [Writing LeRobot](writing-data/lerobot.md).

