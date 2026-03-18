<p align="center">
  <img src="https://macrodata.co/logo.svg" alt="Macrodata" width="180">
</p>

<h1 align="center">Macrodata Refiner</h1>

Refiner is an open-source engine for turning raw, unstructured, and multimodal data into **high-quality datasets** for large model training.

It replaces the brittle scripts and stitched-together data tooling that teams still use for training data work, while offering much better support for multimodal data, robotics workflows, and model-based processing.

It also plugs into the Macrodata platform, which gives you visibility into what is happening to your data while pipelines run: job and shard lifecycle, logs, metrics, manifests, and pipeline behavior. The same code can run locally for development and then scale out through Macrodata's elastic serverless cloud.

## Quickstart

Install:

```bash
pip install macrodata-refiner
```

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

Log in:

```bash
macrodata login
```

Launch a robotics pipeline on Macrodata Cloud.

This requires a valid API key.

```python
import refiner as mdr

MOTION_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_low",
    "observation.images.cam_right_wrist",
)

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(lambda row: row.drop(*MOTION_VIDEO_KEYS))
    .map(
        mdr.robotics.motion_trim(
            threshold=0.001,
            pad_frames=5,
        )
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_motion")
    .launch_cloud(
        name="motion_trim",
        num_workers=1,
    )
)
```

Launch a local pipeline:

```python
import refiner as mdr

(
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .map(lambda row: {"text": row["text"], "text_len": row["text_len"], "source": "docs"})
    .write_parquet("out/")
    .launch_local(
        name="english-cleanup",
        num_workers=2,
    )
)
```

`pip install` gives you:

- the Python package as `refiner`
- the CLI as `macrodata`

## What To Expect

- training-data-first pipeline primitives instead of generic ETL abstractions
- multimodal processing, with robotics support today
- a lot of built-in readers, transforms, sinks, and lifecycle/runtime machinery so you do not have to rebuild the same scaffolding in scripts
- access to any storage backend supported by `fsspec`
- local execution for development and elastic cloud execution for large runs
- built-in observability through the Macrodata platform, so you can inspect how your data is changing instead of debugging blindly after the fact

## Technical Highlights

- **Shard-aware execution**
  Readers plan shards explicitly, and workers claim, heartbeat, complete, or fail them through the runtime lifecycle.
- **Fused execution**
  Adjacent Python row steps and Arrow-backed vectorized segments are compiled into a tighter execution plan rather than materializing between every transform.
- **Structured cloud submission**
  Launchers submit a structured plan, serialized pipeline payloads, and a manifest containing script text, dependency inventory, and ref/version metadata.
- **Secret-aware code capture**
  Captured code and script text are redacted before submission so platform introspection stays useful without leaking secret values.
- **Built-in observability**
  Jobs, stages, workers, shards, logs, metrics, and manifests are all part of the platform path already.
- **Specialized training-data sinks**
  Refiner already includes multistage writer flows like the LeRobot pipeline for robotics datasets.

## Docs

Getting started:

- [Pipeline basics](docs/pipeline-basics.md)
- [Launchers](docs/launchers.md)
- [CLI auth](docs/cli-auth.md)

Core concepts:

- [Local execution](docs/local-execution.md)
- [Readers and sharding](docs/readers-and-sharding.md)
- [Expression transforms](docs/expression-transforms.md)
- [Worker runtime](docs/worker-runtime.md)

Modalities and platform:

- [Robotics](docs/robotics.md)
- [Observability](docs/observability.md)
