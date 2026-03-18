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

### Cloud example

Launch a robotics pipeline on Macrodata Cloud.

This requires a valid API key.

```python
import refiner as mdr

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
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

### Local example

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
    .map(lambda row: {"text": row["text"], "bucket": "long" if row["text_len"] > 512 else "short"})
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

## what refiner gives you

- training-data-first pipeline primitives instead of generic ETL abstractions
- multimodal processing, with robotics support today
- a lot of built-in readers, transforms, sinks, and lifecycle/runtime machinery so you do not have to rebuild the same scaffolding in scripts
- access to any storage backend supported by `fsspec` (S3, GCP, Hugging Face, etc.)
- local execution for development and elastic cloud execution for large runs
- built-in observability through the Macrodata platform, so you can inspect how your data is changing instead of debugging blindly after the fact

## Docs

Getting started:

- [Pipeline basics](docs/pipeline-basics.md)
- [Launchers](docs/launchers.md)
- [CLI](docs/cli.md)

Core concepts:

- [In-process debugging](docs/in-process-debugging.md)
- [Readers and sharding](docs/readers-and-sharding.md)
- [Expression transforms](docs/expression-transforms.md)

Modalities and platform:

- [Robotics](docs/robotics.md)
- [Observability](docs/observability.md)
