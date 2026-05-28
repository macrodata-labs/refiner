<p align="center">
  <img src="https://macrodata.co/logo.svg" alt="Macrodata" width="180">
</p>

<h1 align="center">Macrodata Refiner</h1>

Refiner is an open-source engine for turning raw robotics and multimodal data into **high-quality datasets** for model training.

It gives training-data teams one pipeline model for multimodal data, robotics
workflows, and model-based processing.

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
    .write_lerobot("hf://buckets/acme-robotics/aloha_motion")
    .launch_cloud(
        name="motion_trim",
        num_workers=4,
    )
)
```

Need cloud GPUs? See [Resources, GPUs, and Services](docs/running-pipelines/resources-gpus-and-services.md).

### Local example

Launch a local pipeline:

```python
import refiner as mdr

def add_preview(row):
    return row.update(
        preview=" ".join(row["text"].split()[:20]),
    )

(
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .map(add_preview)
    .write_parquet("s3://my-bucket/english-cleanup/")
    .launch_local(
        name="english-cleanup",
        num_workers=2,
    )
)
```

`pip install` gives you:

- the Python package as `refiner`
- the CLI as `macrodata`

## Batteries included

- training-data-first pipeline primitives instead of generic ETL abstractions
- multimodal processing, with robotics support today
- built-in readers, transforms, sinks, and runtime machinery for common dataset work
- access to any storage backend supported by `fsspec` (S3, GCP, Hugging Face, etc.)
- local execution for development and elastic cloud execution for large runs
- built-in observability through the Macrodata platform for job state, logs, metrics, and manifests

## Docs

Start here:

- [Docs index](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Running pipelines](docs/running-pipelines/index.md)

Build a dataset:

- [Reading data](docs/reading-data/index.md)
- [Episode data](docs/episode-data/index.md)
- [Transforms](docs/transforms/index.md)
- [Episode operations](docs/episode-operations/index.md)
- [Writing data](docs/writing-data/index.md)
- [Examples](docs/examples/index.md)

Operate jobs:

- [Platform](docs/platform/index.md)
- [CLI](docs/cli/index.md)
- [Reference](docs/reference/index.md)

## Community

- join the Macrodata Discord: https://discord.gg/S8kZtmBR2x
