<p align="center">
  <img src="https://macrodata.co/logo.svg" alt="Macrodata" width="180">
</p>

<h1 align="center">Macrodata Refiner</h1>

Refiner is Macrodata's open-source Python library for reading, transforming, and
writing robotics datasets.

It provides one pipeline model for working with robot episodes, frames, videos,
metadata, and model-based processing. Use it to convert formats, transform data,
run inference, and write structured outputs on your own infrastructure.

This repository also includes open-source reference versions of some of the
pipelines we develop at Macrodata. They are useful starting points, but they are
not the full pipelines we adapt, evaluate, and run for customers. If you want to
see what those pipelines can do with your data,
[send us a representative sample](https://macrodata.co/contact).

## Quickstart

Install:

```bash
pip install macrodata-refiner
```

This gives you:

- the Python package as `refiner`
- the CLI as `macrodata`

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

## Batteries included

- a consistent row and episode model for robot trajectories, frames, videos,
  metadata, tasks, and statistics
- readers and writers for LeRobot, HDF5, Zarr, MCAP, Parquet, JSONL, and other
  common data formats
- composable transforms and model inference for converting and enriching data
- open-source reference operations for motion trimming, subtask annotation,
  reward scoring, and hand tracking
- access to storage backends supported by `fsspec`, including S3, GCP, and
  Hugging Face
- in-process debugging and local multi-worker execution

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
- [Reference](docs/reference/index.md)

## Community

- join the Macrodata Discord: https://discord.gg/S8kZtmBR2x
