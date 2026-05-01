---
title: "Pipeline Basics"
description: "Overview of the basic Refiner pipeline model"
---

A Refiner pipeline has three parts:

- one source
- zero or more ordered transforms
- an optional sink

Most pipelines also end with a launcher:

- `.launch_local(...)`
- `.launch_cloud(...)`

So the usual shape is:

```python
import refiner as mdr

pipeline = mdr.from_items(
    [{"text": "hello world", "lang": "en"}]
)
pipeline = pipeline.filter(mdr.col("lang") == "en")
pipeline = pipeline.write_jsonl("s3://my-bucket/example-output/")
```

This page is just the overview. The detailed behavior lives on the dedicated
pages linked below.

## Sources

Sources define how rows enter the pipeline.

Common entry points:

- `read_csv(...)`
- `read_jsonl(...)`
- `read_parquet(...)`
- `read_lerobot(...)`
- `read_tfrecords(...)`
- `read_tfds(...)`
- `from_items(...)`
- `from_source(...)`

There is also a task-style entry point:

- `task(...)`

Use that when there is no input dataset and you want one callback invocation per
rank. See [Task pipelines](task-pipelines.md).

For row shape, sharding, and source-specific behavior, see
[Reading and writing data](reading-and-writing.md).

## Transforms

Transforms sit between the source and the sink.

Examples:

- Python transforms like `.map(...)`, `.flat_map(...)`, `.batch_map(...)`, `.map_async(...)`
- expression-backed transforms like `.filter(...)`, `.with_columns(...)`, `.select(...)`, `.drop(...)`, `.rename(...)`, `.cast(...)`

For the transform surface area, see [Transforms](transforms.md).

For the expression DSL itself, see [Expressions](expressions.md).

## Sinks

Sinks define how output is written once the pipeline is launched.

Common sinks:

- `.write_jsonl(...)`
- `.write_parquet(...)`
- `.write_lerobot(...)`

For the built-in readers and writers, see
[Reading and writing data](reading-and-writing.md).

## Launchers

Launchers turn the pipeline definition into an actual job:

- `.launch_local(...)` runs worker subprocesses on the current machine
- `.launch_cloud(...)` submits the job to Macrodata Cloud

See [Launchers](launchers.md).

## Related Pages

- [Reading and writing data](reading-and-writing.md)
- [Transforms](transforms.md)
- [Expressions](expressions.md)
- [In-process debugging](in-process-debugging.md)
- [Launchers](launchers.md)
- [Task pipelines](task-pipelines.md)
