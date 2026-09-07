---
title: "Writer model"
description: "How Refiner writer sinks run and finalize outputs"
---

# Writer model

A writer is a sink attached to a pipeline. It receives output blocks from
workers and writes files, media, metadata, or reducer inputs.

## Common writer behavior

| Behavior | Why it matters |
| --- | --- |
| Shard-local writes | Workers can write independently. |
| Worker-aware filenames | Avoids collisions between workers. |
| Asset handling | Media columns can be copied, uploaded, remuxed, or transcoded. |
| Follow-up stages | Some formats need final cleanup, merging, or metadata passes. |

## Attaching a writer

```python
pipeline = pipeline.write_parquet("/tmp/output")
```

This returns a new pipeline with a sink. It does not write immediately.

Writers that need multiple execution stages expand automatically when the
pipeline is launched. Keep using the normal `write_*` method; `then(...)` is
only needed when you are composing separate pipelines yourself.

## Launching

```python
pipeline.launch_local(name="write-test", num_workers=2)
```

## Related pages

- [LeRobot Writer](lerobot.md)
- [Media Assets and Reducers](media-assets-and-reducers.md)
- [Running Pipelines](../running-pipelines/index.md)
