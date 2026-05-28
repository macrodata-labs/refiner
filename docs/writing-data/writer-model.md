---
title: "Writer Model"
description: "How Refiner writer sinks run and finalize outputs"
---

# Writer Model

A writer is a sink attached to a pipeline. It receives output blocks from
workers and writes files, media, metadata, or reducer inputs.

## Common Writer Behavior

| Behavior | Why it matters |
| --- | --- |
| Shard-local writes | Workers can write independently. |
| Worker-aware filenames | Avoids collisions between workers. |
| Asset handling | Media columns can be copied, uploaded, remuxed, or transcoded. |
| Reducer stage | Some formats need a final merge or metadata pass. |

## Attaching A Writer

```python
pipeline = pipeline.write_parquet("/tmp/output")
```

This returns a new pipeline with a sink. It does not write immediately.

## Launching

```python
pipeline.launch_local(name="write-test", num_workers=2)
```

or:

```python
pipeline.launch_cloud(name="write-cloud", num_workers=16)
```

## Related Pages

- [LeRobot Writer](lerobot.md)
- [Media Assets and Reducers](media-assets-and-reducers.md)
- [Running Pipelines](../running-pipelines/index.md)

