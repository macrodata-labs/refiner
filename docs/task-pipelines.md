---
title: "Task Pipelines"
description: "Run arbitrary task-style work with rank and world size"
---

Use `mdr.task(...)` when you want Refiner to launch task-style work instead of reading rows from a dataset.

## What It Does

`mdr.task(fn, num_tasks=...)` creates a pipeline with one callback invocation per task rank.

Your callback receives:

- `rank`
- `world_size`

and returns a row-like result.

## Example

```python
import refiner as mdr

def task_worker(rank: int, world_size: int) -> dict:
    return {
        "rank": rank,
        "world_size": world_size,
        "shard_prefix": f"part-{rank:05d}",
    }

(
    mdr.task(task_worker, num_tasks=8)
    .write_jsonl("s3://my-bucket/task-output/")
    .launch_cloud(
        name="task-example",
        num_workers=8,
    )
)
```

## When To Use It

Use task pipelines when:

- there is no source dataset to read from
- you want one unit of work per rank
- you need rank/world-size aware initialization logic
- you want to fan out arbitrary work and still use Refiner launchers, sinks, and observability

## Notes

- task pipelines still participate in normal launcher/runtime behavior
- each task callback is represented as a row-producing stage, so you can keep composing normal transforms and sinks afterward

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Reading and writing data](reading-and-writing.md)
- [Transforms](transforms.md)
- [Launchers](launchers.md)
