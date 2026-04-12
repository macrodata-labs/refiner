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

and should return a row-like result.

In practice, task pipelines are often used for rank-aware side effects. The returned row can be just a small bookkeeping record if the real work happens inside the callback.

## Example

```python
import refiner as mdr

def task_worker(rank: int, world_size: int) -> dict[str, object] | None:
    # Do rank-aware work here, for example:
    # - call an external tool
    # - run inference
    # - write a partition of output data
    # - produce one bookkeeping row per task
    # Returning a row is optional; return None if the task is purely side-effecting.

    return {
        "rank": rank,
        "world_size": world_size,
        "status": "done",
    }

(
    mdr.task(task_worker, num_tasks=8)
    .write_jsonl("s3://my-bucket/task-output/")
    # Add a writer only if you want to persist bookkeeping rows from the task callback.
    .launch_cloud(
        name="task-example",
        # Workers claim task shards until all tasks are complete,
        # so this can be lower than num_tasks.
        num_workers=4,
    )
)
```

The writer is completely optional. Add one if you want Refiner to persist the task results. If the useful work happens entirely inside the callback, the returned row can stay minimal.

## When To Use It

Use task pipelines when:

- there is no source dataset to read from
- you want one unit of work per rank
- you need rank/world-size aware initialization logic
- the real work is side-effecting and the returned row is just bookkeeping
- you want to fan out arbitrary work and still use Refiner launchers, sinks, and cloud observability

## Notes

- task pipelines still participate in normal launcher/runtime behavior
- each task callback is represented as a row-producing stage, so you can still compose normal transforms and sinks afterward
- task callbacks still need to return a row-like result today, even if it is just minimal bookkeeping

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Reading and writing data](reading-and-writing.md)
- [Transforms](transforms.md)
- [Launchers](launchers.md)
