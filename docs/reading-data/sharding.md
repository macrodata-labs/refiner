---
title: "Sharding"
description: "How Refiner divides source data across workers"
---

# Sharding

A shard is a unit of input work. In launched runs, workers claim shards and
process them independently.

## Why Shards Matter

Shard boundaries affect:

- parallelism
- retry granularity
- output file layout
- progress reporting
- reducer work after writer stages

## Common Shard Strategies

| Input type | Typical shard strategy |
| --- | --- |
| Large line-oriented files | Split by byte ranges aligned to row boundaries. |
| Many small files | Pack files into shards by estimated size. |
| LeRobot episode parquet | Plan over episode metadata parquet files. |
| HDF5 demos | Plan files, then emit matched groups. |
| Zarr replay buffers | Plan row ranges from episode boundaries or leading-axis rows. |

## Tuning Shards

Use `target_shard_bytes` when the reader can estimate byte size:

```python
pipeline = mdr.read_parquet(
    "/data/episodes/*.parquet",
    target_shard_bytes=256 * 1024 * 1024,
)
```

Use `num_shards` when you want a coarse target:

```python
pipeline = mdr.read_lerobot(
    "hf://datasets/acme/large-robot-dataset",
    num_shards=128,
)
```

Do not overfit shard count before measuring. Too few shards leave workers idle;
too many shards add scheduling and output overhead.

## Shards And Writers

Writers usually write shard-local files first. Some writers add a reducer stage
to merge or finalize metadata. For example, the LeRobot writer stages per-shard
chunks and then reduces metadata; see [Writing LeRobot](../writing-data/lerobot.md).

## Related Pages

- [Reader Model](reader-model.md)
- [Local Launcher](../running-pipelines/local-launcher.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)

