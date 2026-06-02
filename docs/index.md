---
title: "Refiner Docs"
description: "A guide to building, running, and operating robotics data pipelines with Refiner"
---

# Refiner Docs

Refiner is a Python framework for robotics data pipelines. It reads episode
datasets, exposes frames and videos through a consistent row model, transforms
or enriches those episodes, and writes training-ready outputs such as LeRobot,
Zarr, Parquet, or JSONL.

The docs are organized around the path most teams follow:

| Step | Start here | What you should know after reading |
| --- | --- | --- |
| Try Refiner | [Quickstart](quickstart.md) | How a complete read-transform-write pipeline looks. |
| Run code | [Running Pipelines](running-pipelines/index.md) | How local iteration, local launch, and cloud launch differ. |
| Load data | [Reading Data](reading-data/index.md) | Which reader to use and how input sharding works. |
| Understand rows | [Episode Data](episode-data/index.md) | How episodes, frame tables, videos, metadata, tasks, and stats are represented. |
| Transform data | [Transforms](transforms/index.md) | How `map`, `map_async`, `batch_map`, expressions, and dtypes fit together. |
| Use packaged workflows | [Episode Operations](episode-operations/index.md) | How to trim motion, annotate subtasks, score rewards, and run perception models. |
| Call models | [Inference](inference/index.md) | How to use text, multimodal, structured, vLLM, and pooling inference. |
| Save outputs | [Writing Data](writing-data/index.md) | How writers stage files, media, and reducers. |
| Follow recipes | [Examples](examples/index.md) | End-to-end dataset conversion and enrichment workflows. |
| Use Macrodata Cloud | [Platform](platform/index.md) | Workspaces, API keys, manifests, cloud jobs, files, and secrets. |
| Use commands | [CLI](cli/index.md) | The `macrodata` command surface. |

For quick API lookup, see [Reference](reference/index.md).

## Recommended Reading Order

If you are new to Refiner, read:

1. [Quickstart](quickstart.md)
2. [In-Process Debugging](running-pipelines/in-process-debugging.md)
3. [LeRobot Reader](reading-data/lerobot.md)
4. [Episode Rows](episode-data/episode-rows.md)
5. [Row Transforms](transforms/row-transforms.md)
6. [LeRobot Writer](writing-data/lerobot.md)

If you already have a custom source dataset, start with
[HDF5](reading-data/hdf5.md), [Zarr](reading-data/zarr.md), or
[Converting to Robot Rows](episode-data/converting-to-robot-rows.md).
