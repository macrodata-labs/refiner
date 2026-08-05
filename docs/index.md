---
title: "Refiner docs"
description: "Read, transform, and write robotics datasets with Refiner"
---

# Refiner docs

Refiner is Macrodata's open-source Python library for reading, transforming, and
writing robotics datasets. It provides one pipeline model for working with
robot episodes, frames, videos, metadata, and model-based processing.

These docs also include open-source reference versions of some of the pipelines
we develop at Macrodata. They are useful starting points, but they are not the
full pipelines we adapt, evaluate, and run for customers.

The docs are organized around the path most teams follow:

| Step | Start here | What you should know after reading |
| --- | --- | --- |
| Try Refiner | [Quickstart](quickstart.md) | How a complete read-transform-write pipeline looks. |
| Run code | [Running Pipelines](running-pipelines/index.md) | How to inspect pipelines in process and launch local workers. |
| Load data | [Reading Data](reading-data/index.md) | Which reader to use and how input sharding works. |
| Understand rows | [Episode Data](episode-data/index.md) | How episodes, frame tables, videos, metadata, tasks, and stats are represented. |
| Transform data | [Transforms](transforms/index.md) | How `map`, `map_async`, `batch_map`, expressions, and dtypes fit together. |
| Use packaged workflows | [Episode Operations](episode-operations/index.md) | How to trim motion, annotate subtasks, score rewards, and run perception models. |
| Call models | [Inference](inference/index.md) | How to use text, multimodal, structured, vLLM, and pooling inference. |
| Save outputs | [Writing Data](writing-data/index.md) | How writers stage files, media, and reducers. |
| Follow recipes | [Examples](examples/index.md) | End-to-end dataset conversion and enrichment workflows. |
| Use commands | [CLI](cli/index.md) | The `macrodata` command surface. |

For quick API lookup, see [Reference](reference/index.md).

## Recommended reading order

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

If you want to see what the full pipelines can do with your data,
[send us a representative sample](https://macrodata.co/contact). We will review
it and show you what we can extract.
