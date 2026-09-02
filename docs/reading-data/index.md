---
title: "Reading data"
description: "Choose and configure Refiner readers for robotics data"
---

# Reading data

Readers create the source of a Refiner pipeline. A reader is responsible for
finding input files, planning shards, and emitting rows or table blocks.

```python
import refiner as mdr

pipeline = mdr.read_lerobot("hf://datasets/lerobot/aloha_sim_transfer_cube_human")
```

## Reader selection

| Your data looks like | Use | Read |
| --- | --- | --- |
| LeRobot dataset root | `read_lerobot` | [LeRobot](lerobot.md) |
| One HDF5 file per episode, or grouped HDF5 demos | `read_hdf5` | [HDF5](hdf5.md) |
| Zarr replay buffer with episode boundaries | `read_zarr` | [Zarr](zarr.md) |
| MCAP robotics or autonomy logs | `read_mcap` | [MCAP](mcap.md) |
| Parquet, JSON, JSONL, CSV tables | `read_parquet`, `read_json`, `read_jsonl`, `read_csv` | [Tabular Files](tabular-files.md) |
| Raw files or media files | `read_files`, `read_videos` | [Files and Videos](files-and-videos.md) |
| Hugging Face datasets table | `read_hf_dataset` | [Hugging Face](hugging-face.md) |
| TFRecord files or TensorFlow Datasets | `read_tfrecords`, `read_tfds` | [TensorFlow](tensorflow.md) |
| Versioned Lance dataset | `load_lance` | [Lance](lance.md) |
| Your own source system | `from_source` | [Custom Readers](custom-readers.md) |

## Core ideas

- Readers plan **shards**, the units workers execute.
- Readers emit **rows** or **tabular blocks**.
- Robotics readers usually emit one row per episode.
- Generic readers can be adapted into episode rows with
  [`to_robot_rows`](../episode-data/converting-to-robot-rows.md).

Read [Reader Model](reader-model.md) and [Sharding](sharding.md) before writing
large jobs.

## Run a bounded quick test

Pass `max_rows` to a built-in reader before testing transforms or allocating a
full cloud run:

```python
import refiner as mdr

pipeline = mdr.read_parquet(
    "s3://my-bucket/episodes/*.parquet",
    columns_to_read=["episode_id", "video"],
    max_rows=100,
)

rows = pipeline.take(5)
```

`max_rows` is a global source-output cap, not a per-worker cap. It applies after
reader-level operations such as Parquet or Hugging Face filtering, but before
pipeline transforms. A positive limit uses one source shard so multiple workers
cannot each emit the requested number of rows. Consequently,
`num_workers="auto"` starts at most one worker for a limited source. Set
`max_rows=0` to execute no source shards.

The option is available on `read_csv`, `read_json`, `read_jsonl`, `read_files`,
`read_videos`, `read_hdf5`, `read_zarr`, `read_mcap`, `read_parquet`,
`read_hf_dataset`, `read_lerobot`, `read_tfrecords`, `read_tfds`, and
`load_lance`. All use the same bounded-source wrapper.

Readers stop before opening later source shards and slice the final Arrow batch.
Refiner also reduces configurable batch or concurrency windows for Lance,
Parquet, Hugging Face, file-content, Zarr, LeRobot, TFRecord, and TFDS readers;
bounded split MCAP reads stream one episode at a time when the input supports
it. A reader may still need to load one indivisible physical unit to produce a
row—for example, one whole-file JSON document, HDF5 group, MCAP episode, or
LeRobot episode. The limit bounds later work but cannot make that first logical
row smaller.

## Internal Notes

Spark, Daft, and Ray Data can represent a limit as a distributed logical-plan
node, with their schedulers coordinating partitions and ordering. Beam/Dataflow
treats collections as unordered, so a deterministic leading-row limit requires
additional ordering or runner-specific behavior. Hugging Face Datasets offers
simple local selection or iterable `take` operations. Refiner follows the latter
DX but exposes one deterministic scheduling shard for limited runs. This trades
parallelism—which is not useful for a small smoke test—for an exact global cap
without adding distributed limit coordination to the worker protocol.
