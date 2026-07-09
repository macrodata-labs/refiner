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
| Rerun RRD recordings | `read_rerun` | [Rerun](rerun.md) |
| Parquet, JSON, JSONL, CSV tables | `read_parquet`, `read_json`, `read_jsonl`, `read_csv` | [Tabular Files](tabular-files.md) |
| Raw files or media files | `read_files`, `read_videos` | [Files and Videos](files-and-videos.md) |
| Hugging Face datasets table | `read_hf_dataset` | [Hugging Face](hugging-face.md) |
| TFRecord files or TensorFlow Datasets | `read_tfrecords`, `read_tfds` | [TensorFlow](tensorflow.md) |
| Your own source system | `from_source` | [Custom Readers](custom-readers.md) |

## Core ideas

- Readers plan **shards**, the units workers execute.
- Readers emit **rows** or **tabular blocks**.
- Robotics readers usually emit one row per episode.
- Generic readers can be adapted into episode rows with
  [`to_robot_rows`](../episode-data/converting-to-robot-rows.md).

Read [Reader Model](reader-model.md) and [Sharding](sharding.md) before writing
large jobs.
