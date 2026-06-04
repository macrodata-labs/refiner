---
title: "Examples"
description: "End-to-end Refiner workflows for robotics data"
---

# Examples

These examples are complete patterns you can adapt.

## Formats

| Example | What It Shows |
| --- | --- |
| [ALOHA HDF5](formats/aloha-hdf5.md) | Convert one-file-per-episode HDF5 data. |
| [robomimic HDF5](formats/robomimic-hdf5.md) | Convert grouped HDF5 demonstrations. |
| [Zarr Replay Buffer](formats/zarr-replay-buffer.md) | Convert replay-buffer arrays to episode rows. |
| [MCAP Franka](formats/mcap-franka.md) | Convert MCAP state/action/video topics to LeRobot and Zarr. |
| [Libero HDF5](formats/libero-hdf5.md) | Convert the LIBERO HDF5 eval datasets on cloud workers. |

## Datasets

| Example | What It Shows |
| --- | --- |
| [Merge LeRobot Datasets](datasets/merge-lerobot-datasets.md) | Combine compatible LeRobot roots. |

## Annotations

| Example | What It Shows |
| --- | --- |
| [Video Subtask Annotations](annotations/subtask-annotations.md) | Use a VLM to add segments and annotate videos. |
| [Running Reward Models](annotations/reward-models.md) | Use reward models to score episodes. |
| [Hand Tracking](annotations/hand-tracking.md) | Run ego-vision hand tracking on HomER videos. |

## Cloud

| Example | What It Shows |
| --- | --- |
| [Cloud GPU Job](cloud/cloud-gpu-job.md) | Request GPUs for model-heavy work. |
