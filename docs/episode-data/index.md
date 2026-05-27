---
title: "Episode Data"
description: "How Refiner represents robotics episodes, frames, videos, metadata, tasks, and stats"
---

# Episode Data

Refiner treats robotics data as episode data. An episode row can carry:

- episode-level values, such as `episode_id` or `task`
- frame-level arrays, such as timestamps, actions, states, and observations
- lazy video sources
- metadata, tasks, and statistics used by LeRobot-compatible datasets

## Pages

| Page | Use it for |
| --- | --- |
| [Episode Rows](episode-rows.md) | The row API and semantic episode fields. |
| [Frames and Videos](frames-and-videos.md) | Frame tables, video sources, clipping, and decoding. |
| [Metadata, Tasks, and Stats](metadata-tasks-and-stats.md) | LeRobot metadata and feature statistics. |
| [Converting to Robot Rows](converting-to-robot-rows.md) | Adapting HDF5, Zarr, and custom rows into episode rows. |

Read this section before writing episode-level transforms.

