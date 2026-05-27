---
title: "Episode Operations"
description: "Higher-level operations for robotics episode data"
---

# Episode Operations

Episode operations are reusable workflows built on top of Refiner transforms.
They operate on robotics episode rows rather than generic table columns.

| Operation | Use it for |
| --- | --- |
| [Motion Trimming](motion-trimming.md) | Remove inactive leading/trailing frames and clip videos. |
| [Subtask Annotation](subtask-annotation.md) | Use a VLM to label temporal subtask segments. |
| [Reward Scoring](reward-scoring.md) | Score episode progress/success with a pooling model. |
| [Egocentric Hand Tracking](egocentric-hand-tracking.md) | Run batched hand tracking over episode videos. |

Read [Episode Rows](../episode-data/episode-rows.md) before this section.

