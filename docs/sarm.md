---
title: "Preparing Data for SARM"
description: "Use Refiner to emit LeRobot datasets that train cleanly with LeRobot SARM single-stage mode"
---

LeRobot SARM does not need a special file format. For the simplest `single_stage`
setup, the current Hugging Face docs say each training sample needs:

- `task`
- the configured `policy.image_key`
- the configured `policy.state_key`

In practice that means Refiner only needs to make sure your LeRobot dataset has:

- a stable task string per episode
- a camera/video key you plan to train on
- a frame-level state column you plan to train on

This page shows the simplest preparation flow for `single_stage`. If you want
`dense_only` or `dual`, the extra annotation and visualization steps still happen
in LeRobot after dataset preparation.

## SARM Single-Stage Prep

```python
from __future__ import annotations

import refiner as mdr

INPUT_DATASET = "hf://datasets/your-username/your-robot-dataset"
OUTPUT_DATASET = "hf://buckets/your-username/your-sarm-ready-dataset"
IMAGE_KEY = "observation.images.main"
STATE_KEY = "observation.state"
TASK_DESCRIPTION: str | None = None


def prepare_sarm_episode(row):
    if IMAGE_KEY not in row.videos:
        raise ValueError(
            f"Episode {row.episode_index} is missing required video key {IMAGE_KEY!r}"
        )

    frame_columns = row.frames.table.column_names
    if STATE_KEY not in frame_columns:
        raise ValueError(
            f"Episode {row.episode_index} is missing required state key {STATE_KEY!r}"
        )

    task = TASK_DESCRIPTION
    if task is None:
        task = row.get("task")
    if task is None and len(row.tasks) == 1:
        task = row.tasks[0]
    if task is None:
        raise ValueError(
            "SARM single_stage needs a task string per episode. "
            "Set TASK_DESCRIPTION or add a single task per episode."
        )

    return row.update(task=str(task))


pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map(prepare_sarm_episode)
    .write_lerobot(OUTPUT_DATASET)
)
```

The checked-in version of this example lives at [examples/lerobot/sarm_single_stage.py](/Users/hynky/.codex/worktrees/d544/refiner/examples/lerobot/sarm_single_stage.py).

## Training

After the dataset is written, train SARM in `single_stage` mode with LeRobot:

```bash
lerobot-train \
  --dataset.repo_id=your-username/your-sarm-ready-dataset \
  --policy.type=sarm \
  --policy.annotation_mode=single_stage \
  --policy.image_key=observation.images.main \
  --policy.state_key=observation.state \
  --output_dir=outputs/train/sarm_single \
  --batch_size=32 \
  --steps=5000 \
  --policy.repo_id=your-username/your-model-name
```

## Notes

- If every episode already carries exactly one task, leaving `TASK_DESCRIPTION=None`
  is fine and the example reuses that task text.
- If episodes mix multiple tasks, set `TASK_DESCRIPTION` explicitly or write your own
  mapping function that emits the task string you want SARM to learn against.
- `single_stage` is the simplest path because it does not require sparse or dense
  subtask annotations.

## Internal Notes

- This workflow intentionally stays inside `read_lerobot(...)` / `write_lerobot(...)`
  because SARM trains on LeRobot datasets rather than a separate Refiner-specific
  export format.
- The example validates the chosen camera and state keys before writing so dataset
  problems fail early.
