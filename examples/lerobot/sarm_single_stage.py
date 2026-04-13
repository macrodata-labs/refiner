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


if __name__ == "__main__":
    (
        mdr.read_lerobot(INPUT_DATASET)
        .map(prepare_sarm_episode)
        .write_lerobot(OUTPUT_DATASET)
        .launch_cloud(
            name="lerobot-sarm-single-stage",
            num_workers=1,
            mem_mb_per_worker=1024 * 2,
            secrets={
                "HF_TOKEN": "---",
            },
        )
    )
