"""Temporarily annotate Toloka HomER videos with VLM-generated subtasks."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import refiner as mdr
from refiner.pipeline.data.row import Row

INPUT_DATASET = "toloka/HomER"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket"
VIDEO_KEY = "observation.images.egocentric"
FPS = 30.0

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
OUTPUT_DATASET = f"{OUTPUT_ROOT}/toloka-homer-subtasks-{RUN_ID}"


def add_synthetic_timestamps(row: Row) -> Row:
    duration_sec = float(row["duration_sec"] or 0.0)
    frame_count = max(1, int(math.ceil(duration_sec * FPS)))
    timestamps = [index / FPS for index in range(frame_count)]
    return row.update({"timestamp": timestamps})


pipeline = (
    mdr.read_hf_dataset(
        INPUT_DATASET,
        columns_to_read=[
            "video_id",
            "video_url",
            "task_category",
            "scenario",
            "description",
            "duration_sec",
            "size_mb",
            "worker_id",
        ],
        num_shards=1,
    )
    .map(add_synthetic_timestamps)
    .to_robot_rows(
        episode_id_key="video_id",
        task_key="description",
        fps=FPS,
        timestamp_key="timestamp",
        action_key=None,
        state_key=None,
        video_keys={VIDEO_KEY: "video_url"},
    )
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key=VIDEO_KEY,
            output_column="predicted_subtasks",
            max_concurrent_requests=128,
        ),
        max_in_flight=128,
        preserve_order=False,
    )
    .write_lerobot(OUTPUT_DATASET)
)

pipeline.launch_cloud(
    name="toloka-homer-subtask-annotation",
    num_workers=1,
    cpus_per_worker=1,
    mem_mb_per_worker=4096,
    secrets=mdr.Secrets.dict(
        {
            "HF_TOKEN": None,
            "GOOGLE_GENERATIVE_AI_API_KEY": None,
        }
    ),
)
