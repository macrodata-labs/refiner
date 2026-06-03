"""Annotate LeRobot episodes with VLM-generated temporal subtasks."""

from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr

INPUT_DATASET = "hf://datasets/lerobot/berkeley_cable_routing"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket"
VIDEO_KEY = "observation.images.top_image"


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
OUTPUT_DATASET = f"{OUTPUT_ROOT}/berkeley-cable-routing-subtasks-{RUN_ID}"

pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key=VIDEO_KEY,
            output_column="predicted_subtasks",
            max_concurrent_requests=256,
        ),
        max_in_flight=256,
    )
    .write_lerobot(OUTPUT_DATASET)
)

pipeline.launch_cloud(
    name="berkeley-subtask-annotation",
    num_workers=1,
    cpus_per_worker=1,
    mem_mb_per_worker=2048,
    secrets=[
        mdr.Secrets.env(keys=["HF_TOKEN"]),
        {"GOOGLE_GENERATIVE_AI_API_KEY": None},
    ],
)
