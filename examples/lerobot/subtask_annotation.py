from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr

INPUT_DATASET = "hf://datasets/lerobot/berkeley_cable_routing"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket"
VIDEO_KEY = "observation.images.top_image"


run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
output = f"{OUTPUT_ROOT}/berkeley-cable-routing-subtasks-{run_id}"

pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key=VIDEO_KEY,
            output_column="predicted_subtasks",
        ),
    )
    .write_lerobot(output)
)

pipeline.launch_cloud(
    name="berkeley-subtask-annotation",
    num_workers=4,
    cpus_per_worker=1,
    mem_mb_per_worker=2048,
    secrets=[
        mdr.Secrets.env(keys=["HF_TOKEN"]),
        {"GOOGLE_GENERATIVE_AI_API_KEY": None},
    ],
    env={
        "SUBTASK_ANNOTATION_INPUT_DATASET": INPUT_DATASET,
        "SUBTASK_ANNOTATION_VIDEO_KEY": VIDEO_KEY,
        "SUBTASK_ANNOTATION_OUTPUT": output,
    },
)
