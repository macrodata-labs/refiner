"""Annotate LeRobot episodes with VLM-generated temporal subtasks."""

from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr

INPUT_DATASET = "hf://datasets/lerobot/berkeley_cable_routing"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket"
VIDEO_KEY = "observation.images.top_image"


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
OUTPUT_DATASET = f"{OUTPUT_ROOT}/berkeley-cable-routing-subtasks-{RUN_ID}"

PROFILE = mdr.robotics.DomainProfile(
    domain_id="berkeley-cable-routing",
    version="1",
    policy=mdr.robotics.MANIPULATION_EVENTS_V1,
    gold_set="berkeley-cable-routing-consensus-v1",
)

pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map_async(
        mdr.robotics.subtask_annotation(
            profile=PROFILE,
            provider=mdr.inference.GoogleEndpointProvider(model="gemini-3.5-flash"),
            video_key=VIDEO_KEY,
            output_column="predicted_subtasks",
            result_column="subtask_annotation_result",
            thinking_budget=16384,
            max_concurrent_requests=256,
        ),
        max_in_flight=256,
    )
    .map_async(
        mdr.robotics.subtask_labeling(
            video_key=VIDEO_KEY,
            segments_column="predicted_subtasks",
            output_column="labeled_subtasks",
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
    secrets=mdr.Secrets.dict(
        {
            "HF_TOKEN": None,
            "GOOGLE_GENERATIVE_AI_API_KEY": None,
        }
    ),
)
