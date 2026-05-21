from __future__ import annotations

import os
from typing import cast

import refiner as mdr


RAW_INPUT = os.environ["REFINER_EGO_INPUT"]
HF_OUTPUT = os.environ["REFINER_EGO_OUTPUT"]
HAWOR_EXPORT = os.environ.get("REFINER_HAWOR_EXPORT", "/opt/HaWoR/refiner_export.py")
JOB_NAME = os.environ.get("REFINER_JOB_NAME", "hawor-egocentric-actions")
NUM_WORKERS = int(os.environ.get("REFINER_NUM_WORKERS", "8"))
CUDA_VERSION = cast(
    mdr.CUDAVersion,
    os.environ.get("REFINER_CUDA_VERSION", "12.4"),
)


pipeline = (
    mdr.read_files(RAW_INPUT)
    .map(
        mdr.robotics.egocentric.reconstruct_hands_hawor(
            command=[
                "python",
                HAWOR_EXPORT,
                "--video",
                "{video_path}",
                "--result",
                "{result_path}",
            ],
            output_root="/tmp/hawor-artifacts",
        )
    )
    .map(mdr.robotics.egocentric.make_relative_actions())
    .write_jsonl(HF_OUTPUT)
)


if __name__ == "__main__":
    pipeline.launch_cloud(
        name=JOB_NAME,
        num_workers=NUM_WORKERS,
        gpu=mdr.GPU(count=1, type="h100", cuda_version=CUDA_VERSION),
        secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        env={
            "MACRODATA_BASE_URL": os.environ.get(
                "MACRODATA_BASE_URL",
                "https://dev.macrodata.co",
            ),
        },
    )
