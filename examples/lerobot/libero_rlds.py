from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr

RAW_DATASETS = [
    "hf://datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_goal_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_object_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
]
DEFAULT_OUTPUT_PREFIX = "hf://buckets/macrodata/test_bucket/libero-rlds"
FPS = 10.0


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = f"{DEFAULT_OUTPUT_PREFIX}/{stamp}-full-eval"

    (
        mdr.read_tfds(
            RAW_DATASETS,
            videos={
                "front": "steps/observation/image",
                "wrist": "steps/observation/wrist_image",
            },
            fps=FPS,
        )
        .to_robot_rows(
            nested_frames_key="steps",
            episode_id_key="episode_metadata/file_path",
            task_key="steps/language_instruction",
            action_key="action",
            state_key="observation/state",
            video_keys={
                "observation.images.front": "videos/front",
                "observation.images.wrist": "videos/wrist",
            },
            fps=FPS,
            robot_type="libero",
        )
        .write_lerobot(output, max_video_prepare_in_flight=2)
        .launch_cloud(
            name="libero-rlds-full-eval",
            num_workers=40,
            cpus_per_worker=1,
            mem_mb_per_worker=1024,
            extra_dependencies=(
                "av",
                "huggingface-hub>=1.4.1",
                "pillow",
                "tensorflow",
                "tensorflow-datasets",
            ),
            secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        )
    )


if __name__ == "__main__":
    main()
