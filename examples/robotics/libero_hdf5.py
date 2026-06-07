from __future__ import annotations

from datetime import datetime, timezone

import refiner as mdr


DEFAULT_DATASET_ROOT = "hf://datasets/yifengzhu-hf/LIBERO-datasets"
DEFAULT_OUTPUT_PREFIX = "hf://buckets/macrodata/test_bucket/libero-hdf5"
EVAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FPS = 10.0


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = f"{DEFAULT_OUTPUT_PREFIX}/{stamp}-full-eval"

    (
        mdr.read_hdf5(
            [f"{DEFAULT_DATASET_ROOT}/{suite}" for suite in EVAL_SUITES],
            groups="/data/demo_*",
            datasets={
                "raw_action": "actions",
                "observation.images.image": "obs/agentview_rgb",
                "observation.images.wrist_image": "obs/eye_in_hand_rgb",
                "ee_state": "obs/ee_states",
                "gripper_state": "obs/gripper_states",
            },
            file_path_column="file_path",
            group_path_column="hdf5_group",
            cache_remote_files=True,
        )
        .map(
            lambda row: row.update(
                task=str(row["file_path"])
                .rsplit("/", 1)[-1]
                .removesuffix(".hdf5")
                .removesuffix("_demo")
                .replace("_", " ")
            )
        )
        .to_robot_rows(
            task_key="task",
            fps=FPS,
            robot_type="libero",
            action_key="raw_action",
            state_key=("ee_state", "gripper_state"),
            video_keys={
                "observation.images.image": "observation.images.image",
                "observation.images.wrist_image": "observation.images.wrist_image",
            },
        )
        .write_lerobot(output, max_video_prepare_in_flight=2)
        .launch_cloud(
            name="libero-hdf5-full-eval",
            num_workers=40,
            cpus_per_worker=1,
            mem_mb_per_worker=1024,
            refiner_extras=("hdf5", "hf", "video"),
            secrets=mdr.Secrets.dict({"HF_TOKEN": None}),
        )
    )


if __name__ == "__main__":
    main()
