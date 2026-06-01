from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

import refiner as mdr
from refiner.pipeline.data.row import Row


DEFAULT_DATASET_ROOT = "hf://datasets/yifengzhu-hf/LIBERO-datasets"
EVAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_OUTPUT_PREFIX = "hf://buckets/macrodata/test_bucket/libero-hdf5-benchmark"
FULL_EVAL_FILES = 40
EPISODES_PER_FILE = 50
DEFAULT_CLOUD_DEPENDENCIES = (
    "av",
    "h5py",
    "huggingface-hub>=1.4.1",
    "pillow",
)


def normalize_libero_row(row: Row, *, fps: float) -> Row:
    action = np.asarray(row["raw_action"], dtype=np.float32)
    action = np.concatenate(
        [action[:, :6], (1.0 - np.clip(action[:, -1], 0.0, 1.0))[:, None]],
        axis=1,
    )
    file_path = str(row["file_path"])
    task = Path(file_path.rsplit("/", 1)[-1]).stem.removesuffix("_demo")
    suite = Path(file_path.rsplit("/", 2)[0]).name
    demo = str(row["hdf5_group"]).rsplit("/", 1)[-1]
    length = int(action.shape[0])
    return row.update(
        episode_id=f"{suite}/{task}/{demo}",
        task=task.replace("_", " "),
        action=action,
        **{
            "timestamp": np.arange(length, dtype=np.float32) / fps,
        },
    ).drop("raw_action")


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = f"{DEFAULT_OUTPUT_PREFIX}/{stamp}-{EPISODES_PER_FILE}ep"

    (
        mdr.read_hdf5(
            [f"{DEFAULT_DATASET_ROOT}/{suite}" for suite in EVAL_SUITES],
            groups=[f"/data/demo_{index}" for index in range(EPISODES_PER_FILE)],
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
        .map(partial(normalize_libero_row, fps=20.0))
        .to_robot_rows(
            episode_id_key="episode_id",
            task_key="task",
            fps_key="fps",
            robot_type="libero",
            action_key="action",
            state_key=("ee_state", "gripper_state"),
            video_keys={
                "observation.images.image": "observation.images.image",
                "observation.images.wrist_image": "observation.images.wrist_image",
            },
        )
        .write_lerobot(
            output,
        )
        .launch_cloud(
            name=f"libero-hdf5-eval-{EPISODES_PER_FILE}ep-cached",
            num_workers=FULL_EVAL_FILES,
            cpus_per_worker=1,
            mem_mb_per_worker=1024,
            extra_dependencies=DEFAULT_CLOUD_DEPENDENCIES,
            secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        )
    )


if __name__ == "__main__":
    main()
