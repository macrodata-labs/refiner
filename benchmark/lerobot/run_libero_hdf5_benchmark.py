from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

import refiner as mdr
from refiner.pipeline.data.row import Row


DEFAULT_DATASET_ROOT = "hf://datasets/yifengzhu-hf/LIBERO-datasets"
DEFAULT_OUTPUT_PREFIX = "hf://buckets/macrodata/test_bucket/libero-hdf5-benchmark"
EVAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EPISODES_PER_FILE = 50
FPS = 20.0


def normalize_libero_row(row: Row) -> Row:
    action = np.asarray(row["raw_action"], dtype=np.float32)
    file_path = str(row["file_path"])
    task = file_path.rsplit("/", 1)[-1].removesuffix(".hdf5").removesuffix("_demo")
    demo = str(row["hdf5_group"]).rsplit("/", 1)[-1]
    return row.update(
        episode_id=f"{file_path.rsplit('/', 2)[-2]}/{task}/{demo}",
        task=task.replace("_", " "),
        action=np.concatenate(
            [action[:, :6], (1.0 - np.clip(action[:, -1], 0.0, 1.0))[:, None]],
            axis=1,
        ),
        timestamp=np.arange(action.shape[0], dtype=np.float32) / FPS,
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
        .map(normalize_libero_row)
        .to_robot_rows(
            episode_id_key="episode_id",
            task_key="task",
            fps=FPS,
            robot_type="libero",
            action_key="action",
            state_key=("ee_state", "gripper_state"),
            video_keys={
                "observation.images.image": "observation.images.image",
                "observation.images.wrist_image": "observation.images.wrist_image",
            },
        )
        .write_lerobot(output)
        .launch_cloud(
            name=f"libero-hdf5-eval-{EPISODES_PER_FILE}ep-cached",
            num_workers=40,
            cpus_per_worker=1,
            mem_mb_per_worker=1024,
            extra_dependencies=("av", "h5py", "huggingface-hub>=1.4.1", "pillow"),
            secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        )
    )


if __name__ == "__main__":
    main()
