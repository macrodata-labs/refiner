from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from fsspec import url_to_fs
import numpy as np

import refiner as mdr
from refiner.pipeline.data.row import Row


DEFAULT_INPUT = (
    "hf://datasets/yifengzhu-hf/LIBERO-datasets/libero_goal/turn_on_the_stove_demo.hdf5"
)
DEFAULT_DATASET_ROOT = "hf://datasets/yifengzhu-hf/LIBERO-datasets"
EVAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_OUTPUT_PREFIX = "hf://buckets/macrodata/test_bucket/libero-hdf5-benchmark"
DEFAULT_CLOUD_DEPENDENCIES = (
    "av",
    "h5py",
    "huggingface-hub>=1.4.1",
    "pillow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append")
    parser.add_argument("--eval-suite", action="store_true")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-files-per-suite", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--mem-mb", type=int, default=4096)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--cache-remote-files", action="store_true")
    parser.add_argument("--codec", default="mpeg4")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-video-prepare-in-flight", type=int, default=2)
    parser.add_argument("--secret-env", default="default")
    parser.add_argument("--sync-local-dependencies", action="store_true")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> str | list[str]:
    if not args.eval_suite:
        return args.input or DEFAULT_INPUT

    roots = [f"{args.dataset_root.rstrip('/')}/{suite}" for suite in EVAL_SUITES]
    if args.max_files_per_suite is None:
        return roots

    inputs: list[str] = []
    for root in roots:
        fs, path = url_to_fs(root)
        matches = sorted(fs.glob(f"{path}/*.hdf5"))
        inputs.extend(
            fs.unstrip_protocol(match) for match in matches[: args.max_files_per_suite]
        )
    return inputs


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
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_files_per_suite is not None and args.max_files_per_suite <= 0:
        raise ValueError("--max-files-per-suite must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"{DEFAULT_OUTPUT_PREFIX}/{stamp}-{args.episodes}ep"
    name = args.name or f"libero-hdf5-{args.episodes}ep"
    inputs = resolve_inputs(args)

    pipeline = (
        mdr.read_hdf5(
            inputs,
            groups=[f"/data/demo_{index}" for index in range(args.episodes)],
            datasets={
                "raw_action": "actions",
                "observation.images.image": "obs/agentview_rgb",
                "observation.images.wrist_image": "obs/eye_in_hand_rgb",
                "ee_state": "obs/ee_states",
                "gripper_state": "obs/gripper_states",
            },
            file_path_column="file_path",
            group_path_column="hdf5_group",
            cache_remote_files=args.cache_remote_files,
        )
        .map(partial(normalize_libero_row, fps=args.fps))
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
            codec=args.codec,
            transencoding_threads=args.threads,
            max_video_prepare_in_flight=args.max_video_prepare_in_flight,
        )
    )

    if args.cloud:
        pipeline.launch_cloud(
            name=name,
            num_workers=args.workers,
            cpus_per_worker=args.cpus,
            mem_mb_per_worker=args.mem_mb,
            sync_local_dependencies=args.sync_local_dependencies,
            extra_dependencies=DEFAULT_CLOUD_DEPENDENCIES,
            secrets=mdr.Secrets.env(name=args.secret_env, keys=["HF_TOKEN"]),
        )
    else:
        pipeline.launch_local(name=name, num_workers=args.workers)


if __name__ == "__main__":
    main()
