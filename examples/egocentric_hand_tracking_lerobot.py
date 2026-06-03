from __future__ import annotations

# Cloud usage from a clean environment:
# uv run --no-project \
#   --with "macrodata-refiner[hf,video,robotics]" \
#   --with "ego-vision[models]==0.1.14" \
#   examples/egocentric_hand_tracking_lerobot.py run \
#   --output hf://buckets/macrodata/test_bucket/egocentric-hand-tracking \
#   --cloud

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import refiner as mdr
from refiner.io import DataFolder


DEFAULT_VIDEO = (
    "https://prodtlkcsafiles.blob.core.windows.net/ego-centric/"
    "5_Pouring%20liquids_Video%20Pw.mp4"
)


def egovision_config(egovision: Any) -> Any:
    return egovision.HandTrackingConfig(
        hand_reconstruction=egovision.HaworReconstructionConfig(),
        camera_pose_estimator=egovision.VggtOmegaConfig(),
        device="cuda",
    )


def add_egovision_outputs(row: Any) -> Any:
    from egovision.pipelines.hand_tracking_outputs import hand_tracking_outputs

    outputs = hand_tracking_outputs(row["hand_tracking"])
    return row.update(
        {
            "egovision_rerun_result_json": json.dumps(outputs["rerun_result"]),
            "world_joint_actions": outputs["world_joint_actions"],
            "wrist_mano_actions": outputs["wrist_mano_actions"],
        }
    )


def attach_actions_and_rerun_features(row: Any) -> Any:
    frame_count = _frame_count(row)
    wrist_mano, wrist_mano_valid = _wrist_mano_action_array(
        row["wrist_mano_actions"],
        frame_count=frame_count,
    )
    world_joints, world_joints_valid = _world_joint_action_array(
        row["world_joint_actions"],
        frame_count=frame_count,
    )

    return (
        row.with_actions(wrist_mano)
        .with_observation("egovision.wrist_mano_delta_action", wrist_mano)
        .with_observation("egovision.wrist_mano_delta_valid", wrist_mano_valid)
        .with_observation("egovision.world_joint_delta_action", world_joints)
        .with_observation("egovision.world_joint_delta_valid", world_joints_valid)
        .drop("hand_tracking", "world_joint_actions", "wrist_mano_actions")
    )


def run(args: argparse.Namespace) -> None:
    pipeline = (
        mdr.read_videos(args.videos, file_path_column="video")
        .to_robot_rows(
            episode_id_key="video",
            fps=args.fps,
            robot_type="human_egocentric_hands",
            video_keys={"video": "video"},
        )
        .batch_map(
            mdr.robotics.track_hands(
                video_key="video",
                output_key="hand_tracking",
                config_factory=egovision_config,
            ),
            batch_size=args.batch_size,
        )
        .map(add_egovision_outputs)
        .map(attach_actions_and_rerun_features)
        .write_lerobot(args.output)
    )

    if args.cloud:
        pipeline.launch_cloud(
            name=args.name,
            num_workers=args.num_workers,
            mem_mb_per_worker=args.mem_mb_per_worker,
            gpu=mdr.GPU(count=1, type=args.gpu),
            secrets=mdr.Secrets.env(keys=("HF_TOKEN",)),
        )
    else:
        pipeline.launch_local(
            name=args.name,
            num_workers=args.num_workers,
            gpu=mdr.GPU(count=1, type=args.gpu) if args.gpu else None,
        )


def materialize_rerun_json(args: argparse.Namespace) -> None:
    folder = DataFolder.resolve(args.lerobot)
    files = [path for path in folder.find("meta/episodes") if path.endswith(".parquet")]
    if not files:
        raise ValueError(f"no LeRobot episode parquet files found in {args.lerobot}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for file_index, rel_path in enumerate(sorted(files)):
        with folder.open(rel_path, mode="rb") as handle:
            table = pq.read_table(handle)
        if "egovision_rerun_result_json" not in table.column_names:
            raise ValueError(
                f"{rel_path} is missing egovision_rerun_result_json; "
                "run this example before materializing Rerun JSON"
            )
        episode_ids = (
            table.column("episode_id").to_pylist()
            if "episode_id" in table.column_names
            else [None] * table.num_rows
        )
        payloads = table.column("egovision_rerun_result_json").to_pylist()
        for row_index, (episode_id, payload) in enumerate(
            zip(episode_ids, payloads, strict=True)
        ):
            name = _safe_name(str(episode_id or f"episode-{file_index}-{row_index}"))
            path = output_dir / f"{name}.json"
            path.write_text(str(payload), encoding="utf-8")
            written += 1
    print(f"wrote {written} Rerun result JSON file(s) to {output_dir}")


def _frame_count(row: Any) -> int:
    result = json.loads(row["egovision_rerun_result_json"])
    camera = result.get("camera_trajectory")
    if isinstance(camera, list) and camera:
        return len(camera)
    return int(row.num_frames)


def _wrist_mano_action_array(
    actions: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[list[list[float]], list[list[bool]]]:
    pose_width = max(
        _pose_width(
            np.asarray(
                actions.get(side, {}).get("delta_mano_pose", []),
                dtype=np.float64,
            )
        )
        for side in ("left", "right")
    )
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_transform = np.asarray(
            side_action.get("delta_T_world_wrist", []),
            dtype=np.float64,
        ).reshape(-1, 16)
        delta_pose = np.asarray(
            side_action.get("delta_mano_pose", []),
            dtype=np.float64,
        )
        if delta_pose.ndim == 1:
            delta_pose = delta_pose.reshape(-1, 1)
        valid = np.asarray(side_action.get("valid", []), dtype=bool).reshape(-1, 1)
        count = min(len(delta_transform), len(delta_pose))
        side_array = np.zeros((count, 16 + pose_width), dtype=np.float64)
        if count:
            side_array[:, :16] = delta_transform[:count]
            side_array[:, 16 : 16 + min(pose_width, delta_pose.shape[1])] = delta_pose[
                :count, :pose_width
            ]
        side_arrays.append(_pad_frames(side_array, frame_count=frame_count))
        side_valid.append(_pad_frames(valid, frame_count=frame_count, dtype=bool))
    return (
        np.concatenate(side_arrays, axis=1).tolist(),
        np.concatenate(side_valid, axis=1).tolist(),
    )


def _world_joint_action_array(
    actions: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[list[list[float]], list[list[bool]]]:
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_joints = np.asarray(
            side_action.get("delta_joints_world", []),
            dtype=np.float64,
        ).reshape(-1, 63)
        valid = np.asarray(side_action.get("valid", []), dtype=bool).reshape(-1, 1)
        side_arrays.append(_pad_frames(delta_joints, frame_count=frame_count))
        side_valid.append(_pad_frames(valid, frame_count=frame_count, dtype=bool))
    return (
        np.concatenate(side_arrays, axis=1).tolist(),
        np.concatenate(side_valid, axis=1).tolist(),
    )


def _pad_frames(
    values: np.ndarray,
    *,
    frame_count: int,
    dtype: type = float,
) -> np.ndarray:
    width = int(values.shape[1]) if values.ndim == 2 else 1
    output = np.zeros((frame_count, width), dtype=dtype)
    if len(values):
        count = min(frame_count, len(values))
        output[:count] = values[:count]
    return output


def _pose_width(values: np.ndarray) -> int:
    if values.ndim == 1:
        return 1 if len(values) else 0
    return int(values.shape[1]) if values.ndim == 2 else 0


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ego-vision hand tracking, save wrist+MANO relative actions as "
            "LeRobot actions, keep world-joint relative actions as an alternate "
            "observation, and store Rerun-ready result JSON at episode level."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("videos", nargs="?", default=DEFAULT_VIDEO)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--name", default="egocentric-hand-tracking-lerobot")
    run_parser.add_argument("--fps", type=float, default=30.0)
    run_parser.add_argument("--batch-size", type=int, default=2)
    run_parser.add_argument("--num-workers", type=int, default=1)
    run_parser.add_argument("--mem-mb-per-worker", type=int, default=32 * 1024)
    run_parser.add_argument("--gpu", default="h100")
    run_parser.add_argument("--cloud", action="store_true")
    run_parser.set_defaults(func=run)

    rerun_parser = subparsers.add_parser("materialize-rerun-json")
    rerun_parser.add_argument("lerobot")
    rerun_parser.add_argument("--output-dir", default="artifacts/egovision-rerun")
    rerun_parser.set_defaults(func=materialize_rerun_json)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
