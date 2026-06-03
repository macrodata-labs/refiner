from __future__ import annotations

# Cloud usage from a clean environment:
# uv run --no-project \
#   --with "macrodata-refiner[hand_tracking]" \
#   examples/hand_tracking_lerobot.py \
#   --cloud

import argparse
import importlib
import json
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np

import refiner as mdr


DEFAULT_INPUT_DATASET = "toloka/HomER"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket/homer-hand-tracking"
DEFAULT_NAME = "hand-tracking-lerobot"
DEFAULT_FPS = 30.0
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 1
DEFAULT_MEM_MB_PER_WORKER = 32 * 1024
DEFAULT_GPU = "h100"
DEFAULT_MAX_EPISODES: int | None = None
MANO_POSE_WIDTH = 96
JOINT_STATE_WIDTH = 63


def create_mano_actions(row: Any) -> Any:
    hand_tracking = dict(row["hand_tracking"])
    hand_tracking.pop("relative_actions", None)
    egovision_pipelines = cast(Any, importlib.import_module("egovision.pipelines"))
    mano_actions = egovision_pipelines.to_mano_actions(hand_tracking)
    actions, valid = _wrist_mano_action_array(mano_actions)
    states = _wrist_mano_state_arrays(hand_tracking, len(actions))
    return (
        row.with_actions(actions)
        .with_observation("left_hand_state", states["left"])
        .with_observation("right_hand_state", states["right"])
        .with_observation("egovision.wrist_mano_delta_valid", valid)
        .update(
            {
                "egovision_hand_tracking_metadata": json.dumps(
                    hand_tracking, default=lambda value: value.tolist()
                )
            }
        )
        .drop("hand_tracking")
    )


def create_joint_actions(row: Any) -> Any:
    hand_tracking = dict(row["hand_tracking"])
    hand_tracking.pop("relative_actions", None)
    egovision_pipelines = cast(Any, importlib.import_module("egovision.pipelines"))
    joint_actions = egovision_pipelines.to_joint_actions(hand_tracking)
    actions, valid = _joint_action_array(joint_actions)
    states = _joint_state_arrays(hand_tracking, len(actions))
    return (
        row.with_actions(actions)
        .with_observation("left_hand_state", states["left"])
        .with_observation("right_hand_state", states["right"])
        .with_observation("egovision.joint_delta_valid", valid)
        .update(
            {
                "egovision_hand_tracking_metadata": json.dumps(
                    hand_tracking, default=lambda value: value.tolist()
                )
            }
        )
        .drop("hand_tracking")
    )


def _wrist_mano_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    count = max(
        len(actions.get(side, {}).get("delta_T_world_wrist", []))
        for side in ("left", "right")
    )
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_transform = np.asarray(
            side_action.get("delta_T_world_wrist", []),
            dtype=np.float64,
        )
        delta_pose = np.asarray(
            side_action.get("delta_mano_pose", []),
            dtype=np.float64,
        )
        side_count = min(len(delta_transform), len(delta_pose), count)
        side_array = np.full((count, 16 + MANO_POSE_WIDTH), np.nan)
        if side_count:
            side_array[:side_count] = np.concatenate(
                [
                    delta_transform[:side_count].reshape(side_count, 16),
                    delta_pose[:side_count].reshape(side_count, MANO_POSE_WIDTH),
                ],
                axis=1,
            )
        side_arrays.append(side_array)
        valid = np.zeros(count, dtype=bool)
        valid[:side_count] = np.asarray(
            side_action.get("valid", []),
            dtype=bool,
        )[:side_count]
        side_valid.append(valid)
    return (
        np.concatenate(side_arrays, axis=1).tolist(),
        np.stack(side_valid, axis=1).tolist(),
    )


def _joint_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    count = max(
        len(actions.get(side, {}).get("delta_joints_world", []))
        for side in ("left", "right")
    )
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_joints = np.asarray(
            side_action.get("delta_joints_world", []),
            dtype=np.float64,
        )
        side_count = min(len(delta_joints), count)
        side_array = np.full((count, JOINT_STATE_WIDTH), np.nan)
        if side_count:
            side_array[:side_count] = delta_joints[:side_count].reshape(
                side_count, JOINT_STATE_WIDTH
            )
        side_arrays.append(side_array)
        valid = np.zeros(count, dtype=bool)
        valid[:side_count] = np.asarray(
            side_action.get("valid", []),
            dtype=bool,
        )[:side_count]
        side_valid.append(valid)
    return (
        np.concatenate(side_arrays, axis=1).tolist(),
        np.stack(side_valid, axis=1).tolist(),
    )


def _wrist_mano_state_arrays(
    hand_tracking: dict[str, Any],
    count: int,
) -> dict[str, list[list[float]]]:
    states = {}
    for side in ("left", "right"):
        hand = _world_hand(hand_tracking, side)
        transforms = np.asarray(hand.get("T_world_wrist", []), dtype=np.float64)
        mano_pose = np.asarray(hand.get("mano_pose", []), dtype=np.float64)
        if len(mano_pose) and mano_pose.ndim == 1:
            mano_pose = mano_pose.reshape(len(mano_pose), -1)
        frame_count = min(count, len(transforms), len(mano_pose))
        state = np.full((count, 16 + MANO_POSE_WIDTH), np.nan)
        if frame_count:
            state[:frame_count] = np.concatenate(
                [
                    transforms[:frame_count].reshape(frame_count, 16),
                    mano_pose[:frame_count].reshape(frame_count, MANO_POSE_WIDTH),
                ],
                axis=1,
            )
        states[side] = state.tolist()
    return states


def _joint_state_arrays(
    hand_tracking: dict[str, Any],
    count: int,
) -> dict[str, list[list[float]]]:
    states = {}
    for side in ("left", "right"):
        joints = np.asarray(
            _world_hand(hand_tracking, side).get("joints_world", []),
            dtype=np.float64,
        )
        frame_count = min(count, len(joints))
        state = np.full((count, JOINT_STATE_WIDTH), np.nan)
        if frame_count:
            state[:frame_count] = joints[:frame_count].reshape(
                frame_count, JOINT_STATE_WIDTH
            )
        states[side] = state.tolist()
    return states


def _world_hand(hand_tracking: dict[str, Any], side: str) -> dict[str, Any]:
    hands_world = hand_tracking.get("hands_world", {})
    if isinstance(hands_world, dict):
        hand = hands_world.get(side)
        return hand if isinstance(hand, dict) else {}
    for hand in hands_world:
        if hand.get("handedness") == side:
            return hand
    return {}


def first_n_rows(count: int) -> Any:
    seen = 0

    def _keep(_: Any) -> bool:
        nonlocal seen
        if seen >= count:
            return False
        seen += 1
        return True

    return _keep


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read HomER videos from Hugging Face, run ego-vision hand tracking, "
            "and write the selected hand-action representation as LeRobot."
        )
    )
    parser.add_argument("--input-dataset", default=DEFAULT_INPUT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--mem-mb-per-worker",
        type=int,
        default=DEFAULT_MEM_MB_PER_WORKER,
    )
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"{DEFAULT_OUTPUT_ROOT}/{stamp}"

    pipeline = mdr.read_hf_dataset(
        args.input_dataset,
        split=args.split,
        columns_to_read=(
            "video_id",
            "video_url",
            "description",
        ),
        dtypes={"video_url": mdr.datatype.video_path()},
    )
    if args.max_episodes is not None:
        pipeline = pipeline.filter(first_n_rows(args.max_episodes))

    pipeline = (
        pipeline.to_robot_rows(
            episode_id_key="video_id",
            task_key="description",
            fps=args.fps,
            robot_type="human_hand_tracking",
            video_keys={"video": "video_url"},
        )
        .batch_map(
            mdr.robotics.track_hands(
                video_key="video",
                output_key="hand_tracking",
            ),
            batch_size=args.batch_size,
        )
        .map(create_mano_actions)
        # To train on world-space joint deltas instead, swap the line above for:
        # .map(create_joint_actions)
        .write_lerobot(output)
    )

    if args.cloud:
        pipeline.launch_cloud(
            name=args.name,
            num_workers=args.num_workers,
            mem_mb_per_worker=args.mem_mb_per_worker,
            gpu=mdr.GPU(count=1, type=args.gpu),
            extra_dependencies=("ego-vision[models]==0.1.19",),
            secrets=mdr.Secrets.env(keys=("HF_TOKEN",)),
        )
    else:
        pipeline.launch_local(
            name=args.name,
            num_workers=args.num_workers,
            gpu=mdr.GPU(count=1, type=args.gpu) if args.gpu else None,
        )


if __name__ == "__main__":
    main()
