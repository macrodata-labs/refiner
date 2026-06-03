from __future__ import annotations

# Cloud usage from a clean environment:
# uv run --no-project \
#   --with "macrodata-refiner[hf,video,egocentric]" \
#   --with "ego-vision[models]==0.1.15" \
#   examples/egocentric_hand_tracking_lerobot.py run \
#   --output hf://buckets/macrodata/test_bucket/egocentric-hand-tracking \
#   --cloud

import argparse
import json
from typing import Any

import numpy as np

import refiner as mdr


DEFAULT_VIDEO = (
    "https://prodtlkcsafiles.blob.core.windows.net/ego-centric/"
    "5_Pouring%20liquids_Video%20Pw.mp4"
)


def create_mano_actions(row: Any) -> Any:
    from egovision.pipelines import to_mano_actions

    hand_tracking = _hand_tracking(row)
    mano_actions = to_mano_actions(hand_tracking)
    actions, valid = _wrist_mano_action_array(mano_actions)
    return (
        row.with_actions(actions)
        .with_observation("egovision.wrist_mano_delta_valid", valid)
        .update(
            {"egovision_hand_tracking_metadata": json.dumps(_json_ready(hand_tracking))}
        )
        .drop("hand_tracking")
    )


def create_joint_actions(row: Any) -> Any:
    from egovision.pipelines import to_joint_actions

    hand_tracking = _hand_tracking(row)
    joint_actions = to_joint_actions(hand_tracking)
    actions, valid = _joint_action_array(joint_actions)
    return (
        row.with_actions(actions)
        .with_observation("egovision.joint_delta_valid", valid)
        .update(
            {"egovision_hand_tracking_metadata": json.dumps(_json_ready(hand_tracking))}
        )
        .drop("hand_tracking")
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
            ),
            batch_size=args.batch_size,
        )
        .map(create_mano_actions)
        # To train on world-space joint deltas instead, swap the line above for:
        # .map(create_joint_actions)
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


def _wrist_mano_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_transform = np.asarray(side_action["delta_T_world_wrist"]).reshape(-1, 16)
        delta_pose = np.asarray(side_action["delta_mano_pose"]).reshape(
            len(delta_transform), -1
        )
        count = min(len(delta_transform), len(delta_pose))
        side_arrays.append(
            np.concatenate(
                [delta_transform[:count], delta_pose[:count]],
                axis=1,
            )
        )
        side_valid.append(np.asarray(side_action["valid"], dtype=bool)[:count])
    count = min(len(values) for values in side_arrays)
    return (
        np.concatenate([values[:count] for values in side_arrays], axis=1).tolist(),
        np.stack([values[:count] for values in side_valid], axis=1).tolist(),
    )


def _joint_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    side_arrays = []
    side_valid = []
    for side in ("left", "right"):
        side_action = actions.get(side, {})
        delta_joints = np.asarray(side_action["delta_joints_world"])
        delta_joints = delta_joints.reshape(len(delta_joints), -1)
        side_arrays.append(delta_joints)
        side_valid.append(
            np.asarray(side_action["valid"], dtype=bool)[: len(delta_joints)]
        )
    count = min(len(values) for values in side_arrays)
    return (
        np.concatenate([values[:count] for values in side_arrays], axis=1).tolist(),
        np.stack([values[:count] for values in side_valid], axis=1).tolist(),
    )


def _hand_tracking(row: Any) -> dict[str, Any]:
    hand_tracking = dict(row["hand_tracking"])
    hand_tracking.pop("relative_actions", None)
    return hand_tracking


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ego-vision hand tracking, save wrist+MANO relative actions as "
            "LeRobot actions, and store full ego-vision metadata at episode level."
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
