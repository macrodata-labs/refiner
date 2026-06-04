from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np

import refiner as mdr

INPUT_DATASET = "toloka/HomER"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket/homer-hand-tracking"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DATASET = f"{OUTPUT_ROOT}/{RUN_ID}"

FPS = 30.0
BATCH_SIZE = 2
MANO_POSE_WIDTH = 96
JOINT_STATE_WIDTH = 63


@dataclass(frozen=True)
class FixedFpsVideoSource:
    source: Any
    fps: float

    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "FixedFpsVideoSource":
        return FixedFpsVideoSource(
            self.source.clipped(
                from_timestamp_s=from_timestamp_s,
                to_timestamp_s=to_timestamp_s,
            ),
            fps=self.fps,
        )

    def iter_frames(self) -> AsyncIterator[Any]:
        return self.source.iter_frames()

    def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]:
        return self.source.iter_numpy_frames()

    def iter_frame_windows(
        self,
        *,
        offsets: Any,
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[Any]:
        return self.source.iter_frame_windows(
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )

    async def write_to(
        self,
        writer: Any,
        *,
        frame_observer: Any = None,
        force_transcode: bool = False,
    ) -> Any:
        frames = [frame async for frame in self.source.iter_numpy_frames()]
        return await writer.write_frame_array_video(
            mdr.video.VideoFrameSequence(
                frames,
                fps=self.fps,
                frame_count=len(frames),
            ),
            frame_observer=frame_observer,
        )


def create_mano_actions(row: Any) -> Any:
    hand_tracking = row["hand_tracking"]
    actions, valid = _wrist_mano_action_array(
        _egovision_pipelines().to_mano_actions(hand_tracking)
    )
    states = _wrist_mano_state_arrays(hand_tracking, len(actions))
    return (
        row.with_actions(actions)
        .with_observation("left_hand_state", states["left"])
        .with_observation("right_hand_state", states["right"])
        .with_observation("egovision.wrist_mano_delta_valid", valid)
        .update({"egovision_hand_tracking_metadata": _json_dumps(hand_tracking)})
        .drop("hand_tracking")
    )


def create_joint_actions(row: Any) -> Any:
    hand_tracking = row["hand_tracking"]
    actions, valid = _joint_action_array(
        _egovision_pipelines().to_joint_actions(hand_tracking)
    )
    states = _joint_state_arrays(hand_tracking, len(actions))
    return (
        row.with_actions(actions)
        .with_observation("left_hand_state", states["left"])
        .with_observation("right_hand_state", states["right"])
        .with_observation("egovision.joint_delta_valid", valid)
        .update({"egovision_hand_tracking_metadata": _json_dumps(hand_tracking)})
        .drop("hand_tracking")
    )


def force_video_fps(row: Any) -> Any:
    return row.with_video("video", FixedFpsVideoSource(row.videos["video"], fps=FPS))


def _egovision_pipelines() -> Any:
    return cast(Any, importlib.import_module("egovision.pipelines"))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=lambda item: item.tolist())


def _wrist_mano_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    count = _max_action_count(actions, "delta_T_world_wrist")
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
        side_valid.append(_valid_array(side_action, count, side_count))
    return (
        np.concatenate(side_arrays, axis=1).tolist(),
        np.stack(side_valid, axis=1).tolist(),
    )


def _joint_action_array(
    actions: dict[str, Any],
) -> tuple[list[list[float]], list[list[bool]]]:
    count = _max_action_count(actions, "delta_joints_world")
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
        side_valid.append(_valid_array(side_action, count, side_count))
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
        hand = hand_tracking["hands_world"][side]
        transforms = np.asarray(hand["T_world_wrist"], dtype=np.float64)
        mano_pose = np.asarray(hand["mano_pose"], dtype=np.float64).reshape(
            len(transforms), -1
        )
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
            hand_tracking["hands_world"][side]["joints_world"],
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


def _max_action_count(actions: dict[str, Any], key: str) -> int:
    return max(len(actions.get(side, {}).get(key, [])) for side in ("left", "right"))


def _valid_array(
    side_action: dict[str, Any], count: int, side_count: int
) -> np.ndarray:
    valid = np.zeros(count, dtype=bool)
    valid[:side_count] = np.asarray(side_action.get("valid", []), dtype=bool)[
        :side_count
    ]
    return valid


(
    mdr.read_hf_dataset(
        INPUT_DATASET,
        split="train",
        columns_to_read=("video_id", "video_url", "description"),
        dtypes={"video_url": mdr.datatype.video_path()},
    )
    .to_robot_rows(
        episode_id_key="video_id",
        task_key="description",
        fps=FPS,
        robot_type="human_hand_tracking",
        video_keys={"video": "video_url"},
    )
    .batch_map(
        mdr.robotics.track_hands(video_key="video", output_key="hand_tracking"),
        batch_size=BATCH_SIZE,
    )
    .map(create_mano_actions)
    # To train on world-space joint deltas instead, swap the line above for:
    # .map(create_joint_actions)
    .map(force_video_fps)
    .write_lerobot(OUTPUT_DATASET, codec="libx264")
    .launch_cloud(
        name="hand-tracking",
        num_workers=1,
        mem_mb_per_worker=32 * 1024,
        gpu=mdr.GPU(count=1, type="h100"),
        extra_dependencies=("ego-vision[models]==0.1.25",),
        secrets=mdr.Secrets.env(keys=("HF_TOKEN",)),
    )
)
