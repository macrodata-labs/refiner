from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.egocentric.types import HandSide, HaworResult, as_transform_series

_HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def export_rerun(
    *,
    source_column: str = "hawor",
    output_root: str | Path = "rerun",
    output_column: str = "rerun_path",
    video_path_column: str | None = None,
    app_id: str = "refiner_egocentric_hawor",
    hands: tuple[HandSide, ...] = ("left", "right"),
) -> Any:
    """Return a row mapper that exports a HaWoR reconstruction as a Rerun file."""

    if not source_column:
        raise ValueError("source_column cannot be empty")
    if not output_column:
        raise ValueError("output_column cannot be empty")
    if not app_id:
        raise ValueError("app_id cannot be empty")
    if not hands:
        raise ValueError("hands cannot be empty")

    @describe_builtin(
        "robotics.egocentric:export_rerun",
        source_column=source_column,
        output_root=str(output_root),
        output_column=output_column,
        video_path_column=video_path_column,
        app_id=app_id,
        hands=hands,
    )
    def _export(row: Row) -> Row:
        payload = row[source_column]
        if not isinstance(payload, dict):
            raise ValueError(f"{source_column} must contain a HaWoR result mapping")
        result = HaworResult.from_mapping(payload)
        output_path = _row_output_path(
            row=row,
            result=result,
            output_root=Path(output_root),
        )
        export_hawor_rerun(
            result,
            output_path=output_path,
            video_path=None
            if video_path_column is None
            else row.get(video_path_column),
            app_id=app_id,
            hands=hands,
        )
        if row.shard_id is not None:
            row.log_throughput("rerun_recordings_exported", 1, unit="recordings")
        return row.update({output_column: str(output_path)})

    return _export


def export_hawor_rerun(
    result: HaworResult,
    *,
    output_path: str | Path,
    video_path: str | Path | None = None,
    app_id: str = "refiner_egocentric_hawor",
    hands: tuple[HandSide, ...] = ("left", "right"),
) -> Path:
    """Write a `.rrd` visualization for a normalized HaWoR result."""

    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "export_rerun requires the optional 'rerun-sdk' package. "
            "Install it in the runtime that exports visualizations."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(app_id)
    rr.save(str(output))
    video_size = _log_video(rr, video_path, result) if video_path is not None else None
    _log_camera_trajectory(rr, result)
    for side in hands:
        _log_hand_trajectory(rr, result, side)
    _log_per_frame(rr, result, hands=hands, video_size=video_size)
    return output


def _log_video(
    rr: Any,
    video_path: str | Path,
    result: HaworResult,
) -> tuple[int, int] | None:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video_path does not exist: {path}")

    video_asset = rr.AssetVideo(path=path)
    rr.log("video", video_asset, static=True)

    frame_timestamps_ns = np.asarray(
        [round(timestamp * 1_000_000_000) for timestamp in result.timestamps],
        dtype=np.int64,
    )
    rr.send_columns(
        "video",
        indexes=[
            rr.TimeColumn("frame", sequence=np.arange(len(frame_timestamps_ns))),
            rr.TimeColumn("time", duration=result.timestamps),
        ],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
    )
    return _video_size(path)


def _video_size(video_path: Path) -> tuple[int, int] | None:
    try:
        import av
    except ImportError:
        return None

    with av.open(str(video_path)) as container:
        stream = next((item for item in container.streams.video), None)
        if stream is None:
            return None
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _log_camera_trajectory(rr: Any, result: HaworResult) -> None:
    transforms = result.camera.get("T_world_camera")
    if transforms is None:
        return
    camera = as_transform_series(transforms, name="camera.T_world_camera")
    positions = camera[:, :3, 3]
    rr.log("world/camera/trajectory", rr.LineStrips3D([positions]))
    rr.log("world/camera/poses", rr.Points3D(positions, radii=0.01))


def _log_hand_trajectory(rr: Any, result: HaworResult, side: HandSide) -> None:
    hand = result.hand(side)
    if hand is None or "T_world_wrist" not in hand:
        return
    wrist = as_transform_series(
        hand["T_world_wrist"],
        name=f"{side}_hand.T_world_wrist",
    )
    positions = wrist[:, :3, 3]
    rr.log(f"world/{side}_hand/wrist_trajectory", rr.LineStrips3D([positions]))
    rr.log(f"world/{side}_hand/wrist_points", rr.Points3D(positions, radii=0.015))


def _log_per_frame(
    rr: Any,
    result: HaworResult,
    *,
    hands: tuple[HandSide, ...],
    video_size: tuple[int, int] | None,
) -> None:
    camera = None
    if video_size is not None and "T_world_camera" in result.camera:
        camera = as_transform_series(
            result.camera["T_world_camera"],
            name="camera.T_world_camera",
        )
    for index, timestamp in enumerate(result.timestamps):
        rr.set_time("frame", sequence=index)
        rr.set_time("time", duration=timestamp)
        for side in hands:
            hand = result.hand(side)
            if hand is None:
                continue
            _log_hand_frame(rr, hand, side=side, index=index)
            if camera is not None and video_size is not None:
                _log_video_hand_overlay(
                    rr,
                    result,
                    hand,
                    side=side,
                    index=index,
                    camera=camera,
                    video_size=video_size,
                )


def _log_hand_frame(
    rr: Any, hand: dict[str, Any], *, side: HandSide, index: int
) -> None:
    if "T_world_wrist" in hand:
        wrist = as_transform_series(
            hand["T_world_wrist"],
            name=f"{side}_hand.T_world_wrist",
        )
        rr.log(
            f"world/{side}_hand/current_wrist",
            rr.Points3D([wrist[index, :3, 3]], radii=0.025),
        )
    if "joints_world" in hand:
        joints = np.asarray(hand["joints_world"], dtype=np.float64)
        if joints.ndim == 3 and joints.shape[0] > index and joints.shape[2] == 3:
            rr.log(f"world/{side}_hand/joints", rr.Points3D(joints[index], radii=0.01))
            if joints.shape[1] >= 21:
                rr.log(
                    f"world/{side}_hand/skeleton",
                    rr.LineStrips3D(
                        [
                            [joints[index, start], joints[index, end]]
                            for start, end in _HAND_EDGES
                        ]
                    ),
                )
    if "confidence" in hand:
        confidence = hand["confidence"]
        if len(confidence) > index:
            rr.log(
                f"metrics/{side}_hand/confidence",
                rr.Scalars(float(confidence[index])),
            )


def _log_video_hand_overlay(
    rr: Any,
    result: HaworResult,
    hand: dict[str, Any],
    *,
    side: HandSide,
    index: int,
    camera: np.ndarray,
    video_size: tuple[int, int],
) -> None:
    if "joints_world" not in hand or camera.shape[0] <= index:
        return
    joints_world = np.asarray(hand["joints_world"], dtype=np.float64)
    if joints_world.ndim != 3 or joints_world.shape[0] <= index:
        return
    points = _project_world_points_to_video(
        joints_world[index],
        camera[index],
        img_focal=result.metadata.get("img_focal") if result.metadata else None,
        video_size=video_size,
    )
    if points.size == 0:
        return
    color = [80, 220, 120, 255] if side == "left" else [255, 120, 80, 255]
    rr.log(
        f"video/{side}_hand/joints",
        rr.Points2D(points, radii=4.0, colors=color),
    )
    if points.shape[0] >= 21:
        rr.log(
            f"video/{side}_hand/skeleton",
            rr.LineStrips2D(
                [
                    [points[start], points[end]]
                    for start, end in _HAND_EDGES
                    if start < points.shape[0] and end < points.shape[0]
                ],
                radii=2.0,
                colors=color,
            ),
        )


def _project_world_points_to_video(
    points_world: np.ndarray,
    t_world_camera: np.ndarray,
    *,
    img_focal: Any,
    video_size: tuple[int, int],
) -> np.ndarray:
    width, height = video_size
    focal = float(img_focal) if img_focal is not None else float(max(width, height))
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 2), dtype=np.float64)
    world_to_camera = np.linalg.inv(t_world_camera)
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    points_camera = (world_to_camera @ points_h.T).T[:, :3]
    z = points_camera[:, 2]
    valid = z > 1e-6
    projected = np.empty((int(valid.sum()), 2), dtype=np.float64)
    projected[:, 0] = focal * (points_camera[valid, 0] / z[valid]) + width / 2.0
    projected[:, 1] = focal * (points_camera[valid, 1] / z[valid]) + height / 2.0
    return projected


def _row_output_path(
    *,
    row: Row,
    result: HaworResult,
    output_root: Path,
) -> Path:
    identity = row.get("file_path") or row.get("video_path") or result.timestamps
    digest = hashlib.sha1(str(identity).encode("utf-8")).hexdigest()[:16]
    return output_root / f"hawor-{digest}.rrd"


__all__ = [
    "export_hawor_rerun",
    "export_rerun",
]
