from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

import refiner as mdr
from refiner.media import hydrate_media
from refiner.pipeline import (
    LeRobotStatsConfig as PipelineLeRobotStatsConfig,
    LeRobotVideoConfig as PipelineLeRobotVideoConfig,
)


def _write_video(path: Path, *, fps: int = 10, frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"

        for idx in range(frames):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 10
            image[..., 1] = 255 - idx * 10
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


def _write_large_video(
    path: Path,
    *,
    fps: int = 15,
    frames: int = 240,
    width: int = 640,
    height: int = 360,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for idx in range(frames):
            image = np.full((height, width, 3), 64 + (idx % 192), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


def _episode(
    *,
    episode_index: int,
    task: str,
    video_path: Path,
    from_ts: float,
    to_ts: float,
    values: list[float],
) -> dict:
    return {
        "episode_index": episode_index,
        "task": task,
        "tasks": [task],
        "frames": [
            {
                "frame_index": i,
                "timestamp": float(i) / 10.0,
                "observation.state": [v],
            }
            for i, v in enumerate(values)
        ],
        "observation.images.main": mdr.Video(
            media=mdr.MediaFile(str(video_path)),
            from_timestamp_s=from_ts,
            to_timestamp_s=to_ts,
        ),
        "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
    }


def test_lerobot_configs_export_from_pipeline() -> None:
    assert PipelineLeRobotVideoConfig is mdr.LeRobotVideoConfig
    assert PipelineLeRobotStatsConfig is mdr.LeRobotStatsConfig


def test_write_lerobot_is_deferred_and_roundtrips(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "out"
    pipeline = mdr.from_items(
        [
            _episode(
                episode_index=0,
                task="pick",
                video_path=src_video,
                from_ts=0.0,
                to_ts=0.3,
                values=[0.0, 2.0],
            ),
            _episode(
                episode_index=1,
                task="place",
                video_path=src_video,
                from_ts=0.3,
                to_ts=0.6,
                values=[4.0, 6.0],
            ),
        ],
        items_per_shard=1,
    ).write_lerobot(str(out_root), overwrite=True)
    assert not (out_root / "meta" / "info.json").exists()

    stats = pipeline.launch_local(
        name="lerobot-deferred-roundtrip",
        num_workers=1,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    assert (out_root / "meta" / "info.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert (out_root / "meta" / "tasks.parquet").exists()

    with (out_root / "meta" / "info.json").open("r", encoding="utf-8") as fh:
        info_json = json.load(fh)
    assert info_json["features"]["observation.images.main"] == {
        "dtype": "video",
        "shape": [3, 16, 16],
        "names": ["channels", "height", "width"],
        "info": {
            "video.fps": 10,
            "video.height": 16,
            "video.width": 16,
            "video.channels": 3,
            "video.codec": "mpeg4",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    with (out_root / "meta" / "stats.json").open("r", encoding="utf-8") as fh:
        stats_json = json.load(fh)
    assert stats_json["observation.state"]["mean"] == [3.0]
    assert "observation.images.main" in stats_json

    out_rows = mdr.read_lerobot(str(out_root)).materialize()
    assert [row["task"] for row in out_rows] == ["pick", "place"]


def test_write_lerobot_launch_local_runs_stage1_then_stage2(tmp_path: Path) -> None:
    out_root = tmp_path / "local-launch"
    pipeline = mdr.from_items(
        [
            {
                "episode_index": 0,
                "task": "pick",
                "tasks": ["pick"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [1.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [2.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
            {
                "episode_index": 1,
                "task": "place",
                "tasks": ["place"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [3.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [4.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
        ],
        items_per_shard=1,
    ).write_lerobot(str(out_root), overwrite=True)

    stats = pipeline.launch_local(
        name="lerobot-two-stage-local",
        num_workers=2,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0
    assert stats.claimed >= 3
    assert (out_root / "meta" / "info.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert (out_root / "meta" / "tasks.parquet").exists()


def test_write_lerobot_accepts_decoded_videos(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "decoded"
    pipeline = (
        mdr.from_items(
            [
                _episode(
                    episode_index=0,
                    task="pick",
                    video_path=src_video,
                    from_ts=0.0,
                    to_ts=0.4,
                    values=[1.0, 2.0, 3.0, 4.0],
                ),
            ]
        )
        .map_async(hydrate_media("observation.images.main"))
        .write_lerobot(str(out_root), overwrite=True)
    )

    stats = pipeline.launch_local(
        name="lerobot-decoded-videos",
        num_workers=1,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    rows = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert len(rows) == 1


def test_lerobot_writer_rolls_video_file_when_size_limit_is_hit(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_large_video(
        src_video,
        fps=15,
        frames=900,
        width=1920,
        height=1080,
    )

    out_root = tmp_path / "video-rotate"
    pipeline = mdr.from_items(
        [
            _episode(
                episode_index=0,
                task="pick",
                video_path=src_video,
                from_ts=0.0,
                to_ts=10.0,
                values=[0.0, 2.0],
            ),
            _episode(
                episode_index=1,
                task="place",
                video_path=src_video,
                from_ts=0.0,
                to_ts=10.0,
                values=[4.0, 6.0],
            ),
        ],
        items_per_shard=2,
    ).write_lerobot(
        str(out_root),
        overwrite=True,
        video_files_size_in_mb=1,
        video=mdr.LeRobotVideoConfig(encoder_threads=1, decoder_threads=1),
        stats=mdr.LeRobotStatsConfig(sample_stride=2, quantile_bins=64),
    )

    stats = pipeline.launch_local(
        name="lerobot-video-rotation",
        num_workers=1,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    video_dirs = list((out_root / "videos" / "observation.images.main").glob("chunk-*"))
    video_files = [
        path for directory in video_dirs for path in sorted(directory.glob("*.mp4"))
    ]
    assert any(path.name == "file-001.mp4" for path in video_files)


def test_write_lerobot_preserves_stable_task_index_mapping(tmp_path: Path) -> None:
    out_root = tmp_path / "task-index"
    pipeline = mdr.from_items(
        [
            {
                "episode_index": 0,
                "task": "place",
                "tasks": ["place"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [1.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [2.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
            {
                "episode_index": 1,
                "task": "pick",
                "tasks": ["pick"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [3.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [4.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
        ],
        items_per_shard=10,
    ).write_lerobot(str(out_root), overwrite=True)

    stats = pipeline.launch_local(
        name="lerobot-task-index-stable",
        num_workers=1,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    tasks = pq.read_table(out_root / "meta" / "tasks.parquet")
    assert tasks.column("task").to_pylist() == ["place", "pick"]
    assert tasks.column("task_index").to_pylist() == [0, 1]
