from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterator, cast

import av
import fsspec
import numpy as np
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.pipeline import (
    LeRobotStatsConfig as PipelineLeRobotStatsConfig,
)
from refiner.pipeline import (
    LeRobotVideoConfig as PipelineLeRobotVideoConfig,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sinks.lerobot import (
    LeRobotMetaReduceSink,
    LeRobotWriterConfig,
    LeRobotWriterSink,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_writer import (
    LeRobotVideoWriter,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    reset_opened_video_source_cache,
)
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.context import RunHandle, set_active_run_context
from refiner.worker.lifecycle import RuntimeLifecycle

_ALOHA_REPO_IDS = (
    "macrodata/aloha_static_battery_ep000_004",
    "macrodata/aloha_static_battery_ep005_009",
)


@pytest.fixture(autouse=True)
def _reset_opened_source_cache() -> Iterator[None]:
    reset_opened_video_source_cache()
    yield
    reset_opened_video_source_cache()


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
        "observation.images.main": mdr.VideoFile(
            str(video_path),
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
    ).write_lerobot(str(out_root))
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
    ).write_lerobot(str(out_root))

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


def test_lerobot_video_writer_reuses_opened_remux_source_for_same_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    memfs = fsspec.filesystem("memory")
    uri = "memory://lerobot-remux-reuse/episode.mp4"
    with src_video.open("rb") as src, memfs.open(uri, "wb") as dst:
        dst.write(src.read())

    remux_module = __import__(
        "refiner.pipeline.sinks.lerobot._lerobot_video_remux",
        fromlist=["av"],
    )
    original_av_open = remux_module.av.open
    read_open_calls = 0

    def _counting_av_open(*args, **kwargs):
        nonlocal read_open_calls
        if kwargs.get("mode") == "r":
            read_open_calls += 1
        return original_av_open(*args, **kwargs)

    monkeypatch.setattr(remux_module.av, "open", _counting_av_open)

    writer = LeRobotVideoWriter(
        folder=mdr.DataFolder.resolve(str(tmp_path / "out")),
        chunk_key="000",
        video_key="observation.images.main",
        video_config=mdr.LeRobotVideoConfig(),
        stats_config=mdr.LeRobotStatsConfig(),
        default_fps=10,
        video_bytes_limit=1024 * 1024,
    )
    asyncio.run(
        writer.write_video(
            mdr.VideoFile(uri, from_timestamp_s=0.0, to_timestamp_s=0.3),
            episode_index=0,
        )
    )
    asyncio.run(
        writer.write_video(
            mdr.VideoFile(uri, from_timestamp_s=0.3, to_timestamp_s=0.6),
            episode_index=1,
        )
    )
    writer.finalize()

    assert read_open_calls == 1


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
        video_files_size_in_mb=1,
        video_config=mdr.LeRobotVideoConfig(encoder_threads=1, decoder_threads=1),
        stats_config=mdr.LeRobotStatsConfig(sample_stride=2, quantile_bins=64),
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
    ).write_lerobot(str(out_root))

    stats = pipeline.launch_local(
        name="lerobot-task-index-stable",
        num_workers=1,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    tasks = pq.read_table(out_root / "meta" / "tasks.parquet")
    assert tasks.column("task").to_pylist() == ["place", "pick"]
    assert tasks.column("task_index").to_pylist() == [0, 1]


class _FinalizedWorkersRuntime:
    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        assert stage_index == 0
        return [FinalizedShardWorker(shard_id="shard-1", worker_id="2")]


def test_write_lerobot_stage2_keeps_only_finalized_worker_outputs(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "cleanup"
    config = LeRobotWriterConfig(output=str(out_root))
    row = DictRow(
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
        shard_id="shard-1",
    )

    for worker_id, values in [("1", [1.0, 2.0]), ("2", [9.0, 10.0])]:
        writer = LeRobotWriterSink(config)
        runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime())
        worker_row = row.update(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "observation.state": [values[0]],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "observation.state": [values[1]],
                    },
                ]
            }
        )
        with set_active_run_context(
            run_handle=RunHandle(job_id="job", stage_index=0, worker_id=worker_id),
            runtime_lifecycle=runtime,
        ):
            writer.write_block([worker_row])
            writer.on_shard_complete("shard-1")

    reducer = LeRobotMetaReduceSink(config=config)
    runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime())
    with set_active_run_context(
        run_handle=RunHandle(job_id="job", stage_index=1, worker_id="local"),
        runtime_lifecycle=runtime,
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    worker_1 = RunHandle.worker_token_for("1")
    worker_2 = RunHandle.worker_token_for("2")
    assert not (out_root / "meta" / f"chunk-shard-1__w{worker_1}").exists()
    assert not (out_root / "meta" / f"chunk-shard-1__w{worker_2}").exists()
    assert not (out_root / "data" / f"chunk-shard-1__w{worker_1}").exists()
    assert (out_root / "data" / f"chunk-shard-1__w{worker_2}").exists()
    table = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert table.column("data/chunk_index").to_pylist() == [f"shard-1__w{worker_2}"]


def test_hub_aloha_merge_uses_remux_and_preserves_episode_count(tmp_path: Path) -> None:
    out_root = tmp_path / "hub-aloha-merge"
    source_roots = [f"hf://datasets/{repo_id}" for repo_id in _ALOHA_REPO_IDS]
    stats = (
        mdr.read_lerobot(source_roots)
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-hub-aloha-merge",
            num_workers=1,
            workdir=str(tmp_path / "workdir-hub-aloha-merge"),
        )
    )
    assert stats.failed == 0

    with (out_root / "meta" / "info.json").open("r", encoding="utf-8") as fh:
        info_json = json.load(fh)
    assert info_json["total_episodes"] == 10
    for video_key in [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_low",
        "observation.images.cam_right_wrist",
    ]:
        assert info_json["features"][video_key]["info"]["video.codec"] == "av1"
