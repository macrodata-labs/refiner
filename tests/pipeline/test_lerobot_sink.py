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
from refiner.io import DataFile, DataFolder
from refiner.media.video.remux import reset_opened_video_source_cache
from refiner.media.video.transcode import VideoTranscodeConfig
from refiner.media.video.writer import VideoStreamWriter
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sinks.lerobot import LeRobotWriterSink
from refiner.pipeline.sinks.lerobot_reducer import LeRobotMetaReduceSink
from refiner.robotics.lerobot_format import (
    LEROBOT_TASKS,
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotStatsFile,
    LeRobotTasks,
)
from refiner.robotics.lerobot_format.metadata.stats import _estimate_sample_stride
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


def _sampled_frame_count(
    path: Path,
    *,
    from_ts: float,
    to_ts: float | None,
) -> int:
    frame_count = 0

    with av.open(str(path), mode="r") as container:
        stream = next(
            (item for item in container.streams if item.type == "video"), None
        )
        if stream is None:
            raise ValueError(f"Video source has no video stream: {path}")

        for frame in container.decode(stream):
            if not isinstance(frame, av.VideoFrame):
                continue
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp + 1e-6 < from_ts:
                continue
            if to_ts is not None and timestamp - 1e-6 >= to_ts:
                break
            frame_count += 1

    sample_stride = _estimate_sample_stride(frame_count)
    return len(range(0, frame_count, sample_stride))


def _episode(
    *,
    episode_index: int,
    task: str,
    task_index: int,
    video_path: Path,
    from_ts: float,
    to_ts: float,
    values: list[float],
) -> dict:
    return {
        "episode_index": episode_index,
        "task": task,
        "frames": [
            {
                "frame_index": i,
                "timestamp": float(i) / 10.0,
                "task_index": task_index,
                "observation.state": [v],
            }
            for i, v in enumerate(values)
        ],
        "observation.images.main": mdr.VideoFile(
            DataFile.resolve(str(video_path)),
            from_timestamp_s=from_ts,
            to_timestamp_s=to_ts,
        ),
        LEROBOT_TASKS: {0: "pick", 1: "place"},
        "metadata": _metadata(),
    }


def _metadata(
    index_to_task: dict[int, str] | None = None,
) -> LeRobotMetadata:
    index_to_task = index_to_task or {0: "pick", 1: "place"}
    return LeRobotMetadata(
        info=LeRobotInfo(fps=10, robot_type="mockbot"),
        stats=LeRobotStatsFile.from_json_dict({}),
        tasks=LeRobotTasks(index_to_task),
    )


def _dummy_video_stats(*, count: int) -> dict[str, np.ndarray]:
    return {
        "min": np.zeros((3, 1, 1), dtype=np.float64),
        "max": np.ones((3, 1, 1), dtype=np.float64),
        "mean": np.full((3, 1, 1), 0.5, dtype=np.float64),
        "std": np.full((3, 1, 1), 0.1, dtype=np.float64),
        "count": np.array([count], dtype=np.int64),
        "q01": np.full((3, 1, 1), 0.1, dtype=np.float64),
        "q10": np.full((3, 1, 1), 0.2, dtype=np.float64),
        "q50": np.full((3, 1, 1), 0.5, dtype=np.float64),
        "q90": np.full((3, 1, 1), 0.8, dtype=np.float64),
        "q99": np.full((3, 1, 1), 0.9, dtype=np.float64),
    }


def test_write_lerobot_is_deferred_and_roundtrips(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "out"
    pipeline = mdr.from_items(
        [
            _episode(
                episode_index=0,
                task="pick",
                task_index=0,
                video_path=src_video,
                from_ts=0.0,
                to_ts=0.3,
                values=[0.0, 2.0],
            ),
            _episode(
                episode_index=1,
                task="place",
                task_index=1,
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
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.fps": 10,
            "video.codec": "mpeg4",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    with (out_root / "meta" / "stats.json").open("r", encoding="utf-8") as fh:
        stats_json = json.load(fh)
    assert stats_json["observation.state"]["mean"] == [3.0]
    assert "q01" in stats_json["observation.state"]
    assert "q99" in stats_json["observation.state"]
    assert "observation.images.main" in stats_json
    expected_video_count = _sampled_frame_count(
        src_video,
        from_ts=0.0,
        to_ts=0.3,
    ) + _sampled_frame_count(
        src_video,
        from_ts=0.3,
        to_ts=0.6,
    )
    assert stats_json["observation.images.main"]["count"] == [expected_video_count]

    out_rows = mdr.read_lerobot(str(out_root)).materialize()
    assert sorted(row["task"] for row in out_rows) == ["pick", "place"]


def test_write_lerobot_launch_local_runs_stage1_then_stage2(tmp_path: Path) -> None:
    out_root = tmp_path / "local-launch"
    pipeline = mdr.from_items(
        [
            {
                "episode_index": 0,
                "task": "pick",
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 0,
                        "observation.state": [1.0],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "task_index": 0,
                        "observation.state": [2.0],
                    },
                ],
                LEROBOT_TASKS: {0: "pick", 1: "place"},
                "metadata": _metadata(),
            },
            {
                "episode_index": 1,
                "task": "place",
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 1,
                        "observation.state": [3.0],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "task_index": 1,
                        "observation.state": [4.0],
                    },
                ],
                LEROBOT_TASKS: {0: "pick", 1: "place"},
                "metadata": _metadata(),
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
        "refiner.media.video.remux",
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

    writer = VideoStreamWriter(
        folder=DataFolder.resolve(str(tmp_path / "out")),
        stream_key="observation.images.main",
        transcode_config=VideoTranscodeConfig(),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4",
        output_context={"chunk_index": "000"},
    )
    asyncio.run(
        writer.write_video(
            mdr.VideoFile(
                DataFile.resolve(uri),
                from_timestamp_s=0.0,
                to_timestamp_s=0.3,
            ),
        )
    )
    asyncio.run(
        writer.write_video(
            mdr.VideoFile(
                DataFile.resolve(uri),
                from_timestamp_s=0.3,
                to_timestamp_s=0.6,
            ),
        )
    )
    writer.close()

    assert read_open_calls == 1


def test_write_lerobot_force_recompute_video_stats_ignores_source_video_stats(
    tmp_path: Path,
) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    row = _episode(
        episode_index=0,
        task="pick",
        task_index=0,
        video_path=src_video,
        from_ts=0.0,
        to_ts=0.3,
        values=[0.0, 2.0],
    )
    row.update(
        {
            f"stats/observation.images.main/{name}": value
            for name, value in _dummy_video_stats(count=999).items()
        }
    )

    out_root = tmp_path / "force-recompute"
    stats = (
        mdr.from_items([row])
        .write_lerobot(
            str(out_root),
            force_recompute_video_stats=True,
        )
        .launch_local(
            name="lerobot-force-recompute-video-stats",
            num_workers=1,
            workdir=str(tmp_path / "workdir-force-recompute"),
        )
    )
    assert stats.failed == 0

    with (out_root / "meta" / "stats.json").open("r", encoding="utf-8") as fh:
        stats_json = json.load(fh)

    expected_video_count = _sampled_frame_count(
        src_video,
        from_ts=0.0,
        to_ts=0.3,
    )
    assert stats_json["observation.images.main"]["count"] == [expected_video_count]
    assert stats_json["observation.images.main"]["count"] != [999]


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
                task_index=0,
                video_path=src_video,
                from_ts=0.0,
                to_ts=10.0,
                values=[0.0, 2.0],
            ),
            _episode(
                episode_index=1,
                task="place",
                task_index=1,
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
        transencoding_threads=1,
        quantile_bins=64,
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
                "task": "pick",
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 5,
                        "observation.state": [1.0],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "task_index": 5,
                        "observation.state": [2.0],
                    },
                ],
                LEROBOT_TASKS: {1: "place", 5: "pick"},
                "metadata": _metadata({1: "place", 5: "pick"}),
            },
            {
                "episode_index": 1,
                "task": "place",
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 1,
                        "observation.state": [3.0],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "task_index": 1,
                        "observation.state": [4.0],
                    },
                ],
                LEROBOT_TASKS: {1: "place", 5: "pick"},
                "metadata": _metadata({1: "place", 5: "pick"}),
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
    assert tasks.column("task_index").to_pylist() == [1, 5]
    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes.column("tasks").to_pylist() == [["pick"], ["place"]]
    assert episodes.column("task").to_pylist() == ["pick", "place"]


def test_write_lerobot_raises_on_unmapped_frame_task_index(tmp_path: Path) -> None:
    writer = LeRobotWriterSink(str(tmp_path / "out"))
    with pytest.raises(
        KeyError,
        match="7",
    ):
        writer.write_block(
            [
                DictRow(
                    {
                        "episode_index": 0,
                        "task": "pick",
                        "frames": [
                            {
                                "frame_index": 0,
                                "timestamp": 0.0,
                                "task_index": 7,
                                "observation.state": [1.0],
                            }
                        ],
                        LEROBOT_TASKS: {0: "pick", 1: "place"},
                        "metadata": _metadata(),
                    },
                    shard_id="shard-1",
                )
            ]
        )
        writer.close()


class _FinalizedWorkersRuntime:
    def __init__(
        self,
        rows: list[FinalizedShardWorker] | None = None,
    ) -> None:
        self._rows = rows or [FinalizedShardWorker(shard_id="shard-1", worker_id="2")]

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        assert stage_index == 0
        return self._rows


def test_write_lerobot_stage2_keeps_only_finalized_worker_outputs(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "cleanup"
    row = DictRow(
        {
            "episode_index": 0,
            "task": "pick",
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "task_index": 0,
                    "observation.state": [1.0],
                },
                {
                    "frame_index": 1,
                    "timestamp": 0.1,
                    "task_index": 0,
                    "observation.state": [2.0],
                },
            ],
            LEROBOT_TASKS: {0: "pick", 1: "place"},
            "metadata": _metadata(),
        },
        shard_id="shard-1",
    )

    for worker_id, values in [("1", [1.0, 2.0]), ("2", [9.0, 10.0])]:
        writer = LeRobotWriterSink(str(out_root))
        runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime())
        worker_row = row.update(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 0,
                        "observation.state": [values[0]],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 0.1,
                        "task_index": 0,
                        "observation.state": [values[1]],
                    },
                ],
            }
        )
        with set_active_run_context(
            run_handle=RunHandle(job_id="job", stage_index=0, worker_id=worker_id),
            runtime_lifecycle=runtime,
        ):
            writer.write_block([worker_row])
            writer.on_shard_complete("shard-1")

    reducer = LeRobotMetaReduceSink(output=str(out_root))
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
        assert info_json["features"][video_key]["video_info"]["video.codec"] == "av1"
