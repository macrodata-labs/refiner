from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator, cast

import av
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.io import DataFile, DataFolder
from refiner.video.remux import reset_opened_video_source_cache
from refiner.video.transcode import VideoTranscodeConfig
from refiner.video.types import VideoSource
from refiner.video.writer import VideoStreamWriter
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.lerobot import LeRobotWriterSink
from refiner.pipeline.sinks.reducer.lerobot import LeRobotMetaReduceSink
from refiner.robotics.row import RoboticsRow, _robot_row_converter
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotStatsFile,
    LeRobotTasks,
)
from refiner.robotics.lerobot_format.metadata.stats import _estimate_sample_stride
from refiner.worker.context import set_active_run_context, worker_token_for
from refiner.worker.lifecycle import FinalizedShardWorker, RuntimeLifecycle

_ALOHA_REPO_IDS = (
    "macrodata/aloha_static_battery_ep000_004",
    "macrodata/aloha_static_battery_ep005_009",
)


class _FakeRoboticsRow(Row, RoboticsRow):
    def __init__(
        self,
        *,
        episode_id: str,
        frame_table: Tabular,
        task: str | None = None,
        fps: float | None = None,
        robot_type: str | None = None,
        videos: Mapping[str, VideoSource] | None = None,
    ) -> None:
        self._data = {"episode_id": episode_id}
        self._frame_table = frame_table
        self._task = task
        self._fps = fps
        self._robot_type = robot_type
        self._videos = dict(videos or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def episode_id(self) -> str:
        return str(self._data["episode_id"])

    @property
    def num_frames(self) -> int:
        return self._frame_table.num_rows

    @property
    def task(self) -> str | None:
        return self._task

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def robot_type(self) -> str | None:
        return self._robot_type

    @property
    def videos(self) -> Mapping[str, VideoSource]:
        return self._videos

    @property
    def stats(self) -> Mapping[str, Any]:
        return {}

    def _frame_values(self, key: str) -> Any:
        return self._frame_table.column(key)

    @property
    def timestamps(self) -> Any:
        return (
            self._frame_values("timestamp")
            if "timestamp" in self._frame_table.names
            else None
        )

    @property
    def actions(self) -> Any:
        return (
            self._frame_values("action")
            if "action" in self._frame_table.names
            else None
        )

    @property
    def states(self) -> Any:
        return (
            self._frame_values("observation.state")
            if "observation.state" in self._frame_table.names
            else None
        )

    def observations(self, name: str | None = None) -> Any:
        values: dict[str, Any] = {
            key[len("observation.") :]: self._frame_table.column(key)
            for key in self._frame_table.names
            if key.startswith("observation.")
        }
        values.update({f"videos/{key}": video for key, video in self.videos.items()})
        if name is None:
            return values
        normalized_name = (
            name[len("observation.") :] if name.startswith("observation.") else name
        )
        return values[normalized_name]

    def with_timestamps(self, values: Any) -> "_FakeRoboticsRow":
        return self._with_frame_values("timestamp", values)

    def with_actions(self, values: Any) -> "_FakeRoboticsRow":
        return self._with_frame_values("action", values)

    def with_observation(self, key: str, values: Any) -> "_FakeRoboticsRow":
        frame_key = key if key.startswith("observation.") else f"observation.{key}"
        return self._with_frame_values(frame_key, values)

    def _with_frame_values(self, key: str, values: Any) -> "_FakeRoboticsRow":
        table = self._frame_table.table
        column = (
            values
            if isinstance(values, (pa.Array, pa.ChunkedArray))
            else pa.array(values)
        )
        if key in table.column_names:
            table = table.set_column(table.column_names.index(key), key, column)
        else:
            table = table.append_column(key, column)
        return self._with_frame_table(self._frame_table.with_table(table))

    def select_frames(self, indices: Sequence[int]) -> "_FakeRoboticsRow":
        selected = self._frame_table.table.take(pa.array(indices, type=pa.int64()))
        return self._with_frame_table(self._frame_table.with_table(selected))

    def to_frame_table(self) -> Tabular:
        return self._frame_table

    def with_video(self, key: str, video: VideoSource) -> "_FakeRoboticsRow":
        videos = dict(self._videos)
        videos[key] = video
        return _FakeRoboticsRow(
            episode_id=self.episode_id,
            frame_table=self._frame_table,
            task=self.task,
            fps=self.fps,
            robot_type=self.robot_type,
            videos=videos,
        )

    def drop_stats(self, feature: str) -> "_FakeRoboticsRow":
        _ = feature
        return self

    def _with_frame_table(self, frame_table: Tabular) -> "_FakeRoboticsRow":
        return _FakeRoboticsRow(
            episode_id=self.episode_id,
            frame_table=frame_table,
            task=self.task,
            fps=self.fps,
            robot_type=self.robot_type,
            videos=self.videos,
        )

    def update(
        self,
        patch: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> "_FakeRoboticsRow":
        data = dict(self._data)
        data.update(patch or {})
        data.update(kwargs)
        return _FakeRoboticsRow(
            episode_id=str(data["episode_id"]),
            frame_table=self._frame_table,
            task=self.task,
            fps=self.fps,
            robot_type=self.robot_type,
            videos=self.videos,
        )


@pytest.fixture(autouse=True)
def _reset_opened_source_cache() -> Iterator[None]:
    reset_opened_video_source_cache()
    yield
    reset_opened_video_source_cache()


def test_lerobot_sink_defaults_encoder_options_to_none(tmp_path: Path) -> None:
    writer = LeRobotWriterSink(str(tmp_path / "out"))

    assert writer.video_transcode_config.encoder_options is None


def test_write_lerobot_defaults_gop_to_two(tmp_path: Path) -> None:
    pipeline = mdr.from_items([]).write_lerobot(str(tmp_path / "out"))
    writer = cast(LeRobotWriterSink, pipeline.sink)

    assert writer.video_transcode_config.encoder_options == {"g": "2"}


def test_write_lerobot_allows_gop_override(tmp_path: Path) -> None:
    pipeline = mdr.from_items([]).write_lerobot(
        str(tmp_path / "out"),
        encoder_options={"g": "12", "preset": "veryfast"},
    )
    writer = cast(LeRobotWriterSink, pipeline.sink)

    assert writer.video_transcode_config.encoder_options == {
        "g": "12",
        "preset": "veryfast",
    }


def test_write_lerobot_allows_encoder_options_opt_out(tmp_path: Path) -> None:
    pipeline = mdr.from_items([]).write_lerobot(
        str(tmp_path / "out"),
        encoder_options=None,
    )
    writer = cast(LeRobotWriterSink, pipeline.sink)

    assert writer.video_transcode_config.encoder_options is None


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


def _robot_row(row: Row, **kwargs: Any) -> RoboticsRow:
    return cast(RoboticsRow, _robot_row_converter(**kwargs)(row))


def _episode(
    *,
    episode_index: int,
    task: str,
    task_index: int,
    video_path: Path,
    from_ts: float,
    to_ts: float,
    values: list[float],
) -> RoboticsRow:
    return _robot_row(
        DictRow(
            {
                "episode_id": str(episode_index),
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
                "observation.images.main": mdr.video.VideoFile(
                    DataFile.resolve(str(video_path)),
                    from_timestamp_s=from_ts,
                    to_timestamp_s=to_ts,
                ),
            }
        ),
        episode_id_key="episode_id",
        task_key="task",
        fps=10,
        robot_type="mockbot",
        nested_frames_key="frames",
        video_keys=("observation.images.main",),
    )


def _robotics_episode(
    *,
    episode_index: int,
    task: str,
    task_index: int,
    values: list[float],
) -> RoboticsRow:
    return _robot_row(
        DictRow(
            {
                "episode_id": str(episode_index),
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
            }
        ),
        episode_id_key="episode_id",
        task_key="task",
        fps=10,
        robot_type="mockbot",
        nested_frames_key="frames",
    )


def _lerobot_episode(
    *,
    episode_index: int,
    task: str,
    task_index: int,
    values: list[float],
    metadata: LeRobotMetadata | None = None,
    shard_id: str | None = None,
) -> LeRobotRow:
    frames = [
        DictRow(
            {
                "frame_index": i,
                "timestamp": float(i) / 10.0,
                "task_index": task_index,
                "observation.state": [v],
            }
        )
        for i, v in enumerate(values)
    ]
    return LeRobotRow(
        DictRow(
            {
                "episode_index": episode_index,
                "episode_id": str(episode_index),
                "task": task,
                "length": len(frames),
            },
            shard_id=shard_id,
        ),
        metadata=metadata or _metadata(),
        frames=frames,
    )


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
        rundir=str(tmp_path / "workdir"),
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
            _robotics_episode(
                episode_index=0,
                task="pick",
                task_index=0,
                values=[1.0, 2.0],
            ),
            _robotics_episode(
                episode_index=1,
                task="place",
                task_index=1,
                values=[3.0, 4.0],
            ),
        ],
        items_per_shard=1,
    ).write_lerobot(str(out_root))

    stats = pipeline.launch_local(
        name="lerobot-two-stage-local",
        num_workers=2,
        rundir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0
    assert stats.claimed >= 3
    assert (out_root / "meta" / "info.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert (out_root / "meta" / "tasks.parquet").exists()


def test_write_lerobot_skips_shard_with_only_empty_frame_rows(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "empty-frame-shard"
    writer = LeRobotWriterSink(str(out_root))
    row = _FakeRoboticsRow(
        episode_id="empty",
        task="pick",
        fps=10,
        robot_type="mockbot",
        frame_table=Tabular.from_rows([]),
    )

    writer.write_shard_block("shard-empty", [row])
    writer.on_shard_complete("shard-empty")
    writer.close()

    worker_token = worker_token_for("local")
    assert not (out_root / "meta" / f"chunk-shard-empty__w{worker_token}").exists()


def test_write_lerobot_accepts_generic_robotics_rows(tmp_path: Path) -> None:
    out_root = tmp_path / "generic-robotics"
    rows = [
        _FakeRoboticsRow(
            episode_id="demo-pick",
            task="pick",
            fps=10,
            robot_type="mockbot",
            frame_table=Tabular.from_rows(
                [
                    DictRow(
                        {
                            "frame_index": 0,
                            "timestamp": 0.0,
                            "observation.state": [1.0],
                        }
                    ),
                    DictRow(
                        {
                            "frame_index": 1,
                            "timestamp": 0.1,
                            "observation.state": [2.0],
                        }
                    ),
                ]
            ),
        ),
        _FakeRoboticsRow(
            episode_id="demo-place",
            task="place",
            fps=10,
            robot_type="mockbot",
            frame_table=Tabular.from_rows(
                [
                    DictRow(
                        {
                            "frame_index": 0,
                            "timestamp": 0.0,
                            "observation.state": [3.0],
                        }
                    ),
                    DictRow(
                        {
                            "frame_index": 1,
                            "timestamp": 0.1,
                            "observation.state": [4.0],
                        }
                    ),
                ]
            ),
        ),
    ]

    finalized: list[FinalizedShardWorker] = []
    for shard_id, worker_id, row in [
        ("shard-1", "worker-1", rows[0]),
        ("shard-2", "worker-2", rows[1]),
    ]:
        writer = LeRobotWriterSink(str(out_root))
        finalized.append(FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id))
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime()),
        ):
            writer.write_shard_block(shard_id, [row])
            writer.on_shard_complete(shard_id)

    reducer = LeRobotMetaReduceSink(output=str(out_root))
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="local",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime(finalized)),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    tasks = pq.read_table(out_root / "meta" / "tasks.parquet")
    assert sorted(tasks.column("task").to_pylist()) == ["pick", "place"]
    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert sorted(task[0] for task in episodes.column("tasks").to_pylist()) == [
        "pick",
        "place",
    ]


def test_write_lerobot_generates_missing_generic_episode_ids(tmp_path: Path) -> None:
    out_root = tmp_path / "generated-episode-ids"
    writer = LeRobotWriterSink(str(out_root))

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime()),
    ):
        writer.write_shard_block(
            "shard-a",
            [
                _FakeRoboticsRow(
                    episode_id="-1",
                    fps=10,
                    robot_type="mockbot",
                    frame_table=Tabular.from_rows(
                        [
                            DictRow(
                                {
                                    "timestamp": 0.0,
                                    "observation.state": [1.0],
                                }
                            )
                        ]
                    ),
                ),
                _FakeRoboticsRow(
                    episode_id="-1",
                    fps=10,
                    robot_type="mockbot",
                    frame_table=Tabular.from_rows(
                        [
                            DictRow(
                                {
                                    "timestamp": 0.0,
                                    "observation.state": [2.0],
                                }
                            )
                        ]
                    ),
                ),
            ],
        )
        writer.on_shard_complete("shard-a")

    chunk = f"shard-a__w{worker_token_for('worker-1')}"
    episodes = pq.read_table(
        out_root / "meta" / f"chunk-{chunk}" / "episodes" / "file-000.parquet"
    )

    assert episodes.column("episode_id").to_pylist() == ["shard-a/0", "shard-a/1"]
    frames = pq.read_table(out_root / "data" / f"chunk-{chunk}" / "file-000.parquet")
    assert (
        frames.column("episode_index").to_pylist()
        == episodes.column("episode_index").to_pylist()
    )


def test_write_lerobot_accepts_to_robot_rows_after_vectorized_filter(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "generic-filtered"

    stats = (
        mdr.from_items(
            [
                {
                    "episode_id": "episode-0",
                    "task": "drop",
                    "keep": False,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [0.0],
                            "observation.state": [1.0],
                        }
                    ],
                },
                {
                    "episode_id": "episode-1",
                    "task": "keep",
                    "keep": True,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [1.0],
                            "observation.state": [2.0],
                        }
                    ],
                },
            ]
        )
        .to_robot_rows(
            episode_id_key="episode_id",
            task_key="task",
            nested_frames_key="frames",
            fps=10,
            robot_type="mockbot",
        )
        .filter(mdr.col("keep"))
        .write_lerobot(str(out_root))
        .launch_local(
            name="generic-filtered-write-lerobot",
            num_workers=1,
            rundir=str(tmp_path / "workdir-generic-filtered"),
        )
    )

    assert stats.failed == 0
    rows = mdr.read_lerobot(str(out_root)).materialize()
    assert len(rows) == 1
    assert rows[0]["tasks"] == ["keep"]
    assert cast(LeRobotRow, rows[0]).to_frame_table().column("action").to_pylist() == [
        [1.0]
    ]


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

    original_av_open = av.open
    read_open_calls = 0

    def _counting_av_open(*args, **kwargs):
        nonlocal read_open_calls
        if kwargs.get("mode") == "r":
            read_open_calls += 1
        return original_av_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", _counting_av_open)

    writer = VideoStreamWriter(
        folder=DataFolder.resolve(str(tmp_path / "out")),
        stream_key="observation.images.main",
        transcode_config=VideoTranscodeConfig(),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4",
        output_context={"chunk_index": "000"},
    )
    first_video = mdr.video.VideoFile(
        DataFile.resolve(uri),
        from_timestamp_s=0.0,
        to_timestamp_s=0.3,
    )
    asyncio.run(first_video.write_to(writer))
    second_video = mdr.video.VideoFile(
        DataFile.resolve(uri),
        from_timestamp_s=0.3,
        to_timestamp_s=0.6,
    )
    asyncio.run(second_video.write_to(writer))
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
            rundir=str(tmp_path / "workdir-force-recompute"),
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
        rundir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0

    video_dirs = list((out_root / "videos" / "observation.images.main").glob("chunk-*"))
    video_files = [
        path for directory in video_dirs for path in sorted(directory.glob("*.mp4"))
    ]
    assert any(path.name == "file-001.mp4" for path in video_files)


def test_lerobot_sink_closes_video_writers_concurrently(tmp_path: Path) -> None:
    writer = LeRobotWriterSink(str(tmp_path / "out"))
    started = 0
    started_lock = threading.Lock()
    both_started = threading.Event()
    closed: list[str] = []

    class _FakeVideoWriter:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close_async(self) -> None:
            nonlocal started
            with started_lock:
                started += 1
                if started == 2:
                    both_started.set()
            await asyncio.to_thread(both_started.wait, 0.5)
            if not both_started.is_set():
                raise AssertionError("video writer closes did not overlap")
            closed.append(self.name)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime()),
    ):
        state = writer._state_for_shard("shard-1")
        state.metadata = _metadata()
        state.video_writers = cast(
            dict[str, VideoStreamWriter],
            {
                "left": _FakeVideoWriter("left"),
                "right": _FakeVideoWriter("right"),
            },
        )
        writer._commit_shard(state)

    assert started == 2
    assert sorted(closed) == ["left", "right"]


def test_write_lerobot_preserves_stable_task_index_mapping(tmp_path: Path) -> None:
    out_root = tmp_path / "task-index"
    pipeline = mdr.from_items(
        [
            _lerobot_episode(
                episode_index=0,
                task="pick",
                task_index=5,
                values=[1.0, 2.0],
                metadata=_metadata({1: "place", 5: "pick"}),
            ),
            _lerobot_episode(
                episode_index=1,
                task="place",
                task_index=1,
                values=[3.0, 4.0],
                metadata=_metadata({1: "place", 5: "pick"}),
            ),
        ],
        items_per_shard=10,
    ).write_lerobot(str(out_root))

    stats = pipeline.launch_local(
        name="lerobot-task-index-stable",
        num_workers=1,
        rundir=str(tmp_path / "workdir"),
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
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime()),
    ):
        with pytest.raises(
            KeyError,
            match="7",
        ):
            writer.write_block(
                [
                    _lerobot_episode(
                        episode_index=0,
                        task="pick",
                        task_index=7,
                        values=[1.0],
                        metadata=_metadata(),
                        shard_id="shard-1",
                    )
                ]
            )
            writer.close()


def test_write_lerobot_rejects_fractional_fps(tmp_path: Path) -> None:
    writer = LeRobotWriterSink(str(tmp_path / "out"))
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime()),
    ):
        with pytest.raises(ValueError, match="LeRobot output requires integer fps"):
            writer.write_shard_block(
                "shard-1",
                [
                    _FakeRoboticsRow(
                        episode_id="fractional",
                        fps=29.97,
                        robot_type="mockbot",
                        frame_table=Tabular.from_rows(
                            [
                                DictRow(
                                    {
                                        "frame_index": 0,
                                        "timestamp": 0.0,
                                        "observation.state": [1.0],
                                    }
                                )
                            ]
                        ),
                    )
                ],
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
    for worker_id, values in [("1", [1.0, 2.0]), ("2", [9.0, 10.0])]:
        writer = LeRobotWriterSink(str(out_root))
        runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime())
        worker_row = _lerobot_episode(
            episode_index=0,
            task="pick",
            task_index=0,
            values=values,
            shard_id="shard-1",
        )
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=runtime,
        ):
            writer.write_block([worker_row])
            writer.on_shard_complete("shard-1")

    reducer = LeRobotMetaReduceSink(output=str(out_root))
    runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime())
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="local",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    worker_1 = worker_token_for("1")
    worker_2 = worker_token_for("2")
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
            rundir=str(tmp_path / "workdir-hub-aloha-merge"),
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
