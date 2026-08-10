from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typing import cast

import refiner as mdr
from refiner.io import DataFolder
from refiner.pipeline.data.row import DictRow
from refiner.video import VideoFile
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotFeatureInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotStatsFile,
    LeRobotTasks,
    LeRobotVideoInfo,
    remap_task_index_table,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _episode_rows(reader: LeRobotEpisodeReader) -> list[LeRobotRow]:
    blocks = [
        block
        for shard in reader.list_shards()
        for block in reader.read_shard(shard)
        if isinstance(block, Tabular)
    ]
    assert all(isinstance(block, Tabular) for block in blocks)
    rows = [cast(LeRobotRow, row) for block in blocks for row in block.to_rows()]
    assert all(isinstance(row, LeRobotRow) for row in rows)
    return rows


def _build_sample_dataset(
    root: Path,
    *,
    index_to_task: dict[int, str] | None = None,
    episode_tasks: tuple[str, str] = ("pick", "place"),
    frame_task_indices: tuple[int, int, int, int] = (0, 0, 1, 1),
) -> None:
    index_to_task = index_to_task or {0: "pick", 1: "place"}
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {"observation.images.main": {"dtype": "video"}},
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [-1.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.5],
                    "count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [
            {"task": task, "task_index": task_index}
            for task_index, task in index_to_task.items()
        ],
    )
    _write_parquet(
        root / "meta" / "episodes" / "part-000.parquet",
        [
            {
                "episode_index": 0,
                "length": 2,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": [episode_tasks[0]],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            },
            {
                "episode_index": 1,
                "length": 2,
                "dataset_from_index": 2,
                "dataset_to_index": 4,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": [episode_tasks[1]],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 1,
                "videos/observation.images.main/from_timestamp": 1.0,
                "videos/observation.images.main/to_timestamp": 2.0,
            },
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": frame_task_indices[0],
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": frame_task_indices[1],
            },
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": frame_task_indices[2],
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 1.1,
                "task_index": frame_task_indices[3],
            },
        ],
    )


def test_lerobot_reader_emits_episode_rows(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root))
    rows = _episode_rows(reader)
    assert len(rows) == 2
    first, second = rows

    assert int(first["episode_index"]) == 0
    assert int(second["episode_index"]) == 1
    assert isinstance(first["metadata"], LeRobotMetadata)
    assert first["metadata"].info is second["metadata"].info
    assert first["metadata"].tasks.index_to_task == {0: "pick", 1: "place"}
    assert first["metadata"].tasks is second["metadata"].tasks
    assert first["metadata"].stats["observation.state"].count == 2
    with pytest.raises(FrozenInstanceError):
        first["metadata"].info = first["metadata"].info
    assert first["tasks"] == ["pick"]
    assert first["stats/observation.state/min"] == [-999.0]
    first_frames = first["frames"]
    assert isinstance(first_frames, Tabular)
    frame_rows = first_frames.to_rows()
    assert len(frame_rows) == 2
    assert int(frame_rows[0]["frame_index"]) == 0
    assert int(frame_rows[1]["frame_index"]) == 1

    video = first.videos["observation.images.main"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/videos/observation.images.main/chunk-000/file-000.mp4")
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_raises_on_malformed_frame_count(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": 0,
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
            },
        ],
    )

    reader = LeRobotEpisodeReader(str(root))

    with pytest.raises(ValueError, match="episode 1 expected 2 frames"):
        _episode_rows(reader)


def test_lerobot_reader_can_skip_malformed_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": 0,
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
            },
        ],
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.lerobot.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    metrics: list[tuple[str, int | float, str, str | None]] = []
    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.lerobot.log_throughput",
        lambda label, value, shard_id, *, unit=None, step_index=None: metrics.append(
            (label, value, shard_id, unit)
        ),
    )

    reader = LeRobotEpisodeReader(str(root), skip_malformed_rows=True)
    rows = _episode_rows(reader)

    assert [int(row["episode_index"]) for row in rows] == [0]
    assert len(warnings) == 1
    assert warnings[0] == "Skipping malformed LeRobot episodes"
    assert [(label, value, unit) for label, value, _, unit in metrics] == [
        ("malformed_lerobot_episodes_skipped", 1, "episodes")
    ]


def test_lerobot_row_uses_absolute_video_uri_directly() -> None:
    uri = "hf://datasets/org/repo/videos/chunk-000/file-000.mp4"
    row = LeRobotRow(
        DictRow(
            {
                "episode_index": 0,
                "length": 1,
                "videos/observation.images.main/uri": uri,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            }
        ),
        metadata=LeRobotMetadata(
            info=LeRobotInfo(fps=10),
            stats=LeRobotStatsFile({}),
            tasks=LeRobotTasks({0: "pick"}),
        ),
        frames=[],
        root=DataFolder.resolve("hf://datasets/org/repo"),
    )

    video = row.videos["observation.images.main"]

    assert isinstance(video, VideoFile)
    assert video.uri == uri

    assert row.update(root=None).videos["observation.images.main"].uri == uri


def test_lerobot_row_repr_summarizes_episode() -> None:
    row = LeRobotRow(
        DictRow(
            {
                "episode_index": 7,
                "episode_id": "battery-7",
                "length": 2,
                "task": "battery insertion",
                "videos/observation.images.main/uri": "hf://datasets/org/repo/video.mp4",
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            }
        ),
        metadata=LeRobotMetadata(
            info=LeRobotInfo(
                fps=30,
                robot_type="aloha",
                features={
                    "observation.images.main": LeRobotFeatureInfo(
                        dtype="video",
                        shape=(480, 640, 3),
                        video_info=LeRobotVideoInfo(fps=30),
                    )
                },
            ),
            stats=LeRobotStatsFile({}),
            tasks=LeRobotTasks({0: "pick"}),
        ),
        frames=[
            DictRow({"frame_index": 0, "action": [0.0]}),
            DictRow({"frame_index": 1, "action": [1.0]}),
        ],
        root=None,
    )

    text = repr(row)

    assert text.startswith("LeRobotRow(")
    assert "episode_id='battery-7'" in text
    assert "num_frames=2" in text
    assert "task='battery insertion'" in text
    assert "fps=30" in text
    assert "robot_type='aloha'" in text
    assert "actions (row.actions): double[2, 1]" in text
    assert "frame_index" not in text
    assert "observation.images.main: video[480, 640, 3]@30fps" in text
    assert "metadata=" not in text
    assert "frames=[" not in text
    assert "_row=" not in text
    assert "fields=" not in text


def test_lerobot_reader_exposes_episode_shard_planning_knobs(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root), num_shards=2)
    shards = reader.list_shards()

    assert len(shards) == 2


def test_lerobot_vectorized_filter_realigns_side_data(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = (
        mdr.read_lerobot(str(root)).filter(mdr.col("episode_index") == 1).materialize()
    )

    assert len(rows) == 1
    row = cast(LeRobotRow, rows[0])
    assert int(row["episode_index"]) == 1
    frame_rows = row["frames"].to_rows()
    assert [int(frame["episode_index"]) for frame in frame_rows] == [1, 1]
    assert [int(frame["index"]) for frame in frame_rows] == [2, 3]


def test_lerobot_row_preserving_vectorized_ops_keep_side_data(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = mdr.read_lerobot(str(root)).select("episode_index").materialize()

    assert [int(row["episode_index"]) for row in rows] == [0, 1]
    frame_rows = [cast(LeRobotRow, row)["frames"].to_rows() for row in rows]
    assert int(frame_rows[0][0]["episode_index"]) == 0
    assert int(frame_rows[1][0]["episode_index"]) == 1


def test_lerobot_map_table_can_reorder_rows_with_lineage(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = (
        mdr.read_lerobot(str(root))
        .map_table(lambda table: table.sort_by([("episode_index", "descending")]))
        .materialize()
    )

    assert [int(row["episode_index"]) for row in rows] == [1, 0]
    frame_rows = [cast(LeRobotRow, row)["frames"].to_rows() for row in rows]
    assert int(frame_rows[0][0]["episode_index"]) == 1
    assert int(frame_rows[1][0]["episode_index"]) == 0


def test_lerobot_reader_describe_uses_dataset_roots(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root))

    assert reader.describe() == {
        "path": str(root),
        "inputs": [str(root)],
    }


def test_lerobot_reader_accepts_hf_dataset_url() -> None:
    reader = LeRobotEpisodeReader("hf://datasets/lerobot/aloha_mobile_cabinet")

    assert reader.describe() == {
        "path": "hf://datasets/lerobot/aloha_mobile_cabinet",
        "inputs": ["hf://datasets/lerobot/aloha_mobile_cabinet"],
    }
    assert reader.required_refiner_extras() == ("hf",)


def test_remap_task_index_table_raises_when_index_cannot_be_remapped() -> None:
    table = pa.Table.from_pydict(
        {
            "task_index": [0, 7, 1],
            "frame_index": [0, 1, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"missing from source task metadata: \[7\]",
    ):
        remap_task_index_table(table, {0: 3, 1: 4})


def test_lerobot_reader_offsets_episode_indices_across_multiple_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "lerobot-a"
    second_root = tmp_path / "lerobot-b"
    _build_sample_dataset(first_root)
    _build_sample_dataset(
        second_root,
        index_to_task={0: "place", 1: "stack"},
        episode_tasks=("place", "stack"),
        frame_task_indices=(0, 0, 1, 1),
    )

    reader = LeRobotEpisodeReader([str(first_root), str(second_root)])
    rows = _episode_rows(reader)

    assert len(rows) == 4
    assert [int(row["episode_index"]) for row in rows] == [0, 1, 0, 1]
    expected_tasks = {0: "pick", 1: "place", 2: "stack"}
    assert rows[0]["metadata"].tasks.index_to_task == expected_tasks
    assert rows[2]["metadata"].tasks.index_to_task == expected_tasks
    assert [int(frame["task_index"]) for frame in rows[2]["frames"].to_rows()] == [1, 1]
    assert [int(frame["task_index"]) for frame in rows[3]["frames"].to_rows()] == [2, 2]
