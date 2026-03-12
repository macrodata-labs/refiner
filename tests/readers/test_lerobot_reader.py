from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.sources.row import Row
from refiner.sources.readers.lerobot import (
    LEROBOT_INFO,
    LEROBOT_STATS,
    LeRobotEpisodeReader,
)
from refiner.media import Video


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_info(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 30,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.main": {"dtype": "video"},
            "episode_index": {"dtype": "int64"},
        },
    }
    path.write_text(json.dumps(info), encoding="utf-8")


def _write_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "observation.state": {
            "min": [-1.0],
            "max": [1.0],
            "mean": [0.0],
            "std": [0.5],
            "count": 2,
        }
    }
    path.write_text(json.dumps(stats), encoding="utf-8")


def _build_sample_dataset(root: Path) -> None:
    _write_info(root / "meta" / "info.json")
    _write_stats(root / "meta" / "stats.json")
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [
            {"task_index": 0, "task": "pick"},
            {"task_index": 1, "task": "place"},
        ],
    )
    _write_parquet(
        root / "meta" / "episodes" / "part-000.parquet",
        [
            {
                "episode_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": ["pick"],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            },
            {
                "episode_index": 1,
                "dataset_from_index": 2,
                "dataset_to_index": 4,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": ["place"],
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
                "task_index": 0,
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
            },
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": 1,
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 1.1,
                "task_index": 1,
            },
        ],
    )


def test_lerobot_reader_emits_episode_rows(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root), decode=False)
    shards = reader.list_shards()
    assert len(shards) == 1
    assert shards[0].start == 0
    assert shards[0].end == -1
    assert shards[0].path.endswith("meta/episodes/part-000.parquet")

    rows = list(reader.read_shard(shards[0]))
    assert len(rows) == 2
    first = rows[0]
    second = rows[1]

    assert int(first["episode_index"]) == 0
    assert int(second["episode_index"]) == 1
    assert first["task"] == "pick"
    assert second["task"] == "place"
    assert isinstance(first["metadata"], dict)
    assert first["metadata"][LEROBOT_STATS]["observation.state"]["min"] == [-1.0]
    assert first["metadata"][LEROBOT_STATS]["observation.state"]["max"] == [1.0]
    assert first["metadata"][LEROBOT_STATS]["observation.state"]["mean"] == [0.0]
    assert first["metadata"][LEROBOT_STATS]["observation.state"]["std"] == [0.5]
    assert first["metadata"][LEROBOT_STATS]["observation.state"]["count"] == 2
    assert first["metadata"][LEROBOT_INFO]["fps"] == 30
    assert "stats/observation.state/min" not in first
    assert "videos/observation.images.main/chunk_index" not in first
    assert "meta/episodes/chunk_index" not in first
    assert "data/chunk_index" not in first
    assert isinstance(first["metadata"][LEROBOT_STATS], Mapping)
    assert isinstance(first["metadata"][LEROBOT_INFO], dict)
    assert first["metadata"][LEROBOT_STATS] == second["metadata"][LEROBOT_STATS]

    first_frames = first["frames"]
    assert isinstance(first_frames, list)
    assert len(first_frames) == 2
    assert isinstance(first_frames[0], Row)
    assert int(first_frames[0]["frame_index"]) == 0
    assert int(first_frames[1]["frame_index"]) == 1

    video = first["observation.images.main"]
    assert isinstance(video, Video)
    assert video.uri.endswith("/videos/observation.images.main/chunk-000/file-000.mp4")
    assert video.episode_index == 0
    assert video.file_index == 0
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_supports_legacy_chunk_key_video_path_template(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_key}/file-{file_index:03d}.mp4"
    )
    info_path.write_text(json.dumps(info), encoding="utf-8")

    reader = LeRobotEpisodeReader(str(root), decode=False)
    rows = list(reader.read_shard(reader.list_shards()[0]))

    assert len(rows) == 2
    video = rows[0]["observation.images.main"]
    assert video.uri.endswith("/videos/observation.images.main/chunk-0/file-000.mp4")
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_does_not_eagerly_load_parquet_rows_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root), decode=False)

    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("unexpected eager parquet table read")

    monkeypatch.setattr(pq, "read_table", _fail)
    shards = reader.list_shards()

    assert len(shards) == 1


def test_lerobot_reader_requires_episode_parquet_metadata(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _write_info(root / "meta" / "info.json")
    with pytest.raises(FileNotFoundError):
        mdr.read_lerobot(str(root)).materialize()


def test_lerobot_decode_none_raises_for_timestamped_videos(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    pipeline = mdr.read_lerobot(str(root))
    with pytest.raises(ValueError, match="decode is None"):
        pipeline.materialize()


def test_lerobot_reader_requires_dataset_bounds(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _write_info(root / "meta" / "info.json")
    _write_stats(root / "meta" / "stats.json")
    _write_parquet(
        root / "meta" / "episodes" / "part-000.parquet",
        [
            {
                "episode_index": 0,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "tasks": ["pick"],
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [{"index": 0, "episode_index": 0, "frame_index": 0, "timestamp": 0.0}],
    )

    reader = LeRobotEpisodeReader(str(root), decode=False)
    with pytest.raises(
        ValueError,
        match="missing required dataset_from_index/dataset_to_index",
    ):
        list(reader.read_shard(reader.list_shards()[0]))


def test_lerobot_reader_raises_when_bounds_outside_data_file(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _write_info(root / "meta" / "info.json")
    _write_stats(root / "meta" / "stats.json")
    _write_parquet(
        root / "meta" / "episodes" / "part-000.parquet",
        [
            {
                "episode_index": 0,
                "dataset_from_index": 10,
                "dataset_to_index": 12,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "tasks": ["pick"],
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [{"index": 0, "episode_index": 0, "frame_index": 0, "timestamp": 0.0}],
    )

    reader = LeRobotEpisodeReader(str(root), decode=False)
    with pytest.raises(
        ValueError,
        match="dataset bounds are out of range for data parquet file",
    ):
        list(reader.read_shard(reader.list_shards()[0]))
