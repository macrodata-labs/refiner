from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.media import Video


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_info(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 30,
        "features": {
            "observation.images.main": {"dtype": "video"},
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

    reader = LeRobotEpisodeReader(str(root))
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
    assert first["metadata"]["lerobot_stats"]["observation.state"]["min"] == [-1.0]
    assert first["metadata"]["lerobot_stats"]["observation.state"]["max"] == [1.0]
    assert first["metadata"]["lerobot_stats"]["observation.state"]["mean"] == [0.0]
    assert first["metadata"]["lerobot_stats"]["observation.state"]["std"] == [0.5]
    assert first["metadata"]["lerobot_stats"]["observation.state"]["count"] == 2
    assert "stats/observation.state/min" not in first
    assert "videos/observation.images.main/chunk_index" not in first
    assert "meta/episodes/chunk_index" not in first
    assert (
        first["metadata"]["lerobot_stats"]["observation.state"]
        == second["metadata"]["lerobot_stats"]["observation.state"]
    )

    first_frames = first["frames"]
    assert isinstance(first_frames, list)
    assert len(first_frames) == 2
    assert int(first_frames[0]["frame_index"]) == 0
    assert int(first_frames[1]["frame_index"]) == 1

    video = first["observation.images.main"]
    assert isinstance(video, Video)
    assert video.uri.endswith("/videos/observation.images.main/chunk-000/file-000.mp4")
    assert video.episode_index == 0
    assert video.file_index == 0
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_does_not_eagerly_load_parquet_rows_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root))

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


def test_lerobot_reader_keeps_timestamped_video_references(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = mdr.read_lerobot(str(root)).materialize()
    assert len(rows) == 2
    assert isinstance(rows[0]["observation.images.main"], Video)


def test_lerobot_reader_exposes_media_async_window_settings(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(
        str(root),
        media_max_in_flight=3,
        media_preserve_order=False,
    )

    assert reader.describe()["media_max_in_flight"] == 3
    assert reader.describe()["media_preserve_order"] is False


def test_lerobot_reader_rejects_non_positive_media_max_in_flight(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    with pytest.raises(ValueError, match="media_max_in_flight must be > 0"):
        LeRobotEpisodeReader(str(root), media_max_in_flight=0)
