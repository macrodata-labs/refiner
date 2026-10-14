from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.lerobot_v2_1 import LeRobotV2_1EpisodeReader
from refiner.robotics.lerobot_format import LeRobotMetadata, LeRobotRow
from refiner.video import VideoFile


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_lerobot_v2_1_rejects_explicit_shard_count_above_limit(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    reader = LeRobotV2_1EpisodeReader(root, num_shards=10_001)

    with pytest.raises(ValueError, match=r"num_shards must be <= 10,000"):
        reader.list_shards()


def _build_sample_dataset(
    root: Path,
    *,
    index_to_task: dict[int, str] | None = None,
    episode_tasks: tuple[str, str] = ("pick", "place"),
    frame_task_indices: tuple[int, int, int, int] = (0, 0, 1, 1),
) -> None:
    index_to_task = index_to_task or {0: "pick", 1: "place"}
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "chunks_size": 1000,
                "features": {"observation.images.main": {"dtype": "video"}},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": idx, "task": task} for idx, task in index_to_task.items()],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "length": 2, "tasks": [episode_tasks[0]]},
            {"episode_index": 1, "length": 2, "tasks": [episode_tasks[1]]},
        ],
    )
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "observation.state": {
                        "min": [-1.0],
                        "max": [1.0],
                        "mean": [0.0],
                        "std": [0.5],
                        "count": [2],
                    }
                },
            },
            {
                "episode_index": 1,
                "stats": {
                    "observation.state": {
                        "min": [-999.0],
                        "max": [1.0],
                        "mean": [0.0],
                        "std": [0.5],
                        "count": [2],
                    }
                },
            },
        ],
    )

    _write_parquet(
        root / "data" / "chunk-000" / "episode_000000.parquet",
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
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "episode_000001.parquet",
        [
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


def _episode_rows(reader: LeRobotV2_1EpisodeReader) -> list[LeRobotRow]:
    blocks = [
        block
        for shard in reader.list_shards()
        for block in reader.read_shard(shard)
        if isinstance(block, Tabular)
    ]
    rows = [cast(LeRobotRow, row) for block in blocks for row in block.to_rows()]
    assert all(isinstance(row, LeRobotRow) for row in rows)
    return rows


def test_lerobot_v2_1_reader_emits_episode_rows(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotV2_1EpisodeReader(str(root))
    rows = _episode_rows(reader)
    assert len(rows) == 2
    first, second = rows

    assert int(first["episode_index"]) == 0
    assert int(second["episode_index"]) == 1
    assert isinstance(first["metadata"], LeRobotMetadata)
    assert first["metadata"].info is second["metadata"].info
    assert first["metadata"].tasks.index_to_task == {0: "pick", 1: "place"}
    assert first["metadata"].tasks is second["metadata"].tasks
    assert first["metadata"].stats["observation.state"].count == [4.0]
    with pytest.raises(FrozenInstanceError):
        first["metadata"].info = first["metadata"].info
    assert first["tasks"] == ["pick"]
    assert first["stats/observation.state/min"] == [-1.0]
    first_frames = first["frames"]
    assert isinstance(first_frames, Tabular)
    frame_rows = first_frames.to_rows()
    assert len(frame_rows) == 2
    assert int(frame_rows[0]["frame_index"]) == 0
    assert int(frame_rows[1]["frame_index"]) == 1

    video = first.videos["observation.images.main"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith(
        "/videos/chunk-000/observation.images.main/episode_000000.mp4"
    )
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == pytest.approx(2 / 30)

    second_video = second.videos["observation.images.main"]
    assert second_video.uri.endswith(
        "/videos/chunk-000/observation.images.main/episode_000001.mp4"
    )
    assert second_video.from_timestamp_s == 0.0
    assert second_video.to_timestamp_s == pytest.approx(2 / 30)


def test_lerobot_v2_1_reader_raises_on_malformed_frame_count(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    _write_parquet(
        root / "data" / "chunk-000" / "episode_000001.parquet",
        [
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": 1,
            },
        ],
    )

    reader = LeRobotV2_1EpisodeReader(str(root))

    with pytest.raises(ValueError, match="episode 1 expected 2 frames"):
        _episode_rows(reader)


def test_lerobot_v2_1_reader_can_skip_malformed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    _write_parquet(
        root / "data" / "chunk-000" / "episode_000001.parquet",
        [
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": 1,
            },
        ],
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.lerobot_v2_1.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    metrics: list[tuple[str, int | float, str, str | None]] = []
    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.lerobot_v2_1.log_throughput",
        lambda label, value, shard_id, *, unit=None, step_index=None: metrics.append(
            (label, value, shard_id, unit)
        ),
    )

    reader = LeRobotV2_1EpisodeReader(str(root), skip_malformed_rows=True)
    rows = _episode_rows(reader)

    assert [int(row["episode_index"]) for row in rows] == [0]
    assert len(warnings) == 1
    assert warnings[0] == "Skipping malformed LeRobot v2.1 episodes"
    assert [(label, value, unit) for label, value, _, unit in metrics] == [
        ("malformed_lerobot_episodes_skipped", 1, "episodes")
    ]


def test_lerobot_v2_1_reader_exposes_episode_shard_planning_knobs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotV2_1EpisodeReader(str(root), num_shards=2)
    shards = reader.list_shards()

    assert len(shards) == 2


def test_lerobot_v2_1_vectorized_filter_realigns_side_data(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = (
        mdr.read_lerobot_v2_1(str(root))
        .filter(mdr.col("episode_index") == 1)
        .materialize()
    )

    assert len(rows) == 1
    row = cast(LeRobotRow, rows[0])
    assert int(row["episode_index"]) == 1
    frame_rows = row["frames"].to_rows()
    assert [int(frame["episode_index"]) for frame in frame_rows] == [1, 1]
    assert [int(frame["index"]) for frame in frame_rows] == [2, 3]


def test_lerobot_v2_1_row_preserving_vectorized_ops_keep_side_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = mdr.read_lerobot_v2_1(str(root)).select("episode_index").materialize()

    assert [int(row["episode_index"]) for row in rows] == [0, 1]
    frame_rows = [cast(LeRobotRow, row)["frames"].to_rows() for row in rows]
    assert int(frame_rows[0][0]["episode_index"]) == 0
    assert int(frame_rows[1][0]["episode_index"]) == 1


def test_lerobot_v2_1_map_table_can_reorder_rows_with_lineage(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    rows = (
        mdr.read_lerobot_v2_1(str(root))
        .map_table(lambda table: table.sort_by([("episode_index", "descending")]))
        .materialize()
    )

    assert [int(row["episode_index"]) for row in rows] == [1, 0]
    frame_rows = [cast(LeRobotRow, row)["frames"].to_rows() for row in rows]
    assert int(frame_rows[0][0]["episode_index"]) == 1
    assert int(frame_rows[1][0]["episode_index"]) == 0


def test_lerobot_v2_1_reader_describe_uses_dataset_roots(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotV2_1EpisodeReader(str(root))

    assert reader.describe() == {
        "path": str(root),
        "inputs": [str(root)],
    }


def test_lerobot_v2_1_reader_accepts_hf_dataset_url() -> None:
    reader = LeRobotV2_1EpisodeReader("hf://datasets/lerobot/aloha_mobile_cabinet")

    assert reader.describe() == {
        "path": "hf://datasets/lerobot/aloha_mobile_cabinet",
        "inputs": ["hf://datasets/lerobot/aloha_mobile_cabinet"],
    }
    assert reader.required_refiner_extras() == ("hf",)


def test_lerobot_v2_1_reader_remaps_task_indices_across_multiple_roots(
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

    reader = LeRobotV2_1EpisodeReader([str(first_root), str(second_root)])
    rows = _episode_rows(reader)

    assert len(rows) == 4
    assert [int(row["episode_index"]) for row in rows] == [0, 1, 0, 1]
    expected_tasks = {0: "pick", 1: "place", 2: "stack"}
    assert rows[0]["metadata"].tasks.index_to_task == expected_tasks
    assert rows[2]["metadata"].tasks.index_to_task == expected_tasks
    assert [int(frame["task_index"]) for frame in rows[2]["frames"].to_rows()] == [1, 1]
    assert [int(frame["task_index"]) for frame in rows[3]["frames"].to_rows()] == [2, 2]
