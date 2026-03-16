from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.io import DataFolder
from refiner.media import hydrate_video
from refiner.media.video.types import DecodedVideo
from refiner.pipeline import (
    LeRobotStatsConfig as PipelineLeRobotStatsConfig,
    LeRobotVideoConfig as PipelineLeRobotVideoConfig,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sinks.lerobot import (
    LeRobotMetaReduceSink,
    LeRobotWriterConfig,
    LeRobotWriterSink,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    _PendingVideoRun,
    _PendingVideoSegment,
    _VIDEO_PROBE_CACHE_MAX_ENTRIES,
    _VideoProbeCache,
    run_can_extend,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_writer import LeRobotVideoWriter
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.context import RunHandle, set_active_run_context
from refiner.worker.lifecycle import RuntimeLifecycle


_HUB_LEROBOT_MERGE_REPO_IDS = (
    "macrodata/aloha_static_battery_ep000_004",
    "macrodata/aloha_static_battery_ep005_009",
)
_RUN_HUB_LEROBOT_MERGE_TEST_ENV = "REFINER_RUN_HUB_LEROBOT_MERGE_TEST"


def _write_video(
    path: Path,
    *,
    fps: int = 10,
    frames: int = 6,
    codec: str = "mpeg4",
    encoder_options: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = cast(
            Any, container.add_stream(codec, rate=fps, options=encoder_options)
        )
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
        stream = cast(Any, container.add_stream("mpeg4", rate=fps))
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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _build_segmented_lerobot_source(root: Path) -> Path:
    src_video = (
        root / "videos" / "observation.images.main" / "chunk-000" / "file-000.mp4"
    )
    _write_video(src_video, fps=10, frames=6)

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "robot_type": "mockbot",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": None,
                    },
                    "observation.images.main": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [0.0],
                    "max": [5.0],
                    "mean": [2.5],
                    "std": [1.707825127659933],
                    "count": 6,
                },
                "observation.images.main": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[1.0]], [[1.0]], [[1.0]]],
                    "mean": [[[0.5]], [[0.5]], [[0.5]]],
                    "std": [[[0.25]], [[0.25]], [[0.25]]],
                    "count": [6],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [
            {"task_index": 0, "task": "pick"},
            {"task_index": 1, "task": "place"},
        ],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "tasks": ["pick"],
                "task": "pick",
                "length": 3,
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 0.3,
                "stats/observation.images.main/min": [[[0.1]], [[0.2]], [[0.3]]],
                "stats/observation.images.main/max": [[[0.4]], [[0.5]], [[0.6]]],
                "stats/observation.images.main/mean": [[[0.2]], [[0.3]], [[0.4]]],
                "stats/observation.images.main/std": [[[0.01]], [[0.01]], [[0.01]]],
                "stats/observation.images.main/count": [3],
            },
            {
                "episode_index": 1,
                "tasks": ["place"],
                "task": "place",
                "length": 3,
                "dataset_from_index": 3,
                "dataset_to_index": 6,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.3,
                "videos/observation.images.main/to_timestamp": 0.6,
                "stats/observation.images.main/min": [[[0.6]], [[0.5]], [[0.4]]],
                "stats/observation.images.main/max": [[[0.9]], [[0.8]], [[0.7]]],
                "stats/observation.images.main/mean": [[[0.7]], [[0.6]], [[0.5]]],
                "stats/observation.images.main/std": [[[0.02]], [[0.02]], [[0.02]]],
                "stats/observation.images.main/count": [3],
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
                "observation.state": [0.0],
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
                "observation.state": [1.0],
            },
            {
                "index": 2,
                "episode_index": 0,
                "frame_index": 2,
                "timestamp": 0.2,
                "task_index": 0,
                "observation.state": [2.0],
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 0.3,
                "task_index": 1,
                "observation.state": [3.0],
            },
            {
                "index": 4,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 0.4,
                "task_index": 1,
                "observation.state": [4.0],
            },
            {
                "index": 5,
                "episode_index": 1,
                "frame_index": 2,
                "timestamp": 0.5,
                "task_index": 1,
                "observation.state": [5.0],
            },
        ],
    )
    return src_video


def _build_multi_file_lerobot_source(root: Path) -> tuple[Path, Path]:
    src_video_a = (
        root / "videos" / "observation.images.main" / "chunk-000" / "file-000.mp4"
    )
    src_video_b = (
        root / "videos" / "observation.images.main" / "chunk-000" / "file-001.mp4"
    )
    _write_video(src_video_a, fps=10, frames=3)
    _write_video(src_video_b, fps=10, frames=3)

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "robot_type": "mockbot",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": None,
                    },
                    "observation.images.main": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [0.0],
                    "max": [5.0],
                    "mean": [2.5],
                    "std": [1.707825127659933],
                    "count": 6,
                },
                "observation.images.main": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[1.0]], [[1.0]], [[1.0]]],
                    "mean": [[[0.5]], [[0.5]], [[0.5]]],
                    "std": [[[0.25]], [[0.25]], [[0.25]]],
                    "count": [6],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [
            {"task_index": 0, "task": "pick"},
            {"task_index": 1, "task": "place"},
        ],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "tasks": ["pick"],
                "task": "pick",
                "length": 3,
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 0.3,
                "stats/observation.images.main/min": [[[0.1]], [[0.2]], [[0.3]]],
                "stats/observation.images.main/max": [[[0.4]], [[0.5]], [[0.6]]],
                "stats/observation.images.main/mean": [[[0.2]], [[0.3]], [[0.4]]],
                "stats/observation.images.main/std": [[[0.01]], [[0.01]], [[0.01]]],
                "stats/observation.images.main/count": [3],
            },
            {
                "episode_index": 1,
                "tasks": ["place"],
                "task": "place",
                "length": 3,
                "dataset_from_index": 3,
                "dataset_to_index": 6,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 1,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 0.3,
                "stats/observation.images.main/min": [[[0.6]], [[0.5]], [[0.4]]],
                "stats/observation.images.main/max": [[[0.9]], [[0.8]], [[0.7]]],
                "stats/observation.images.main/mean": [[[0.7]], [[0.6]], [[0.5]]],
                "stats/observation.images.main/std": [[[0.02]], [[0.02]], [[0.02]]],
                "stats/observation.images.main/count": [3],
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
                "observation.state": [0.0],
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
                "observation.state": [1.0],
            },
            {
                "index": 2,
                "episode_index": 0,
                "frame_index": 2,
                "timestamp": 0.2,
                "task_index": 0,
                "observation.state": [2.0],
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": 1,
                "observation.state": [3.0],
            },
            {
                "index": 4,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 1,
                "observation.state": [4.0],
            },
            {
                "index": 5,
                "episode_index": 1,
                "frame_index": 2,
                "timestamp": 0.2,
                "task_index": 1,
                "observation.state": [5.0],
            },
        ],
    )
    return src_video_a, src_video_b


def _build_aligned_subsegment_lerobot_source(root: Path) -> Path:
    src_video = (
        root / "videos" / "observation.images.main" / "chunk-000" / "file-000.mp4"
    )
    try:
        _write_video(
            src_video,
            fps=10,
            frames=6,
            codec="h264",
            encoder_options={"g": "1"},
        )
    except av.FFmpegError as exc:
        pytest.skip(f"h264 encoder unavailable for aligned remux test: {exc}")

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "robot_type": "mockbot",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": None,
                    },
                    "observation.images.main": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [1.0],
                    "max": [2.0],
                    "mean": [1.5],
                    "std": [0.5],
                    "count": 2,
                },
                "observation.images.main": {
                    "min": [[[0.1]], [[0.2]], [[0.3]]],
                    "max": [[[0.4]], [[0.5]], [[0.6]]],
                    "mean": [[[0.2]], [[0.3]], [[0.4]]],
                    "std": [[[0.01]], [[0.01]], [[0.01]]],
                    "count": [2],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task_index": 0, "task": "pick"}],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "tasks": ["pick"],
                "task": "pick",
                "length": 2,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.1,
                "videos/observation.images.main/to_timestamp": 0.3,
                "stats/observation.images.main/min": [[[0.1]], [[0.2]], [[0.3]]],
                "stats/observation.images.main/max": [[[0.4]], [[0.5]], [[0.6]]],
                "stats/observation.images.main/mean": [[[0.2]], [[0.3]], [[0.4]]],
                "stats/observation.images.main/std": [[[0.01]], [[0.01]], [[0.01]]],
                "stats/observation.images.main/count": [2],
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.1,
                "task_index": 0,
                "observation.state": [1.0],
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.2,
                "task_index": 0,
                "observation.state": [2.0],
            },
        ],
    )
    return src_video


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


def _format_chunked_path(
    template: str,
    *,
    chunk: str | int,
    file_index: int,
    video_key: str | None = None,
) -> Path:
    return Path(
        template.format(
            video_key="" if video_key is None else video_key,
            chunk=chunk,
            chunk_key=chunk,
            chunk_index=chunk,
            file=file_index,
            file_idx=file_index,
            file_index=file_index,
        )
    )


def _load_lerobot_info(root: Path) -> dict[str, Any]:
    with (root / "meta" / "info.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_task_index(root: Path) -> tuple[dict[int, str], list[str]]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    tasks = [str(task) for task in table.column("task").to_pylist()]
    indices = [int(idx) for idx in table.column("task_index").to_pylist()]
    return dict(zip(indices, tasks, strict=True)), tasks


def _load_episode_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "index": int(frame["index"]),
        "episode_index": int(frame["episode_index"]),
        "frame_index": int(frame["frame_index"]),
        "timestamp": float(frame["timestamp"]),
    }
    if "task_index" in frame and frame["task_index"] is not None:
        normalized["task_index"] = int(frame["task_index"])
    if "observation.state" in frame:
        normalized["observation.state"] = [
            float(value) for value in _normalize_scalar(frame["observation.state"])
        ]
    return normalized


def _normalize_episode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "episode_index": int(row["episode_index"]),
        "task": row.get("task"),
        "tasks": [str(task) for task in row.get("tasks", [])],
        "frames": [_normalize_frame(frame) for frame in row["frames"]],
    }

    video = row.get("observation.images.main")
    if isinstance(video, mdr.Video):
        normalized["observation.images.main"] = {
            "from_timestamp_s": float(video.from_timestamp_s),
            "to_timestamp_s": float(video.to_timestamp_s),
        }

    return normalized


def _normalize_observation_state(value: Any) -> list[float]:
    scalar = _normalize_scalar(value)
    if isinstance(scalar, list):
        return [float(item) for item in scalar]
    return [float(scalar)]


def _normalize_hf_merged_rows(root: Path) -> list[dict[str, Any]]:
    info = _load_lerobot_info(root)
    rows: list[dict[str, Any]] = []

    for episode in _load_episode_rows(root):
        data_path = root / _format_chunked_path(
            str(info["data_path"]),
            chunk=episode["data/chunk_index"],
            file_index=int(episode["data/file_index"]),
        )
        frame_rows = pq.read_table(data_path).to_pylist()

        normalized_frames: list[dict[str, Any]] = []
        from_index = int(episode["dataset_from_index"])
        to_index = int(episode["dataset_to_index"])
        for frame in frame_rows:
            frame_index = int(frame["index"])
            if frame_index < from_index or frame_index >= to_index:
                continue

            normalized = _normalize_frame(frame)
            if "observation.state" in frame:
                normalized["observation.state"] = _normalize_observation_state(
                    frame["observation.state"]
                )
            normalized_frames.append(normalized)

        rows.append(
            {
                "episode_index": int(episode["episode_index"]),
                "task": episode.get("task"),
                "tasks": [str(task) for task in episode.get("tasks", [])],
                "frames": normalized_frames,
            }
        )

    rows.sort(key=lambda row: row["episode_index"])
    return rows


def _task_rows(root: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        task_name = row.get("task", row.get("__index_level_0__"))
        normalized.append(
            {
                "task_index": int(row["task_index"]),
                "task": str(task_name),
            }
        )
    normalized.sort(key=lambda row: row["task_index"])
    return normalized


def _merge_with_refiner_reader(
    roots: list[str | Path],
    out_root: Path,
    workdir: Path,
) -> None:
    merged_rows = mdr.read_lerobot([str(root) for root in roots]).materialize()
    stats = (
        mdr.from_items(merged_rows, items_per_shard=max(1, len(merged_rows)))
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-reader-merge-equivalence",
            num_workers=1,
            workdir=str(workdir),
        )
    )
    assert stats.failed == 0


def test_read_lerobot_multiple_roots_roundtrips_into_one_dataset(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "source-a"
    second_root = tmp_path / "source-b"
    _build_segmented_lerobot_source(first_root)
    _build_segmented_lerobot_source(second_root)

    out_root = tmp_path / "merged-refiner"
    stats = (
        mdr.read_lerobot([str(first_root), str(second_root)])
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-multi-root-merge",
            num_workers=1,
            workdir=str(tmp_path / "workdir-multi-root-merge"),
        )
    )
    assert stats.failed == 0

    info = _load_lerobot_info(out_root)
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 12

    rows = mdr.read_lerobot(str(out_root)).materialize()
    assert [int(row["episode_index"]) for row in rows] == [0, 1, 2, 3]


def _assert_state_stats_match_expected(
    stats_json: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    values = np.asarray(
        [
            frame["observation.state"]
            for row in rows
            for frame in row["frames"]
            if "observation.state" in frame
        ],
        dtype=np.float64,
    )
    assert values.size > 0

    np.testing.assert_allclose(
        np.asarray(stats_json["observation.state"]["min"], dtype=np.float64),
        values.min(axis=0),
    )
    np.testing.assert_allclose(
        np.asarray(stats_json["observation.state"]["max"], dtype=np.float64),
        values.max(axis=0),
    )
    np.testing.assert_allclose(
        np.asarray(stats_json["observation.state"]["mean"], dtype=np.float64),
        values.mean(axis=0),
    )
    assert int(stats_json["observation.state"]["count"]) == int(values.shape[0])


def _download_hub_lerobot_snapshot(repo_id: str) -> Path:
    snapshot_download = pytest.importorskip("huggingface_hub").snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


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
    assert "observation.images.main" in stats_json

    out_rows = mdr.read_lerobot(str(out_root)).materialize()
    assert [row["task"] for row in out_rows] == ["pick", "place"]


def test_write_lerobot_remuxes_full_source_video_runs(tmp_path: Path) -> None:
    src_root = tmp_path / "source-lerobot"
    _build_segmented_lerobot_source(src_root)

    out_root = tmp_path / "remuxed-full-source-lerobot"
    stats = (
        mdr.read_lerobot(str(src_root))
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-remux-full-source-run",
            num_workers=1,
            workdir=str(tmp_path / "workdir-remux-full-source"),
        )
    )
    assert stats.failed == 0

    out_videos = sorted(
        (out_root / "videos" / "observation.images.main").rglob("file-*.mp4")
    )
    assert len(out_videos) == 1
    out_video = out_videos[0]

    with av.open(str(out_video), mode="r") as container:
        stream = cast(
            Any, next(item for item in container.streams if item.type == "video")
        )
        duration_s = float(stream.duration * stream.time_base)
        codec = stream.codec_context.codec.canonical_name

    assert duration_s == pytest.approx(0.6, abs=0.05)
    assert codec == "mpeg4"

    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    episodes.sort(key=lambda row: int(row["episode_index"]))

    assert [
        (
            row["videos/observation.images.main/file_index"],
            row["videos/observation.images.main/from_timestamp"],
            row["videos/observation.images.main/to_timestamp"],
        )
        for row in episodes
    ] == [
        (0, 0.0, 0.3),
        (0, 0.3, 0.6),
    ]
    assert episodes[0]["stats/observation.images.main/mean"] == [
        [[0.2]],
        [[0.3]],
        [[0.4]],
    ]
    assert episodes[1]["stats/observation.images.main/mean"] == [
        [[0.7]],
        [[0.6]],
        [[0.5]],
    ]


def test_write_lerobot_remuxes_full_source_files_into_one_output(
    tmp_path: Path,
) -> None:
    src_root = tmp_path / "source-lerobot-remux"
    src_video_a, src_video_b = _build_multi_file_lerobot_source(src_root)

    out_root = tmp_path / "remuxed-lerobot"
    stats = (
        mdr.read_lerobot(str(src_root))
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-remux-full-source-files",
            num_workers=1,
            workdir=str(tmp_path / "workdir-remux"),
        )
    )
    assert stats.failed == 0

    out_videos = sorted(
        (out_root / "videos" / "observation.images.main").rglob("file-*.mp4")
    )
    assert len(out_videos) == 1
    assert out_videos[0].stat().st_size >= src_video_a.stat().st_size
    assert out_videos[0].stat().st_size >= src_video_b.stat().st_size

    with av.open(str(out_videos[0]), mode="r") as container:
        stream = cast(
            Any, next(item for item in container.streams if item.type == "video")
        )
        duration_s = float(stream.duration * stream.time_base)
        codec = stream.codec_context.codec.canonical_name

    assert duration_s == pytest.approx(0.6, abs=0.05)
    assert codec == "mpeg4"

    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    episodes.sort(key=lambda row: int(row["episode_index"]))
    assert [
        (
            row["videos/observation.images.main/file_index"],
            row["videos/observation.images.main/from_timestamp"],
            row["videos/observation.images.main/to_timestamp"],
        )
        for row in episodes
    ] == [
        (0, 0.0, 0.3),
        (0, 0.3, 0.6),
    ]
    assert episodes[0]["stats/observation.images.main/mean"] == [
        [[0.2]],
        [[0.3]],
        [[0.4]],
    ]
    assert episodes[1]["stats/observation.images.main/mean"] == [
        [[0.7]],
        [[0.6]],
        [[0.5]],
    ]


def test_write_lerobot_remuxes_aligned_subsegment_without_transcoding(
    tmp_path: Path,
) -> None:
    src_root = tmp_path / "source-lerobot-subsegment"
    _build_aligned_subsegment_lerobot_source(src_root)

    out_root = tmp_path / "remuxed-subsegment"
    stats = (
        mdr.read_lerobot(str(src_root))
        .write_lerobot(str(out_root))
        .launch_local(
            name="lerobot-remux-aligned-subsegment",
            num_workers=1,
            workdir=str(tmp_path / "workdir-subsegment"),
        )
    )
    assert stats.failed == 0

    out_videos = sorted(
        (out_root / "videos" / "observation.images.main").rglob("file-*.mp4")
    )
    assert len(out_videos) == 1

    with av.open(str(out_videos[0]), mode="r") as container:
        stream = cast(
            Any, next(item for item in container.streams if item.type == "video")
        )
        duration_s = float(stream.duration * stream.time_base)
        codec = stream.codec_context.codec.canonical_name

    assert duration_s == pytest.approx(0.2, abs=0.05)
    assert codec == "h264"

    info_json = _load_lerobot_info(out_root)
    assert (
        info_json["features"]["observation.images.main"]["info"]["video.codec"]
        == "h264"
    )

    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert len(episodes) == 1
    assert episodes[0][
        "videos/observation.images.main/from_timestamp"
    ] == pytest.approx(0.0, abs=0.01)
    assert episodes[0]["videos/observation.images.main/to_timestamp"] == pytest.approx(
        0.2, abs=0.05
    )
    assert episodes[0]["stats/observation.images.main/mean"] == [
        [[0.2]],
        [[0.3]],
        [[0.4]],
    ]


@pytest.mark.xfail(
    reason=(
        "Current reader+write_lerobot merge path does not match direct LeRobot "
        "merge output for rebased frame indices, task handling, and dataset metadata."
    )
)
def test_reader_merge_matches_lerobot_library_output(tmp_path: Path) -> None:
    if os.getenv(_RUN_HUB_LEROBOT_MERGE_TEST_ENV) != "1":
        pytest.skip(
            "Set REFINER_RUN_HUB_LEROBOT_MERGE_TEST=1 to run the Hub-backed "
            "LeRobot merge equivalence test."
        )

    merge_module = pytest.importorskip("lerobot.datasets.dataset_tools")
    dataset_module = pytest.importorskip("lerobot.datasets.lerobot_dataset")

    merge_datasets = merge_module.merge_datasets
    LeRobotDataset = dataset_module.LeRobotDataset

    snapshot_roots = [
        _download_hub_lerobot_snapshot(repo_id)
        for repo_id in _HUB_LEROBOT_MERGE_REPO_IDS
    ]

    merge_datasets(
        [
            LeRobotDataset(
                repo_id=repo_id,
                root=root,
                download_videos=False,
            )
            for repo_id, root in zip(
                _HUB_LEROBOT_MERGE_REPO_IDS,
                snapshot_roots,
                strict=True,
            )
        ],
        output_repo_id="merged-hf",
        output_dir=tmp_path / "merged-hf",
    )

    merged_root = tmp_path / "merged-refiner"
    _merge_with_refiner_reader(
        [f"hf://datasets/{repo_id}" for repo_id in _HUB_LEROBOT_MERGE_REPO_IDS],
        merged_root,
        tmp_path / "workdir-merged",
    )

    assert _normalize_hf_merged_rows(tmp_path / "merged-hf") == [
        _normalize_episode_row(row)
        for row in mdr.read_lerobot(str(merged_root)).materialize()
    ]
    assert _task_rows(tmp_path / "merged-hf") == _task_rows(merged_root)

    hf_info = _load_lerobot_info(tmp_path / "merged-hf")
    refiner_info = _load_lerobot_info(merged_root)
    assert hf_info == refiner_info

    hf_stats = json.loads((tmp_path / "merged-hf" / "meta" / "stats.json").read_text())
    refiner_stats = json.loads((merged_root / "meta" / "stats.json").read_text())
    assert hf_stats == refiner_stats


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
        .map_async(hydrate_video("observation.images.main"))
        .write_lerobot(str(out_root))
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


def test_lerobot_decoded_video_runs_are_non_extendable() -> None:
    decoded = DecodedVideo(
        frames=(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ),
        original_file=mdr.VideoFile("memory://decoded.mp4"),
        width=16,
        height=16,
    )
    first = decoded
    second = decoded
    run = _PendingVideoRun(
        video_key="observation.images.main",
        segments=[_PendingVideoSegment(episode_index=0, video=first)],
    )

    assert run_can_extend(run, second) is False


def test_lerobot_probe_cache_is_bounded() -> None:
    cache = _VideoProbeCache()

    for idx in range(_VIDEO_PROBE_CACHE_MAX_ENTRIES + 5):
        cache.cache(("observation.images.main", f"memory://video-{idx}.mp4"), None)

    assert len(cache._cache) == _VIDEO_PROBE_CACHE_MAX_ENTRIES
    assert ("observation.images.main", "memory://video-0.mp4") not in cache._cache
    assert (
        "observation.images.main",
        f"memory://video-{_VIDEO_PROBE_CACHE_MAX_ENTRIES + 4}.mp4",
    ) in cache._cache


def test_lerobot_video_writer_can_force_schedule_pending_run(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    writer = LeRobotVideoWriter(
        folder=DataFolder.resolve(tmp_path / "out"),
        chunk_key="000",
        video_key="observation.images.main",
        video_config=PipelineLeRobotVideoConfig(encoder_threads=1, decoder_threads=1),
        stats_config=PipelineLeRobotStatsConfig(sample_stride=1, quantile_bins=16),
        default_fps=10,
        video_bytes_limit=1024 * 1024,
        prepare_max_in_flight=1,
    )
    writer.submit(
        episode_index=0,
        video=mdr.VideoFile(str(src_video), from_timestamp_s=0.0, to_timestamp_s=0.3),
        source_stats={"max": np.zeros((3, 1, 1), dtype=np.float64)},
    )
    writer.submit(
        episode_index=1,
        video=mdr.VideoFile(str(src_video), from_timestamp_s=0.3, to_timestamp_s=0.6),
        source_stats={"max": np.zeros((3, 1, 1), dtype=np.float64)},
    )

    assert writer.drain_completed() == []

    deadline = time.time() + 5.0
    completed_runs: list[Any] = []
    while time.time() < deadline and not completed_runs:
        completed_runs = writer.drain_completed(force_schedule_pending_run=True)
        if completed_runs:
            break
        time.sleep(0.05)

    try:
        assert len(completed_runs) == 1
        assert [segment.episode_index for segment in completed_runs[0].segments] == [
            0,
            1,
        ]
    finally:
        writer.flush()


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
