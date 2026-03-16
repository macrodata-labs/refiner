from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io import DataFolder
from refiner.media import DecodedVideo, Video, VideoFile
from refiner.pipeline.sources.readers.lerobot import (
    LEROBOT_EPISODE_STATS,
    LEROBOT_INFO,
)
from refiner.pipeline.sinks.lerobot._lerobot_frames import (
    compute_episode_stats,
    frame_table,
)
from refiner.pipeline.sinks.lerobot._lerobot_stats import (
    _cast_stats_to_numpy,
    _flatten_stats_for_episode,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_writer import (
    DEFAULT_VIDEO_PATH,
    LeRobotVideoWriter,
    _CompletedVideoRun,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotVideoConfig,
        LeRobotWriterConfig,
    )


_DEFAULT_DATA_PATH = "data/chunk-{chunk_key}/file-{file_index:03d}.parquet"
_DEFAULT_CODEBASE_VERSION = "v3.0"


def _cpu_thread_count() -> int:
    try:
        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        if sched_getaffinity is None:
            raise AttributeError
        return max(1, len(sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_video_threads(
    *,
    requested_threads: int | None,
    videos_in_row: int,
) -> int | None:
    if requested_threads is not None:
        return int(requested_threads)
    if videos_in_row <= 0:
        return None
    return max(1, _cpu_thread_count() // int(videos_in_row))


def _format_chunk_path(
    template: str,
    *,
    chunk: str,
    file_idx: int,
    video_key: str | None = None,
) -> str:
    return template.format(
        video_key="" if video_key is None else video_key,
        chunk=chunk,
        chunk_key=chunk,
        chunk_index=chunk,
        file=file_idx,
        file_idx=file_idx,
        file_index=file_idx,
    )


@dataclass(slots=True)
class _PendingEpisode:
    row: dict[str, Any]
    frame_stats: dict[str, dict[str, np.ndarray]]
    pending_video_keys: set[str]
    video_stats: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass(slots=True)
class _LeRobotShardWriter:
    config: "LeRobotWriterConfig"
    chunk_key: str
    folder: DataFolder = field(init=False)

    _task_to_index: dict[str, int] = field(default_factory=dict, init=False)
    _task_order: list[str] = field(default_factory=list, init=False)

    _episodes_writer: pq.ParquetWriter | None = field(default=None, init=False)
    _episodes_writer_file: IO[bytes] | None = field(default=None, init=False)
    _episodes_schema: pa.Schema | None = field(default=None, init=False)
    _stats_file: IO[str] | None = field(default=None, init=False)

    _fps: int | None = field(default=None, init=False)
    _robot_type: str | None = field(default=None, init=False)
    features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    _total_episodes: int = field(default=0, init=False)
    _global_frame_index: int = field(default=0, init=False)

    _data_file_index: int = field(default=0, init=False)
    _data_bytes_written: int = field(default=0, init=False)
    _data_writer: pq.ParquetWriter | None = field(default=None, init=False)
    _data_writer_file: IO[bytes] | None = field(default=None, init=False)
    _data_schema: pa.Schema | None = field(default=None, init=False)

    _pending_episodes: OrderedDict[int, _PendingEpisode] = field(
        default_factory=OrderedDict,
        init=False,
    )
    _video_writers: dict[str, LeRobotVideoWriter] = field(
        default_factory=dict, init=False
    )
    _video_config: "LeRobotVideoConfig" = field(init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(self.config.output)
        self._video_config = self.config.video

    def _meta_path(self, filename: str) -> str:
        return f"meta/chunk-{self.chunk_key}/{filename}"

    @property
    def _data_bytes_limit(self) -> int:
        return int(self.config.data_files_size_in_mb) * 1024 * 1024

    @property
    def _video_bytes_limit(self) -> int:
        return int(self.config.video_files_size_in_mb) * 1024 * 1024

    @property
    def _pending_episode_soft_limit(self) -> int:
        return int(self.config.max_buffered_episodes) * max(1, len(self._video_writers))

    @property
    def _has_videos(self) -> bool:
        return any(spec.get("dtype") == "video" for spec in self.features.values())

    def consume_row(self, row: Mapping[str, Any]) -> None:
        episode_index = int(row["episode_index"])
        frames = self._require_required_fields(row)
        tasks = self._tasks(row)
        source_episode_stats = self._source_episode_stats(row)

        self._initialize_info_from_row(row=row, frames=frames)

        data_meta = self._write_frames_to_data(
            episode_index=episode_index,
            tasks=tasks,
            frames=frames,
        )
        frame_stats = compute_episode_stats(frames=frames)

        episode_row: dict[str, Any] = {
            "episode_index": episode_index,
            "length": len(frames),
            "tasks": tasks,
            "data/chunk_index": data_meta["chunk_index"],
            "data/file_index": data_meta["file_index"],
            "dataset_from_index": data_meta["dataset_from_index"],
            "dataset_to_index": data_meta["dataset_to_index"],
            "meta/episodes/chunk_index": self.chunk_key,
            "meta/episodes/file_index": 0,
        }
        if tasks:
            episode_row["task"] = tasks[0]

        video_items: list[tuple[str, Video]] = []
        for key, value in row.items():
            if key in {
                "frames",
                "task",
                "tasks",
                "metadata",
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            }:
                continue
            if isinstance(value, Video):
                video_items.append((key, value))
                continue
            episode_row[key] = value

        self._pending_episodes[episode_index] = _PendingEpisode(
            row=episode_row,
            frame_stats=frame_stats,
            pending_video_keys={key for key, _ in video_items},
        )

        for key, value in video_items:
            self._queue_video_segment(
                episode_index=episode_index,
                video_key=key,
                video=value,
                source_stats=source_episode_stats.get(key),
            )

        self._drain_video_writers()
        self._flush_ready_episodes()
        while len(self._pending_episodes) > self._pending_episode_soft_limit:
            pending_before = len(self._pending_episodes)
            self._drain_video_writers(force_schedule_pending_runs=True)
            self._flush_ready_episodes()
            if len(self._pending_episodes) >= pending_before:
                break
        self._total_episodes += 1

    def finalize(self) -> None:
        for writer in self._video_writers.values():
            self._process_video_runs(writer.flush())
        self._flush_ready_episodes()
        if self._pending_episodes:
            pending = sorted(self._pending_episodes)
            raise RuntimeError(
                "LeRobot writer finalized with unresolved episode video metadata: "
                f"{pending!r}"
            )

        self._video_writers.clear()
        self._close_data()
        if self._episodes_writer is not None:
            self._episodes_writer.close()
            self._episodes_writer = None
        if self._episodes_writer_file is not None:
            self._episodes_writer_file.close()
            self._episodes_writer_file = None
        if self._stats_file is not None:
            self._stats_file.close()
            self._stats_file = None

        task_lines = [
            json.dumps(
                {"task": task, "task_index": task_index},
                sort_keys=True,
            )
            for task_index, task in enumerate(self._task_order)
        ]
        if task_lines:
            with self.folder.open(
                self._meta_path("tasks.jsonl"),
                mode="wt",
                encoding="utf-8",
            ) as out:
                out.write("\n".join(task_lines))
                out.write("\n")

        info_record = {
            "codebase_version": _DEFAULT_CODEBASE_VERSION,
            "fps": self._fps,
            "robot_type": self._robot_type,
            "features": self.features,
            "data_files_size_in_mb": self.config.data_files_size_in_mb,
            "video_files_size_in_mb": self.config.video_files_size_in_mb,
            "data_path": _DEFAULT_DATA_PATH,
            "video_path": DEFAULT_VIDEO_PATH if self._has_videos else None,
            "total_episodes": self._total_episodes,
            "total_frames": self._global_frame_index,
        }

        with self.folder.open(
            self._meta_path("info.jsonl"),
            mode="wt",
            encoding="utf-8",
        ) as out:
            out.write(json.dumps(info_record, sort_keys=True))
            out.write("\n")

    def _require_required_fields(
        self,
        row: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        frames_raw = row["frames"]
        if not isinstance(frames_raw, list):
            raise ValueError("LeRobot writer requires frames as a list on each row")
        if not all(isinstance(item, Mapping) for item in frames_raw):
            raise ValueError("LeRobot writer requires each frame to be a mapping")
        return [item for item in frames_raw if isinstance(item, Mapping)]

    def _source_episode_stats(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, dict[str, np.ndarray]]:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            return {}
        raw_stats = metadata.get(LEROBOT_EPISODE_STATS)
        if not isinstance(raw_stats, Mapping):
            return {}
        return _cast_stats_to_numpy(raw_stats)

    def _queue_video_segment(
        self,
        *,
        episode_index: int,
        video_key: str,
        video: Video,
        source_stats: dict[str, np.ndarray] | None,
    ) -> None:
        writer = self._video_writers.get(video_key)
        if writer is None:
            writer = LeRobotVideoWriter(
                folder=self.folder,
                chunk_key=self.chunk_key,
                video_key=video_key,
                video_config=self._video_config,
                stats_config=self.config.stats,
                default_fps=self._fps,
                video_bytes_limit=self._video_bytes_limit,
                prepare_max_in_flight=self.config.max_buffered_episodes,
            )
            self._video_writers[video_key] = writer

        writer.submit(
            episode_index=episode_index,
            video=video,
            source_stats=source_stats,
        )

    def _drain_video_writers(
        self, *, force_schedule_pending_runs: bool = False
    ) -> None:
        for writer in self._video_writers.values():
            self._process_video_runs(
                writer.drain_completed(
                    force_schedule_pending_run=force_schedule_pending_runs
                )
            )

    def _process_video_runs(
        self,
        completed_runs: Sequence[_CompletedVideoRun],
    ) -> None:
        for run in completed_runs:
            if run.feature is not None:
                self.features[run.video_key] = run.feature
            for segment in run.segments:
                pending = self._pending_episodes.get(segment.episode_index)
                if pending is None:
                    raise RuntimeError(
                        f"Missing pending episode for {segment.episode_index}"
                    )
                pending.row = pending.row.update(
                    {
                        f"videos/{segment.video_key}/chunk_index": segment.chunk_key,
                        f"videos/{segment.video_key}/file_index": segment.file_index,
                        f"videos/{segment.video_key}/from_timestamp": segment.from_timestamp,
                        f"videos/{segment.video_key}/to_timestamp": segment.to_timestamp,
                    }
                )
                pending.video_stats[segment.video_key] = segment.stats
                pending.pending_video_keys.discard(segment.video_key)

    def _flush_ready_episodes(self) -> None:
        while self._pending_episodes:
            episode_index = next(iter(self._pending_episodes))
            pending = self._pending_episodes[episode_index]
            if pending.pending_video_keys:
                break

            episode_stats = dict(pending.frame_stats)
            episode_stats.update(pending.video_stats)
            pending.row.update(_flatten_stats_for_episode(episode_stats))
            table = pa.Table.from_pylist([pending.row])
            if self._episodes_writer is not None:
                if self._episodes_schema != table.schema:
                    raise ValueError(
                        "LeRobot writer requires a stable episode schema across rows"
                    )
                self._episodes_writer.write_table(table)
            else:
                episodes_rel = self._meta_path("episodes/file-000.parquet")
                self._episodes_schema = table.schema
                self._episodes_writer_file = self.folder.open(episodes_rel, mode="wb")
                self._episodes_writer = pq.ParquetWriter(
                    self._episodes_writer_file,
                    schema=table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
                self._episodes_writer.write_table(table)
            self._pending_episodes.popitem(last=False)

    def _tasks(self, row: Mapping[str, Any]) -> list[str]:
        tasks = [
            value
            for value in [
                *(row["tasks"] if isinstance(row["tasks"], list) else []),
                row["task"],
            ]
            if isinstance(value, str) and value
        ]
        tasks = list(dict.fromkeys(tasks))
        for task in tasks:
            if task in self._task_to_index:
                continue
            self._task_to_index[task] = len(self._task_order)
            self._task_order.append(task)
        return tasks

    def _initialize_info_from_row(
        self,
        row: Mapping[str, Any],
        frames: Sequence[Mapping[str, Any]],
    ) -> None:
        if self.features:
            return

        metadata = row["metadata"]
        if isinstance(metadata, Mapping):
            info = metadata.get(LEROBOT_INFO)
            if isinstance(info, Mapping):
                fps_raw = info.get("fps")
                self._fps = int(fps_raw) if fps_raw is not None else None
                robot_type_raw = info.get("robot_type")
                self._robot_type = (
                    str(robot_type_raw) if robot_type_raw is not None else None
                )

        for key, value in row.items():
            if key in {"frames", "task", "tasks", "metadata", "episode_index"}:
                continue
            spec = self._feature_spec(value)
            if spec is not None:
                self.features.setdefault(key, spec)

        if frames:
            for key, value in frames[0].items():
                spec = self._feature_spec(value)
                if spec is not None:
                    self.features.setdefault(key, spec)

        for key, spec in {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }.items():
            self.features.setdefault(key, spec)

        video_count = sum(
            1 for value in row.values() if isinstance(value, (VideoFile, DecodedVideo))
        )
        self._video_config = replace(
            self.config.video,
            encoder_threads=_resolve_video_threads(
                requested_threads=self.config.video.encoder_threads,
                videos_in_row=video_count,
            ),
            decoder_threads=_resolve_video_threads(
                requested_threads=self.config.video.decoder_threads,
                videos_in_row=video_count,
            ),
        )

    def _feature_spec(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, (VideoFile, DecodedVideo)):
            return {"dtype": "video", "shape": None, "names": None, "info": None}
        if isinstance(value, bool):
            return {"dtype": "bool", "shape": [1], "names": None}

        array = np.asarray(value)
        if array.ndim == 0:
            if array.dtype.kind in {"i", "u"}:
                return {"dtype": "int64", "shape": [1], "names": None}
            if array.dtype.kind == "f":
                return {"dtype": "float64", "shape": [1], "names": None}
            return None

        if array.dtype.kind in {"i", "u"}:
            dtype = "int64"
        elif array.dtype.kind == "f":
            dtype = "float64"
        elif array.dtype.kind == "b":
            dtype = "bool"
        else:
            return None
        return {
            "dtype": dtype,
            "shape": [int(size) for size in array.shape],
            "names": None,
        }

    def _write_frames_to_data(
        self,
        *,
        episode_index: int,
        tasks: list[str],
        frames: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        start_index = self._global_frame_index
        current_chunk = self.chunk_key
        current_file = self._data_file_index

        if not frames:
            return {
                "chunk_index": current_chunk,
                "file_index": current_file,
                "dataset_from_index": start_index,
                "dataset_to_index": start_index,
            }

        table = frame_table(
            frames=frames,
            episode_index=episode_index,
            start_index=start_index,
            task_index=self._task_to_index.get(tasks[0]) if tasks else None,
        )

        if (
            self._data_writer is not None
            and self._data_bytes_written >= self._data_bytes_limit
        ):
            self._close_data()
            self._data_file_index += 1
            self._data_bytes_written = 0
            current_chunk = self.chunk_key
            current_file = self._data_file_index

        self._ensure_data_writer(table.schema)
        if self._data_writer is None:
            raise RuntimeError("data writer is not initialized")

        data_writer_file = self._data_writer_file
        if data_writer_file is None:
            raise RuntimeError("data writer file is not initialized")

        self._data_writer.write_table(table)
        self._data_bytes_written = data_writer_file.tell()

        self._global_frame_index += int(table.num_rows)
        return {
            "chunk_index": current_chunk,
            "file_index": current_file,
            "dataset_from_index": start_index,
            "dataset_to_index": self._global_frame_index,
        }

    def _ensure_data_writer(self, schema: pa.Schema) -> None:
        if self._data_writer is not None:
            if self._data_schema is None:
                self._data_schema = schema
                return
            if self._data_schema != schema:
                raise ValueError(
                    "LeRobot writer requires a stable frame schema across episodes"
                )
            return

        self._data_schema = schema
        rel = _format_chunk_path(
            _DEFAULT_DATA_PATH,
            chunk=self.chunk_key,
            file_idx=self._data_file_index,
        )
        self._data_writer_file = self.folder.open(rel, mode="wb")
        self._data_writer = pq.ParquetWriter(
            self._data_writer_file,
            schema=schema,
            compression="snappy",
            use_dictionary=True,
        )
        self._data_bytes_written = 0

    def _close_data(self) -> None:
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        if self._data_writer_file is not None:
            self._data_writer_file.close()
            self._data_writer_file = None
        self._data_bytes_written = 0


__all__ = [
    "_DEFAULT_CODEBASE_VERSION",
    "_DEFAULT_DATA_PATH",
    "_LeRobotShardWriter",
]
