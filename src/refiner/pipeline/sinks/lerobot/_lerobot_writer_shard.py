from __future__ import annotations

from concurrent.futures import wait
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.asyncio.runtime import submit
from refiner.io import DataFolder
from refiner.media import DecodedVideo, Video, VideoFile
from refiner.pipeline.sources.readers.lerobot import LEROBOT_INFO

from refiner.pipeline.sinks.lerobot._lerobot_frames import (
    compute_episode_stats,
    frame_table,
)
from refiner.pipeline.sinks.lerobot._lerobot_stats import (
    _flatten_stats_for_episode,
    _serialize_stats,
)
from refiner.pipeline.sinks.lerobot._lerobot_video import (
    _append_video_segment,
    _resolve_video_fps,
    VideoTrackWriter,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotVideoConfig,
        LeRobotWriterConfig,
    )


_DEFAULT_DATA_PATH = "data/chunk-{chunk_key}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4"
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

    _video_next_file_index: dict[str, int] = field(
        default_factory=lambda: defaultdict(int),
        init=False,
    )
    _video_track_writers: dict[str, VideoTrackWriter] = field(
        default_factory=dict,
        init=False,
    )
    _video_config: "LeRobotVideoConfig" = field(init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(
            self.config.root,
            fs=self.config.fs,
            storage_options=self.config.storage_options,
        )
        self._video_config = self.config.video

    @property
    def _meta_chunk_dir(self) -> str:
        return f"meta/chunk-{self.chunk_key}"

    def _meta_path(self, filename: str) -> str:
        return f"{self._meta_chunk_dir}/{filename}"

    @property
    def _data_bytes_limit(self) -> int:
        return int(self.config.data_files_size_in_mb) * 1024 * 1024

    @property
    def _video_bytes_limit(self) -> int:
        return int(self.config.video_files_size_in_mb) * 1024 * 1024

    @property
    def _has_videos(self) -> bool:
        return any(spec.get("dtype") == "video" for spec in self.features.values())

    def consume_row(self, row: Mapping[str, Any]) -> None:
        episode_index = int(row["episode_index"])
        frames = self._require_required_fields(row)
        tasks = self._tasks(row)

        self._initialize_info_from_row(row=row, frames=frames)

        data_meta = self._write_frames_to_data(
            episode_index=episode_index,
            tasks=tasks,
            frames=frames,
        )
        video_meta, video_stats = self._write_videos(row=row)
        episode_stats = compute_episode_stats(frames=frames, video_stats=video_stats)

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
            if isinstance(value, (VideoFile, DecodedVideo)):
                continue
            episode_row[key] = value

        episode_row.update(video_meta)
        episode_row.update(_flatten_stats_for_episode(episode_stats))

        self._append_episode_row(episode_row)
        self._append_stats_record(
            {
                "episode_index": episode_index,
                "stats": _serialize_stats(episode_stats),
            }
        )
        self._total_episodes += 1

    def finalize(self) -> None:
        self._flush_videos()
        self._close_data()
        self._close_episodes_writer()
        self._close_stats_file()

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
            "chunks_size": self.config.chunk_size,
            "data_files_size_in_mb": self.config.data_files_size_in_mb,
            "video_files_size_in_mb": self.config.video_files_size_in_mb,
            "data_path": _DEFAULT_DATA_PATH,
            "video_path": _DEFAULT_VIDEO_PATH if self._has_videos else None,
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

    def _append_episode_row(self, row: dict[str, Any]) -> None:
        if self._episodes_writer is not None:
            table = pa.Table.from_pylist([row])
            if self._episodes_schema != table.schema:
                raise ValueError(
                    "LeRobot writer requires a stable episode schema across rows"
                )
            self._episodes_writer.write_table(table)
            return

        episodes_rel = self._meta_path("episodes/file-000.parquet")
        table = pa.Table.from_pylist([row])
        self._episodes_schema = table.schema
        self._episodes_writer_file = self.folder.open(episodes_rel, mode="wb")
        self._episodes_writer = pq.ParquetWriter(
            self._episodes_writer_file,
            schema=table.schema,
            compression="snappy",
            use_dictionary=True,
        )
        self._episodes_writer.write_table(table)

    def _append_stats_record(self, record: dict[str, Any]) -> None:
        if self._stats_file is None:
            self._stats_file = self.folder.open(
                self._meta_path("stats.jsonl"),
                mode="wt",
                encoding="utf-8",
            )
        self._stats_file.write(json.dumps(record, sort_keys=True))
        self._stats_file.write("\n")

    def _close_episodes_writer(self) -> None:
        if self._episodes_writer is not None:
            self._episodes_writer.close()
            self._episodes_writer = None
        if self._episodes_writer_file is not None:
            self._episodes_writer_file.close()
            self._episodes_writer_file = None

    def _close_stats_file(self) -> None:
        if self._stats_file is not None:
            self._stats_file.close()
            self._stats_file = None

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

    def _write_videos(
        self,
        row: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
        out: dict[str, Any] = {}
        out_stats: dict[str, dict[str, np.ndarray]] = {}
        video_items = [
            (key, value)
            for key, value in row.items()
            if isinstance(value, (VideoFile, DecodedVideo))
        ]
        futures = []

        for key, value in video_items:
            futures.append(
                submit(
                    self._write_video_track(
                        video_key=key,
                        video=value,
                    )
                )
            )

        if not futures:
            return out, out_stats

        wait(futures)
        for future in futures:
            video_meta, video_stats = future.result()
            out.update(video_meta)
            out_stats.update(video_stats)

        return out, out_stats

    async def _write_video_track(
        self,
        *,
        video_key: str,
        video: Video,
    ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
        fps = await _resolve_video_fps(
            video=video,
            default_fps=self._fps,
            video_key=video_key,
        )
        writer = self._get_or_create_video_writer(
            video_key=video_key,
            fps=fps,
        )
        if writer.size_bytes >= self._video_bytes_limit:
            writer = self._rotate_video_writer(
                video_key=video_key,
                fps=fps,
            )

        from_ts = float(writer.duration_s)
        clip_duration_s, clip_stats = await _append_video_segment(
            writer=writer,
            video=video,
            clip_from=(
                float(video.from_timestamp_s)
                if isinstance(video, VideoFile) and video.from_timestamp_s is not None
                else 0.0
            ),
            clip_to=video.to_timestamp_s if isinstance(video, VideoFile) else None,
            video_config=self._video_config,
            stats_config=self.config.stats,
        )
        to_ts = float(from_ts + clip_duration_s)
        writer.duration_s = to_ts

        out: dict[str, Any] = {
            f"videos/{video_key}/chunk_index": writer.chunk_key,
            f"videos/{video_key}/file_index": writer.file_idx,
            f"videos/{video_key}/from_timestamp": from_ts,
            f"videos/{video_key}/to_timestamp": to_ts,
        }
        self._set_video_feature(video_key=video_key, writer=writer)

        return out, {video_key: clip_stats}

    def _set_video_feature(self, *, video_key: str, writer: VideoTrackWriter) -> None:
        stream = writer.stream
        if stream is None:
            return
        self.features[video_key] = {
            "dtype": "video",
            "shape": [3, int(stream.height), int(stream.width)],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": writer.fps,
                "video.height": int(stream.height),
                "video.width": int(stream.width),
                "video.channels": 3,
                "video.codec": self._video_config.codec,
                "video.pix_fmt": self._video_config.pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    def _get_or_create_video_writer(
        self,
        *,
        video_key: str,
        fps: int,
    ) -> VideoTrackWriter:
        target_file_idx = self._video_next_file_index[video_key]
        writer = self._video_track_writers.get(video_key)
        if writer is None or writer.file_idx != target_file_idx:
            return self._open_video_writer(
                video_key=video_key,
                file_idx=target_file_idx,
                fps=fps,
            )
        return writer

    def _rotate_video_writer(
        self,
        *,
        video_key: str,
        fps: int,
    ) -> VideoTrackWriter:
        self._flush_video_writer(video_key)
        return self._open_video_writer(
            video_key=video_key,
            file_idx=self._video_next_file_index[video_key],
            fps=fps,
        )

    def _open_video_writer(
        self,
        *,
        video_key: str,
        file_idx: int,
        fps: int,
    ) -> VideoTrackWriter:
        output_rel = _format_chunk_path(
            _DEFAULT_VIDEO_PATH,
            chunk=self.chunk_key,
            file_idx=file_idx,
            video_key=video_key,
        )
        output_abs = self.folder._join(output_rel)
        self.folder.fs.makedirs(self.folder.fs._parent(output_abs), exist_ok=True)
        output_file = self.folder.open(output_rel, mode="wb")
        writer = VideoTrackWriter(
            chunk_key=self.chunk_key,
            video_key=video_key,
            file_idx=file_idx,
            fps=fps,
            config=self._video_config,
        )
        try:
            writer.open(output_file)
        except Exception:
            output_file.close()
            raise
        self._video_next_file_index[video_key] = writer.file_idx
        self._video_track_writers[video_key] = writer
        return writer

    def _flush_video_writer(self, video_key: str) -> None:
        writer = self._video_track_writers.pop(video_key, None)
        if writer is None:
            return

        writer.close()
        self._video_next_file_index[video_key] = writer.file_idx + 1

    def _flush_videos(self) -> None:
        for key in list(self._video_track_writers.keys()):
            self._flush_video_writer(key)


__all__ = [
    "_LeRobotShardWriter",
    "_DEFAULT_DATA_PATH",
    "_DEFAULT_VIDEO_PATH",
    "_DEFAULT_CODEBASE_VERSION",
]
