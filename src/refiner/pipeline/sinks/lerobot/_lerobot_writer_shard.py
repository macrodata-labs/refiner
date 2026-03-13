from __future__ import annotations

from concurrent.futures import wait
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.asyncio.runtime import submit
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sources.readers.lerobot import LEROBOT_INFO

from refiner.pipeline.sinks.lerobot._lerobot_frames import (
    compute_episode_stats,
    collect_episode_tasks,
    infer_features,
    _arrow_frame_table,
    resolve_task_index,
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
    available_cpus: int | None = None,
) -> int | None:
    if requested_threads is not None:
        return int(requested_threads)
    if videos_in_row <= 0:
        return None
    resolved_cpus = (
        available_cpus if available_cpus is not None else _cpu_thread_count()
    )
    return max(1, int(resolved_cpus) // int(videos_in_row))


def _resolve_video_encoder_threads(
    *,
    requested_threads: int | None,
    videos_in_row: int,
    available_cpus: int | None = None,
) -> int | None:
    return _resolve_video_threads(
        requested_threads=requested_threads,
        videos_in_row=videos_in_row,
        available_cpus=available_cpus,
    )


def _resolve_video_decoder_threads(
    *,
    requested_threads: int | None,
    videos_in_row: int,
    available_cpus: int | None = None,
) -> int | None:
    return _resolve_video_threads(
        requested_threads=requested_threads,
        videos_in_row=videos_in_row,
        available_cpus=available_cpus,
    )


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
    config: Any
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
    _features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

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
    _video_append_config: dict[str, Any] = field(default_factory=dict, init=False)
    _resolved_video_encoder_threads: int | None = field(default=None, init=False)
    _resolved_video_decoder_threads: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(
            self.config.root,
            fs=self.config.fs,
            storage_options=self.config.storage_options,
        )
        self._video_append_config = {
            "video_codec": self.config.video_codec,
            "video_pix_fmt": self.config.video_pix_fmt,
            "video_encoder_threads": self.config.video_encoder_threads,
            "video_decoder_threads": self.config.video_decoder_threads,
            "video_encoder_options": self.config.video_encoder_options,
            "enable_video_stats": self.config.enable_video_stats,
            "video_stats_sample_stride": self.config.video_stats_sample_stride,
            "video_stats_quantile_bins": self.config.video_stats_quantile_bins,
        }

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

    def consume_row(self, row: Mapping[str, Any]) -> None:
        episode_index = int(row["episode_index"])
        frames = self._require_required_fields(row)

        tasks = collect_episode_tasks(
            tasks_raw=row["tasks"],
            task_raw=row["task"],
        )
        for task in tasks:
            resolve_task_index(self._task_to_index, self._task_order, task)

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
            if key.startswith("__"):
                continue
            if isinstance(value, Video):
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
            "features": self._features,
            "chunks_size": self.config.chunk_size,
            "data_files_size_in_mb": self.config.data_files_size_in_mb,
            "video_files_size_in_mb": self.config.video_files_size_in_mb,
            "data_path": _DEFAULT_DATA_PATH,
            "video_path": (
                _DEFAULT_VIDEO_PATH
                if any(ft.get("dtype") == "video" for ft in self._features.values())
                else None
            ),
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
        return list(frames_raw)

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

    def _initialize_info_from_row(
        self,
        row: Mapping[str, Any],
        frames: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._features:
            return

        metadata = row["metadata"]
        if isinstance(metadata, Mapping):
            info = metadata[LEROBOT_INFO]
            if isinstance(info, Mapping):
                self._fps = int(info["fps"])
                robot_type_raw = info["robot_type"]
                self._robot_type = (
                    str(robot_type_raw) if robot_type_raw is not None else None
                )

        infer_features(features=self._features, row=row, frames=frames)
        video_features = sum(
            1 for spec in self._features.values() if spec.get("dtype") == "video"
        )
        self._resolved_video_encoder_threads = _resolve_video_encoder_threads(
            requested_threads=self.config.video_encoder_threads,
            videos_in_row=video_features,
        )
        self._resolved_video_decoder_threads = _resolve_video_decoder_threads(
            requested_threads=self.config.video_decoder_threads,
            videos_in_row=video_features,
        )
        self._video_append_config["video_encoder_threads"] = (
            self._resolved_video_encoder_threads
        )
        self._video_append_config["video_decoder_threads"] = (
            self._resolved_video_decoder_threads
        )

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

        default_task_idx = self._task_to_index.get(tasks[0]) if tasks else None
        table = _arrow_frame_table(
            frames=frames,
            episode_index=episode_index,
            start_index=start_index,
            default_task_idx=default_task_idx,
        )
        if table is None:
            frame_rows: list[dict[str, Any]] = []
            for local_idx, frame in enumerate(frames):
                out = dict(frame)
                out["episode_index"] = episode_index
                out["index"] = start_index + local_idx
                if "frame_index" not in out:
                    out["frame_index"] = local_idx
                if "task_index" not in out and default_task_idx is not None:
                    out["task_index"] = default_task_idx
                frame_rows.append(out)
            table = pa.Table.from_pylist(frame_rows)

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
            (key, value) for key, value in row.items() if isinstance(value, Video)
        ]
        video_config = self._video_append_config
        futures = []

        for key, value in video_items:
            clip_from = (
                float(value.from_timestamp_s)
                if value.from_timestamp_s is not None
                else 0.0
            )
            clip_to = float(value.to_timestamp_s)
            futures.append(
                submit(
                    self._write_video_track(
                        video_key=key,
                        video=value,
                        clip_from=clip_from,
                        clip_to=clip_to,
                        video_config=video_config,
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
        clip_from: float,
        clip_to: float,
        video_config: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
        fps = await _resolve_video_fps(
            video=video,
            default_fps=self._fps,
            video_key=video_key,
        )
        writer = self._get_or_create_video_writer(
            video_key=video_key,
            fps=fps,
            video_config=video_config,
        )
        if writer.size_bytes >= self._video_bytes_limit:
            writer = self._rotate_video_writer(
                video_key=video_key,
                fps=fps,
                video_config=video_config,
            )

        from_ts = float(writer.duration_s)
        clip_duration_s, clip_stats = await _append_video_segment(
            writer=writer,
            video=video,
            video_key=video_key,
            clip_from=clip_from,
            clip_to=clip_to,
            video_config=video_config,
        )
        to_ts = float(from_ts + clip_duration_s)
        writer.duration_s = to_ts

        out: dict[str, Any] = {
            f"videos/{video_key}/chunk_index": writer.chunk_key,
            f"videos/{video_key}/file_index": writer.file_idx,
            f"videos/{video_key}/from_timestamp": from_ts,
            f"videos/{video_key}/to_timestamp": to_ts,
        }

        out_stats = {}
        if clip_stats is not None:
            out_stats[video_key] = clip_stats
        return out, out_stats

    def _get_or_create_video_writer(
        self,
        *,
        video_key: str,
        fps: int,
        video_config: Mapping[str, Any],
    ) -> VideoTrackWriter:
        target_file_idx = self._video_next_file_index[video_key]
        writer = self._video_track_writers.get(video_key)
        if writer is None or writer.file_idx != target_file_idx:
            return self._open_video_writer(
                video_key=video_key,
                file_idx=target_file_idx,
                fps=fps,
                video_config=video_config,
            )
        return writer

    def _rotate_video_writer(
        self,
        *,
        video_key: str,
        fps: int,
        video_config: Mapping[str, Any],
    ) -> VideoTrackWriter:
        self._flush_video_writer(video_key)
        return self._open_video_writer(
            video_key=video_key,
            file_idx=self._video_next_file_index[video_key],
            fps=fps,
            video_config=video_config,
        )

    def _open_video_writer(
        self,
        *,
        video_key: str,
        file_idx: int,
        fps: int,
        video_config: Mapping[str, Any],
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
            config=video_config,
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
