from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from refiner.io import DataFolder
from refiner.media import VideoFile
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
    _CompletedVideoItem,
)
from refiner.pipeline.sources.readers.lerobot import (
    LEROBOT_EPISODE_STATS,
    LEROBOT_INFO,
    LEROBOT_TASKS,
)

if TYPE_CHECKING:
    from refiner.pipeline.data.row import Row
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotVideoConfig,
        LeRobotWriterConfig,
    )


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index}/file-{file_index:03d}.parquet"
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


@dataclass(slots=True)
class _FrameWriteInfo:
    episode_index: int
    length: int
    tasks: list[str]
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    frame_stats: dict[str, dict[str, np.ndarray]]


@dataclass(slots=True)
class _LeRobotShardWriter:
    config: "LeRobotWriterConfig"
    chunk_key: str
    folder: DataFolder = field(init=False)

    _task_to_index: dict[str, int] = field(default_factory=dict, init=False)
    _task_order: list[str] = field(default_factory=list, init=False)
    _fps: int | None = field(default=None, init=False)
    _robot_type: str | None = field(default=None, init=False)
    features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _video_config: "LeRobotVideoConfig" = field(init=False)

    _total_episodes: int = field(default=0, init=False)
    _global_frame_index: int = field(default=0, init=False)

    _data_file_index: int = field(default=0, init=False)
    _data_bytes_written: int = field(default=0, init=False)
    _data_writer: pq.ParquetWriter | None = field(default=None, init=False)
    _data_writer_file: Any = field(default=None, init=False)
    _data_schema: pa.Schema | None = field(default=None, init=False)

    _video_writers: dict[str, LeRobotVideoWriter] = field(
        default_factory=dict, init=False
    )
    _episode_rows: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(self.config.output)
        self._video_config = self.config.video

    def finalize(self) -> None:
        for writer in self._video_writers.values():
            writer.finalize()
        self._video_writers.clear()
        self._close_data()
        self._write_stage1_meta()

    def _meta_path(self, filename: str) -> str:
        return f"meta/chunk-{self.chunk_key}/{filename}"

    @property
    def _data_bytes_limit(self) -> int:
        return int(self.config.data_files_size_in_mb) * 1024 * 1024

    @property
    def _video_bytes_limit(self) -> int:
        return int(self.config.video_files_size_in_mb) * 1024 * 1024

    @property
    def _has_videos(self) -> bool:
        return any(spec.get("dtype") == "video" for spec in self.features.values())

    async def write_row(
        self,
        *,
        row: Row | Mapping[str, Any],
    ) -> None:
        if not row["frames"]:
            # all frames were trimmed or similar
            return
        self._initialize_info_from_rows([row])
        episode_index = int(row["episode_index"])
        source_episode_stats = self._source_episode_stats(row)

        writer_tasks: list[asyncio.Task[_CompletedVideoItem]] = []
        for video_key, video in row.items():
            if not isinstance(video, VideoFile):
                continue

            writer_tasks.append(
                asyncio.create_task(
                    self._video_writer(video_key).write_video(
                        video=video,
                        episode_index=episode_index,
                        source_stats=source_episode_stats.get(video_key),
                    )
                )
            )

        frame_write_task = asyncio.create_task(self._write_frame_table(row))

        try:
            frame_info = await frame_write_task
            video_info = await asyncio.gather(*writer_tasks) if writer_tasks else []
        except Exception:
            for task in writer_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*writer_tasks, return_exceptions=True)
            if not frame_write_task.done():
                frame_write_task.cancel()
                await asyncio.gather(frame_write_task, return_exceptions=True)
            # TODO: decide how to clean up partially written data/video artifacts.
            raise

        episode_row = self._write_episode_row(
            row=row,
            frame_info=frame_info,
            video_info=video_info,
        )
        self._episode_rows.append(episode_row)
        self._total_episodes += 1

    def _video_writer(self, video_key: str) -> LeRobotVideoWriter:
        writer = self._video_writers.get(video_key)
        if writer is not None:
            return writer

        writer = LeRobotVideoWriter(
            folder=self.folder,
            chunk_key=self.chunk_key,
            video_key=video_key,
            video_config=self._video_config,
            stats_config=self.config.stats,
            video_bytes_limit=self._video_bytes_limit,
        )
        self._video_writers[video_key] = writer
        return writer

    async def _write_frame_table(
        self,
        row: Row | Mapping[str, Any],
    ) -> _FrameWriteInfo:
        episode_index = int(row["episode_index"])
        frames = self._require_required_fields(row)
        frame_stats = compute_episode_stats(frames=frames)
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "LeRobot writer requires metadata with canonical "
                f"'{LEROBOT_TASKS}' entries"
            )
        raw_tasks = metadata.get(LEROBOT_TASKS)
        if not isinstance(raw_tasks, Mapping):
            raise ValueError(
                "LeRobot writer requires metadata."
                f"{LEROBOT_TASKS} as a mapping of task to task_index"
            )

        task_to_index: dict[str, int] = {}
        for task, task_index_raw in raw_tasks.items():
            if not isinstance(task, str) or not task:
                raise ValueError(
                    "LeRobot writer requires each metadata."
                    f"{LEROBOT_TASKS} key to be a non-empty task string"
                )
            try:
                task_to_index[task] = int(task_index_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "LeRobot writer requires each metadata."
                    f"{LEROBOT_TASKS} value to be an integer task_index"
                ) from exc

        max_task_index = max(task_to_index.values(), default=-1)
        task_names_by_index = [""] * (max_task_index + 1)
        for task, task_index in task_to_index.items():
            existing_task = task_names_by_index[task_index]
            if existing_task and existing_task != task:
                raise ValueError(
                    "LeRobot writer encountered conflicting canonical task mappings "
                    f"for task_index={task_index}"
                )
            task_names_by_index[task_index] = task
            existing_index = self._task_to_index.get(task)
            if existing_index is None:
                self._task_to_index[task] = task_index
            elif existing_index != task_index:
                raise ValueError(
                    "LeRobot writer encountered conflicting canonical task indices "
                    f"for task={task!r}"
                )

        if any(not task for task in task_names_by_index):
            raise ValueError(
                "LeRobot writer requires metadata."
                f"{LEROBOT_TASKS} to define a contiguous task_index space"
            )

        if self._task_to_index:
            max_task_index = max(self._task_to_index.values())
            self._task_order = [""] * (max_task_index + 1)
            for task, task_index in self._task_to_index.items():
                existing_task = self._task_order[task_index]
                if existing_task and existing_task != task:
                    raise ValueError(
                        "LeRobot writer encountered conflicting canonical task names "
                        f"for task_index={task_index}"
                    )
                self._task_order[task_index] = task
            if any(not task for task in self._task_order):
                raise ValueError(
                    "LeRobot writer requires metadata."
                    f"{LEROBOT_TASKS} to define a contiguous task_index space"
                )

        frame_task_indices: list[int] = []
        for frame_idx, frame in enumerate(frames):
            task_index_raw = frame.get("task_index")
            if task_index_raw is None:
                raise ValueError(
                    "LeRobot writer requires every frame to contain an integer "
                    f"'task_index'; frame {frame_idx} is missing a usable value"
                )
            try:
                frame_task_indices.append(int(task_index_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "LeRobot writer requires every frame to contain an integer "
                    f"'task_index'; frame {frame_idx} is missing a usable value"
                ) from exc

        unmapped_task_index = next(
            (
                task_index
                for task_index in frame_task_indices
                if task_index < 0 or task_index >= len(task_names_by_index)
            ),
            None,
        )
        if unmapped_task_index is not None:
            raise ValueError(
                "LeRobot writer could not map frame task_index="
                f"{unmapped_task_index} through metadata.{LEROBOT_TASKS}"
            )

        tasks = (
            list(
                dict.fromkeys(
                    pc.take(
                        pa.array(task_names_by_index, type=pa.string()),
                        pa.array(frame_task_indices, type=pa.int64()),
                    ).to_pylist()
                )
            )
            if frame_task_indices
            else []
        )
        table = (
            frame_table(
                frames=frames,
                episode_index=episode_index,
                start_index=self._global_frame_index,
                task_index=None,
            )
            if frames
            else None
        )
        data_file_index, dataset_from_index, dataset_to_index = (
            self._append_frame_table(table)
        )
        return _FrameWriteInfo(
            episode_index=episode_index,
            length=len(frames),
            tasks=tasks,
            data_file_index=data_file_index,
            dataset_from_index=dataset_from_index,
            dataset_to_index=dataset_to_index,
            frame_stats=frame_stats,
        )

    def _write_episode_row(
        self,
        *,
        row: Row | Mapping[str, Any],
        frame_info: _FrameWriteInfo,
        video_info: list[_CompletedVideoItem],
    ) -> dict[str, Any]:
        episode_row: dict[str, Any] = {
            "episode_index": frame_info.episode_index,
            "length": frame_info.length,
            "tasks": frame_info.tasks,
            "data/chunk_index": self.chunk_key,
            "data/file_index": frame_info.data_file_index,
            "dataset_from_index": frame_info.dataset_from_index,
            "dataset_to_index": frame_info.dataset_to_index,
            "meta/episodes/chunk_index": self.chunk_key,
            "meta/episodes/file_index": 0,
        }

        for key, value in row.items():
            if key in {
                "frames",
                "tasks",
                "metadata",
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            }:
                continue
            if isinstance(value, VideoFile):
                continue
            episode_row[key] = value

        video_stats: dict[str, dict[str, np.ndarray]] = {}
        for completed in video_info:
            if completed.feature is not None:
                self.features[completed.video_key] = completed.feature
            segment = completed.segment
            episode_row.update(
                {
                    f"videos/{segment.video_key}/chunk_index": segment.chunk_key,
                    f"videos/{segment.video_key}/file_index": segment.file_index,
                    f"videos/{segment.video_key}/from_timestamp": segment.from_timestamp,
                    f"videos/{segment.video_key}/to_timestamp": segment.to_timestamp,
                }
            )
            video_stats[segment.video_key] = segment.stats

        episode_stats = dict(frame_info.frame_stats)
        episode_stats.update(video_stats)
        episode_row.update(_flatten_stats_for_episode(episode_stats))
        return episode_row

    def _write_stage1_meta(self) -> None:
        if self._episode_rows:
            episodes_table = pa.Table.from_pylist(self._episode_rows)
            with self.folder.open(
                self._meta_path("episodes/file-000.parquet"), mode="wb"
            ) as out:
                pq.write_table(
                    episodes_table,
                    out,
                    compression="snappy",
                    use_dictionary=True,
                )

        tasks_table = pa.Table.from_pydict(
            {
                "task": list(self._task_order),
                "task_index": list(range(len(self._task_order))),
            }
        )
        with self.folder.open(self._meta_path("tasks.parquet"), mode="wb") as out:
            pq.write_table(
                tasks_table,
                out,
                compression="snappy",
                use_dictionary=True,
            )

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
            self._meta_path("info.jsonl"), mode="wt", encoding="utf-8"
        ) as out:
            out.write(json.dumps(info_record, sort_keys=True))
            out.write("\n")

    def _append_frame_table(
        self,
        table: pa.Table | None,
    ) -> tuple[int, int, int]:
        if (
            table is not None
            and self._data_writer is not None
            and self._data_bytes_written >= self._data_bytes_limit
        ):
            self._close_data()
            self._data_file_index += 1

        current_file_index = self._data_file_index
        dataset_from_index = self._global_frame_index
        dataset_to_index = dataset_from_index + (int(table.num_rows) if table else 0)

        if table is not None and table.num_rows > 0:
            self._ensure_data_writer(table.schema)
            if self._data_writer is None or self._data_writer_file is None:
                raise RuntimeError("data writer is not initialized")
            self._data_writer.write_table(table)
            self._data_bytes_written = int(self._data_writer_file.tell())

        self._global_frame_index = dataset_to_index
        return current_file_index, dataset_from_index, dataset_to_index

    def _require_required_fields(
        self,
        row: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        return self._require_required_fields_from_value(row["frames"])

    def _require_required_fields_from_value(
        self,
        frames_raw: Any,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(frames_raw, list):
            raise ValueError("LeRobot writer requires frames as a list on each row")
        if not all(isinstance(item, Mapping) for item in frames_raw):
            raise ValueError("LeRobot writer requires each frame to be a mapping")
        return [item for item in frames_raw if isinstance(item, Mapping)]

    def _source_episode_stats(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, dict[str, np.ndarray]]:
        return self._source_episode_stats_from_value(row.get("metadata"))

    def _source_episode_stats_from_value(
        self,
        metadata: Any,
    ) -> dict[str, dict[str, np.ndarray]]:
        if not isinstance(metadata, Mapping):
            return {}
        raw_stats = metadata.get(LEROBOT_EPISODE_STATS)
        if not isinstance(raw_stats, Mapping):
            return {}
        return _cast_stats_to_numpy(raw_stats)

    def _initialize_info_from_rows(
        self, rows: Sequence[Row | Mapping[str, Any]]
    ) -> None:
        if self.features:
            return

        first = rows[0]
        frames = self._require_required_fields(first)
        metadata = first.get("metadata")
        if isinstance(metadata, Mapping):
            info = metadata.get(LEROBOT_INFO)
            if isinstance(info, Mapping):
                fps_raw = info.get("fps")
                self._fps = int(fps_raw) if fps_raw is not None else None
                robot_type_raw = info.get("robot_type")
                self._robot_type = (
                    str(robot_type_raw) if robot_type_raw is not None else None
                )

        for row in rows:
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

        max_videos_in_row = max(
            (
                sum(1 for value in row.values() if isinstance(value, VideoFile))
                for row in rows
            ),
            default=0,
        )
        self._video_config = replace(
            self.config.video,
            encoder_threads=_resolve_video_threads(
                requested_threads=self.config.video.encoder_threads,
                videos_in_row=max_videos_in_row,
            ),
            decoder_threads=_resolve_video_threads(
                requested_threads=self.config.video.decoder_threads,
                videos_in_row=max_videos_in_row,
            ),
        )

    def _feature_spec(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, VideoFile):
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
        rel = _DEFAULT_DATA_PATH.format(
            chunk_index=self.chunk_key,
            file_index=self._data_file_index,
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
