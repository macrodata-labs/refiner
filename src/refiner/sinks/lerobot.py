from __future__ import annotations

from fractions import Fraction
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import av
from fsspec.spec import AbstractFileSystem
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io import DataFolder
from refiner.media import DecodedVideo, Video
from refiner.processors.step import SinkStep
from refiner.sources.readers.lerobot import LEROBOT_INFO
from refiner.sources.row import Row

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_DATA_FILE_SIZE_IN_MB = 100
_DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_DEFAULT_STATS_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
_WORKER_CHUNK_STRIDE = 1_000_000


@dataclass(frozen=True, slots=True)
class LeRobotWriterConfig:
    root: str
    fs: AbstractFileSystem | None = None
    storage_options: Mapping[str, Any] | None = None
    overwrite: bool = False
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB
    video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.data_files_size_in_mb <= 0:
            raise ValueError("data_files_size_in_mb must be > 0")
        if self.video_files_size_in_mb <= 0:
            raise ValueError("video_files_size_in_mb must be > 0")


@dataclass(slots=True)
class LeRobotWriterSinkStep(SinkStep):
    """Stage 1 sink: write chunked data/videos/meta/chunk-* artifacts."""

    config: LeRobotWriterConfig
    index: int
    op_name: str | None = "write_lerobot"
    passthrough: bool = False
    _runtime: _LeRobotWriterRuntime | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _is_finalized: bool = field(default=False, init=False, repr=False, compare=False)

    def launch_prepare(self) -> None:
        _prepare_output_root(self.config)

    def start_run(self) -> None:
        self._runtime = _LeRobotWriterRuntime(self.config)
        self._is_finalized = False

    def consume_row(self, row: Row) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("LeRobot writer sink is not initialized")
        runtime.consume_row(row)

    def finalize(self) -> None:
        if self._is_finalized:
            return
        self._is_finalized = True
        runtime = self._runtime
        if runtime is None:
            return
        runtime.finalize()


@dataclass(slots=True)
class LeRobotReduceSinkStep(SinkStep):
    """Stage 2 sink: reduce chunked metadata into final /meta and cleanup chunks."""

    config: LeRobotWriterConfig
    index: int
    op_name: str | None = "reduce_lerobot_meta"
    passthrough: bool = False
    _reduced: bool = field(default=False, init=False, repr=False, compare=False)

    def start_run(self) -> None:
        self._reduced = False

    def consume_row(self, row: Row) -> None:
        if self._reduced:
            return
        task_rank = _as_int(row.get("task_rank"))
        if task_rank is not None and task_rank != 0:
            return
        _LeRobotMetaReducer(self.config).reduce()
        self._reduced = True

    def finalize(self) -> None:
        if self._reduced:
            return
        _LeRobotMetaReducer(self.config).reduce()
        self._reduced = True


@dataclass(slots=True)
class _VideoWriterState:
    chunk_idx: int
    file_idx: int
    temp_path: str
    container: Any
    stream: Any | None
    fps: int
    frames_written: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0


@dataclass(slots=True)
class _LeRobotWriterRuntime:
    config: LeRobotWriterConfig
    folder: DataFolder = field(init=False)
    _worker_chunk_base: int = field(init=False)

    _episode_rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _stats_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _tasks: set[str] = field(default_factory=set, init=False)

    _fps: int | None = field(default=None, init=False)
    _robot_type: str | None = field(default=None, init=False)
    _features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    _total_episodes: int = field(default=0, init=False)
    _total_frames: int = field(default=0, init=False)

    _global_frame_index: int = field(default=0, init=False)

    _data_chunk: int = field(default=0, init=False)
    _data_file_index: int = field(default=0, init=False)
    _data_estimated_bytes: int = field(default=0, init=False)
    _data_writer: pq.ParquetWriter | None = field(default=None, init=False)
    _data_writer_file: Any | None = field(default=None, init=False)
    _data_schema: pa.Schema | None = field(default=None, init=False)

    _video_next_chunk: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    _video_next_file_index: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    _video_writer_states: dict[str, _VideoWriterState] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(
            self.config.root,
            fs=self.config.fs,
            storage_options=self.config.storage_options,
        )
        self._worker_chunk_base = _worker_chunk_base()
        self._data_chunk = self._worker_chunk_base
        self._video_next_chunk = defaultdict(lambda: self._worker_chunk_base)

    @property
    def _data_bytes_limit(self) -> int:
        return int(self.config.data_files_size_in_mb) * 1024 * 1024

    @property
    def _video_bytes_limit(self) -> int:
        return int(self.config.video_files_size_in_mb) * 1024 * 1024

    def consume_row(self, row: Row) -> None:
        episode_index = _as_int(row.get("episode_index"))
        if episode_index is None:
            raise ValueError("LeRobot writer requires episode_index on each row")

        frames_raw = row.get("frames")
        if not isinstance(frames_raw, list):
            raise ValueError("LeRobot writer requires frames as a list on each row")

        frames = [_row_to_dict(item) for item in frames_raw]
        tasks = _extract_tasks(row)
        self._tasks.update(tasks)

        self._initialize_info_from_row(row)

        data_meta = self._write_episode_frames(
            episode_index=episode_index,
            tasks=tasks,
            frames=frames,
        )
        video_meta = self._write_episode_videos(row)
        episode_stats = _compute_episode_stats(row=row, frames=frames)

        episode_row: dict[str, Any] = {
            "episode_index": episode_index,
            "length": len(frames),
            "tasks": tasks,
            "data/chunk_index": data_meta["chunk_index"],
            "data/file_index": data_meta["file_index"],
            "dataset_from_index": data_meta["dataset_from_index"],
            "dataset_to_index": data_meta["dataset_to_index"],
            "meta/episodes/chunk_index": self._worker_chunk_base,
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

        self._episode_rows.append(episode_row)
        self._stats_records.append(
            {
                "episode_index": episode_index,
                "stats": _serialize_stats(episode_stats),
            }
        )
        self._total_episodes += 1
        self._total_frames += len(frames)

    def finalize(self) -> None:
        self._flush_video_writers()
        self._close_data_writer()

        chunk_dir = f"meta/chunk-{self._worker_chunk_base:06d}"
        episodes_rel = f"{chunk_dir}/episodes/file-000.parquet"
        tasks_rel = f"{chunk_dir}/tasks.jsonl"
        stats_rel = f"{chunk_dir}/stats.jsonl"
        info_rel = f"{chunk_dir}/info.jsonl"

        if self._episode_rows:
            episodes_table = pa.Table.from_pylist(self._episode_rows)
            with self.folder.open(episodes_rel, mode="wb") as out:
                pq.write_table(episodes_table, out)

        task_lines = [
            json.dumps({"task": t}, sort_keys=True) for t in sorted(self._tasks)
        ]
        if task_lines:
            with self.folder.open(tasks_rel, mode="wt", encoding="utf-8") as out:
                out.write("\n".join(task_lines))
                out.write("\n")

        if self._stats_records:
            with self.folder.open(stats_rel, mode="wt", encoding="utf-8") as out:
                for record in self._stats_records:
                    out.write(json.dumps(record, sort_keys=True))
                    out.write("\n")

        info_record = {
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
            "total_frames": self._total_frames,
        }
        with self.folder.open(info_rel, mode="wt", encoding="utf-8") as out:
            out.write(json.dumps(info_record, sort_keys=True))
            out.write("\n")

    def _initialize_info_from_row(self, row: Mapping[str, Any]) -> None:
        if self._features:
            return

        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            info = metadata.get(LEROBOT_INFO)
            if isinstance(info, Mapping):
                self._fps = _as_int(info.get("fps"))
                robot_type_raw = info.get("robot_type")
                self._robot_type = (
                    str(robot_type_raw) if robot_type_raw is not None else None
                )

        for key, value in row.items():
            if key in {"frames", "metadata", "task", "tasks"}:
                continue
            if isinstance(value, Video):
                self._features[key] = {
                    "dtype": "video",
                    "shape": [3, 1, 1],
                    "names": None,
                }
            elif _is_number(value):
                self._features[key] = {"dtype": "float64", "shape": [1], "names": None}

        frames_raw = row.get("frames")
        if isinstance(frames_raw, list) and frames_raw:
            sample = _row_to_dict(frames_raw[0])
            for key, value in sample.items():
                if key in {"index", "episode_index", "task_index"}:
                    self._features.setdefault(
                        key,
                        {"dtype": "int64", "shape": [1], "names": None},
                    )
                    continue
                dtype = _dtype_for_value(value)
                if dtype is None:
                    continue
                self._features.setdefault(
                    key,
                    {"dtype": dtype, "shape": _shape_for_value(value), "names": None},
                )

        self._features.setdefault(
            "timestamp", {"dtype": "float32", "shape": [1], "names": None}
        )
        self._features.setdefault(
            "frame_index", {"dtype": "int64", "shape": [1], "names": None}
        )
        self._features.setdefault(
            "episode_index", {"dtype": "int64", "shape": [1], "names": None}
        )
        self._features.setdefault(
            "index", {"dtype": "int64", "shape": [1], "names": None}
        )
        self._features.setdefault(
            "task_index", {"dtype": "int64", "shape": [1], "names": None}
        )

    def _write_episode_frames(
        self,
        *,
        episode_index: int,
        tasks: list[str],
        frames: list[dict[str, Any]],
    ) -> dict[str, int]:
        start_index = self._global_frame_index
        current_chunk = self._data_chunk
        current_file = self._data_file_index

        if not frames:
            return {
                "chunk_index": current_chunk,
                "file_index": current_file,
                "dataset_from_index": start_index,
                "dataset_to_index": start_index,
            }

        frame_rows: list[dict[str, Any]] = []
        task_to_index = {task: idx for idx, task in enumerate(sorted(self._tasks))}
        default_task_idx = task_to_index[tasks[0]] if tasks else None

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
        estimated_bytes = max(1, int(table.nbytes))
        if (
            self._data_writer is not None
            and self._data_estimated_bytes + estimated_bytes >= self._data_bytes_limit
        ):
            self._close_data_writer()
            self._data_chunk, self._data_file_index = _update_chunk_file_indices(
                self._data_chunk,
                self._data_file_index,
                self.config.chunk_size,
            )
            self._data_estimated_bytes = 0
            current_chunk = self._data_chunk
            current_file = self._data_file_index

        self._ensure_data_writer(table.schema)
        if self._data_writer is None:
            raise RuntimeError("data writer is not initialized")
        self._data_writer.write_table(table)
        self._data_estimated_bytes += estimated_bytes

        self._global_frame_index += len(frame_rows)
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
        rel = _DEFAULT_DATA_PATH.format(
            chunk_index=self._data_chunk,
            file_index=self._data_file_index,
        )
        self._data_writer_file = self.folder.open(rel, mode="wb")
        self._data_writer = pq.ParquetWriter(self._data_writer_file, schema=schema)

    def _close_data_writer(self) -> None:
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        if self._data_writer_file is not None:
            self._data_writer_file.close()
            self._data_writer_file = None

    def _write_episode_videos(self, row: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if not isinstance(value, Video):
                continue

            clip_from = (
                float(value.from_timestamp_s)
                if value.from_timestamp_s is not None
                else 0.0
            )
            clip_to = value.to_timestamp_s
            if clip_to is None:
                clip_to = _probe_video_duration_s(value)

            state = self._video_writer_states.get(key)
            fps = _resolve_video_fps(video=value, default_fps=self._fps)
            if state is None:
                state = self._open_video_writer(
                    video_key=key,
                    chunk_idx=self._video_next_chunk[key],
                    file_idx=self._video_next_file_index[key],
                    fps=fps,
                )

            estimated_clip_size = self._estimate_video_segment_size_bytes(
                video=value,
                clip_from=clip_from,
                clip_to=clip_to,
            )
            if (
                state.size_bytes > 0
                and estimated_clip_size > 0
                and state.size_bytes + estimated_clip_size >= self._video_bytes_limit
            ):
                self._flush_video_writer(key)
                state = self._open_video_writer(
                    video_key=key,
                    chunk_idx=self._video_next_chunk[key],
                    file_idx=self._video_next_file_index[key],
                    fps=fps,
                )

            from_ts = float(state.duration_s)
            clip_duration_s = self._append_video_segment(
                state=state,
                video=value,
                clip_from=clip_from,
                clip_to=clip_to,
            )
            to_ts = float(from_ts + clip_duration_s)
            state.duration_s = to_ts
            try:
                state.size_bytes = max(
                    state.size_bytes, int(os.path.getsize(state.temp_path))
                )
            except OSError:
                pass
            self._video_writer_states[key] = state

            out[f"videos/{key}/chunk_index"] = state.chunk_idx
            out[f"videos/{key}/file_index"] = state.file_idx
            out[f"videos/{key}/from_timestamp"] = from_ts
            out[f"videos/{key}/to_timestamp"] = to_ts

        return out

    def _open_video_writer(
        self,
        *,
        video_key: str,
        chunk_idx: int,
        file_idx: int,
        fps: int,
    ) -> _VideoWriterState:
        if fps <= 0:
            raise ValueError(f"Invalid video FPS for key {video_key!r}: {fps}")

        fd, temp_path = tempfile.mkstemp(
            prefix=f"refiner_lerobot_{video_key.replace('/', '_')}_",
            suffix=".mp4",
        )
        os.close(fd)
        container = av.open(temp_path, mode="w", options={"movflags": "faststart"})
        state = _VideoWriterState(
            chunk_idx=chunk_idx,
            file_idx=file_idx,
            temp_path=temp_path,
            container=container,
            stream=None,
            fps=fps,
        )
        self._video_writer_states[video_key] = state
        return state

    def _flush_video_writers(self) -> None:
        for key in list(self._video_writer_states.keys()):
            self._flush_video_writer(key)

    def _flush_video_writer(self, video_key: str) -> None:
        state = self._video_writer_states.pop(video_key, None)
        if state is None:
            return

        try:
            if state.stream is not None:
                for packet in state.stream.encode(None):
                    state.container.mux(packet)
            state.container.close()
            rel = _DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=state.chunk_idx,
                file_index=state.file_idx,
            )
            with (
                open(state.temp_path, "rb") as src,
                self.folder.open(rel, mode="wb") as dst,
            ):
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        finally:
            try:
                os.remove(state.temp_path)
            except FileNotFoundError:
                pass

        (
            self._video_next_chunk[video_key],
            self._video_next_file_index[video_key],
        ) = _update_chunk_file_indices(
            state.chunk_idx,
            state.file_idx,
            self.config.chunk_size,
        )

    def _append_video_segment(
        self,
        *,
        state: _VideoWriterState,
        video: Video,
        clip_from: float,
        clip_to: float | None,
    ) -> float:
        decoded = _decoded_video_payload(video)
        if decoded is not None:
            return self._append_video_segment_from_frames(
                state=state,
                video=decoded,
            )
        selected_frames = 0
        epsilon = 1e-6

        with video.media.cached_path(suffix=".mp4") as local_path:
            with av.open(local_path) as input_container:
                input_stream = input_container.streams.video[0]
                for frame in input_container.decode(input_stream):
                    ts = None
                    if frame.pts is not None and frame.time_base is not None:
                        ts = float(frame.pts * frame.time_base)

                    if ts is None:
                        continue
                    if ts + epsilon < clip_from:
                        continue
                    if clip_to is not None and ts - epsilon >= clip_to:
                        break

                    if state.stream is None:
                        state.stream = state.container.add_stream(
                            "mpeg4", rate=state.fps
                        )
                        state.stream.width = frame.width
                        state.stream.height = frame.height
                        state.stream.pix_fmt = "yuv420p"

                    out_frame = frame.reformat(
                        width=state.stream.width,
                        height=state.stream.height,
                        format=state.stream.pix_fmt,
                    )
                    out_frame.pts = state.frames_written
                    out_frame.time_base = Fraction(1, state.fps)
                    for packet in state.stream.encode(out_frame):
                        state.container.mux(packet)
                    state.frames_written += 1
                    selected_frames += 1

        if selected_frames <= 0:
            raise ValueError(
                f"Video segment for {video.uri!r} contains no decodable frames in "
                f"[{clip_from:.6f}, {clip_to if clip_to is not None else 'end'})."
            )
        return selected_frames / float(state.fps)

    def _estimate_video_segment_size_bytes(
        self,
        *,
        video: Video,
        clip_from: float,
        clip_to: float | None,
    ) -> int:
        decoded = _decoded_video_payload(video)
        if decoded is not None:
            if decoded.frame_count <= 0:
                return 0
            if decoded.width is not None and decoded.height is not None:
                return max(
                    1,
                    int(decoded.width * decoded.height * 3 * decoded.frame_count),
                )
            return max(1, decoded.frame_count * 1024)

        with video.media.cached_path(suffix=".mp4") as local_path:
            full_size = int(os.path.getsize(local_path))
            if full_size <= 0:
                return 0

            full_duration = _probe_video_duration_local_path(local_path)
            if full_duration is None or full_duration <= 0:
                return full_size
            if clip_to is None:
                clip_duration = max(0.0, full_duration - max(0.0, clip_from))
            else:
                clip_duration = max(0.0, clip_to - clip_from)
            if clip_duration <= 0:
                return max(1, min(full_size, 1024))

            ratio = min(1.0, clip_duration / full_duration)
            return max(1, int(full_size * ratio))

    def _append_video_segment_from_frames(
        self,
        *,
        state: _VideoWriterState,
        video: DecodedVideo,
    ) -> float:
        if video.frame_count <= 0:
            raise ValueError(
                f"Decoded video segment for {video.uri!r} contains no decodable frames."
            )
        frames_written = 0
        width = video.width
        height = video.height
        for frame_data in video.frames:
            if state.stream is None:
                if width is None or height is None:
                    if isinstance(frame_data, np.ndarray) and frame_data.ndim >= 2:
                        height = int(frame_data.shape[0])
                        width = int(frame_data.shape[1])
                if width is None or height is None:
                    raise RuntimeError(
                        "Decoded video frame shape missing width/height metadata."
                    )
                state.stream = state.container.add_stream("mpeg4", rate=state.fps)
                state.stream.width = width
                state.stream.height = height
                state.stream.pix_fmt = "yuv420p"

            out_frame = av.VideoFrame.from_ndarray(
                frame_data,
                format=video.pix_fmt or "rgb24",
            )
            out_frame.pts = state.frames_written
            out_frame.time_base = Fraction(1, state.fps)
            if state.stream is None:
                raise RuntimeError("Video stream not initialized")
            out_frame = out_frame.reformat(
                width=state.stream.width,
                height=state.stream.height,
                format=state.stream.pix_fmt,
            )
            for packet in state.stream.encode(out_frame):
                state.container.mux(packet)
            state.frames_written += 1
            frames_written += 1

        return float(frames_written) / float(state.fps)


def _decoded_video_payload(video: Video) -> DecodedVideo | None:
    media = video.media
    if isinstance(media, DecodedVideo):
        return media
    return None


@dataclass(slots=True)
class _LeRobotMetaReducer:
    config: LeRobotWriterConfig
    folder: DataFolder = field(init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(
            self.config.root,
            fs=self.config.fs,
            storage_options=self.config.storage_options,
        )

    def reduce(self) -> None:
        episodes_rows = self._load_stage1_episode_rows()
        if not episodes_rows:
            return

        tasks = self._load_stage1_tasks()
        stats_list = self._load_stage1_stats()
        infos = self._load_stage1_infos()

        for row in episodes_rows:
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0

        episodes_table = pa.Table.from_pylist(episodes_rows)
        with self.folder.open(
            "meta/episodes/chunk-000/file-000.parquet", mode="wb"
        ) as out:
            pq.write_table(episodes_table, out)

        tasks_table = pa.table(
            {
                "task_index": list(range(len(tasks))),
                "task": tasks,
            }
        )
        with self.folder.open("meta/tasks.parquet", mode="wb") as out:
            pq.write_table(tasks_table, out)

        merged_stats = _aggregate_stats(stats_list)
        with self.folder.open("meta/stats.json", mode="wt", encoding="utf-8") as out:
            json.dump(_serialize_stats(merged_stats), out, indent=2, sort_keys=True)

        info = self._merge_infos(infos=infos, tasks=tasks, episodes_rows=episodes_rows)
        with self.folder.open("meta/info.json", mode="wt", encoding="utf-8") as out:
            json.dump(info, out, indent=2, sort_keys=True)

        self._cleanup_stage1_chunks()

    def _load_stage1_episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rel in self._iter_stage1_episode_files():
            with self.folder.open(rel, mode="rb") as src:
                table = pq.read_table(src)
            rows.extend(table.to_pylist())
        rows.sort(key=lambda row: _as_int(row.get("episode_index")) or 0)
        return rows

    def _load_stage1_tasks(self) -> list[str]:
        tasks: set[str] = set()
        for rel in self._iter_stage1_jsonl_files(filename="tasks.jsonl"):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                for line in src:
                    payload = line.strip()
                    if not payload:
                        continue
                    item = json.loads(payload)
                    task = item.get("task")
                    if isinstance(task, str) and task:
                        tasks.add(task)
        return sorted(tasks)

    def _load_stage1_stats(self) -> list[dict[str, dict[str, np.ndarray]]]:
        out: list[dict[str, dict[str, np.ndarray]]] = []
        for rel in self._iter_stage1_jsonl_files(filename="stats.jsonl"):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                for line in src:
                    payload = line.strip()
                    if not payload:
                        continue
                    item = json.loads(payload)
                    raw = item.get("stats")
                    if isinstance(raw, Mapping):
                        out.append(_cast_stats_to_numpy(raw))
        return out

    def _load_stage1_infos(self) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for rel in self._iter_stage1_jsonl_files(filename="info.jsonl"):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                for line in src:
                    payload = line.strip()
                    if not payload:
                        continue
                    item = json.loads(payload)
                    if isinstance(item, dict):
                        infos.append(item)
        return infos

    def _merge_infos(
        self,
        *,
        infos: list[dict[str, Any]],
        tasks: list[str],
        episodes_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        first = infos[0] if infos else {}
        total_episodes = len(episodes_rows)
        total_frames = sum(
            max(
                0,
                int(_as_int(row.get("dataset_to_index")) or 0)
                - int(_as_int(row.get("dataset_from_index")) or 0),
            )
            for row in episodes_rows
        )

        return {
            "chunks_size": int(first.get("chunks_size") or self.config.chunk_size),
            "data_files_size_in_mb": int(
                first.get("data_files_size_in_mb") or self.config.data_files_size_in_mb
            ),
            "video_files_size_in_mb": int(
                first.get("video_files_size_in_mb")
                or self.config.video_files_size_in_mb
            ),
            "data_path": str(first.get("data_path") or _DEFAULT_DATA_PATH),
            "video_path": first.get("video_path") or _DEFAULT_VIDEO_PATH,
            "fps": _as_int(first.get("fps")),
            "robot_type": first.get("robot_type"),
            "features": first.get("features")
            if isinstance(first.get("features"), dict)
            else {},
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "splits": {"train": f"0:{total_episodes}"},
        }

    def _cleanup_stage1_chunks(self) -> None:
        root_meta = self.folder.file("meta").path
        for abs_path in self.folder.fs.glob(f"{root_meta}/chunk-*"):
            self.folder.fs.rm(abs_path, recursive=True)

    def _iter_stage1_episode_files(self) -> list[str]:
        return self._iter_stage1_files(
            predicate=lambda abs_path: "/episodes/file-" in abs_path
            and abs_path.endswith(".parquet"),
        )

    def _iter_stage1_jsonl_files(self, *, filename: str) -> list[str]:
        return self._iter_stage1_files(
            predicate=lambda abs_path: abs_path.endswith(f"/{filename}"),
        )

    def _iter_stage1_files(self, *, predicate: Callable[[str], bool]) -> list[str]:
        root_meta = self.folder.file("meta").path
        if not self.folder.fs.exists(root_meta):
            return []

        matches: list[str] = []
        for abs_path in self.folder.fs.find(root_meta):
            if "/chunk-" not in abs_path:
                continue
            if not bool(predicate(abs_path)):
                continue
            matches.append(_to_rel(self.folder, abs_path))
        matches.sort()
        return matches


def _prepare_output_root(config: LeRobotWriterConfig) -> None:
    folder = DataFolder.resolve(
        config.root,
        fs=config.fs,
        storage_options=config.storage_options,
    )
    paths = [
        folder.file("data").path,
        folder.file("videos").path,
        folder.file("meta").path,
    ]
    already_exists = any(folder.fs.exists(path) for path in paths)
    if already_exists and not config.overwrite:
        raise FileExistsError(
            f"Output root already contains LeRobot data under {config.root!r}; set overwrite=True to replace it."
        )
    if already_exists and config.overwrite:
        for path in paths:
            if folder.fs.exists(path):
                folder.fs.rm(path, recursive=True)


def _update_chunk_file_indices(
    chunk_idx: int,
    file_idx: int,
    chunk_size: int,
) -> tuple[int, int]:
    if file_idx >= chunk_size - 1:
        return (chunk_idx + 1, 0)
    return (chunk_idx, file_idx + 1)


def _worker_chunk_base() -> int:
    rank_raw = os.environ.get("REFINER_WORKER_RANK")
    rank = _as_int(rank_raw)
    if rank is None or rank < 0:
        return 0
    return rank * _WORKER_CHUNK_STRIDE


def _extract_tasks(row: Mapping[str, Any]) -> list[str]:
    tasks_raw = row.get("tasks")
    out: list[str] = []
    if isinstance(tasks_raw, list):
        for value in tasks_raw:
            if isinstance(value, str) and value:
                out.append(value)
    task_raw = row.get("task")
    if isinstance(task_raw, str) and task_raw and task_raw not in out:
        out.append(task_raw)
    return out


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Row):
        return row.to_dict()
    if isinstance(row, Mapping):
        return dict(row)
    raise ValueError(f"Unsupported frame row type: {type(row)!r}")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.number))


def _as_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, bool) or value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype.kind not in {"i", "u", "f"}:
        return None
    if arr.size == 0:
        return None
    if not np.isfinite(arr.astype(np.float64, copy=False)).all():
        return None
    return arr.astype(np.float64, copy=False)


def _compute_episode_stats(
    *,
    row: Mapping[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    stats = _compute_frame_stats(frames)
    for key, value in row.items():
        if not isinstance(value, Video):
            continue
        video_stats = _compute_video_stats(value)
        if video_stats is not None:
            stats[key] = video_stats
    return stats


def _compute_frame_stats(
    frames: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    if not frames:
        return {}

    excluded = {"index", "episode_index", "task_index"}
    by_feature: dict[str, list[np.ndarray]] = defaultdict(list)

    for frame in frames:
        for key, value in frame.items():
            if key in excluded:
                continue
            arr = _as_numeric_array(value)
            if arr is None:
                continue
            by_feature[key].append(arr)

    out: dict[str, dict[str, np.ndarray]] = {}
    for key, values in by_feature.items():
        if not values:
            continue
        first_shape = values[0].shape
        if any(v.shape != first_shape for v in values):
            continue
        stacked = np.stack(values, axis=0)
        keepdims = stacked.ndim == 1
        out[key] = _feature_stats(stacked, keepdims=keepdims)
    return out


def _compute_video_stats(video: Video) -> dict[str, np.ndarray] | None:
    pixels = _sample_video_pixels(video)
    if pixels is None or len(pixels) < 2:
        return None

    ft_stats = _feature_stats(pixels, keepdims=False)
    normalized: dict[str, np.ndarray] = {}
    for key, value in ft_stats.items():
        if key == "count":
            normalized[key] = value
            continue
        normalized[key] = np.asarray(value, dtype=np.float64).reshape(3, 1, 1) / 255.0
    return normalized


def _sample_video_pixels(video: Video) -> np.ndarray | None:
    decoded = _decoded_video_payload(video)
    if decoded is not None:
        if not decoded.frames:
            return None
        samples = []
        for frame in decoded.frames:
            if not isinstance(frame, np.ndarray):
                continue
            if frame.ndim != 3:
                continue
            chw = np.transpose(frame, (2, 0, 1))
            down = _auto_downsample_height_width(chw)
            pixels = np.transpose(down, (1, 2, 0)).reshape(-1, down.shape[0])
            samples.append(pixels.astype(np.float64, copy=False))
        if not samples:
            return None
        return np.concatenate(samples, axis=0)

    start_ts = float(video.from_timestamp_s or 0.0)
    end_ts = float(video.to_timestamp_s) if video.to_timestamp_s is not None else None

    samples: list[np.ndarray] = []
    with video.media.cached_path(suffix=".mp4") as local_path:
        with av.open(local_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                ts = None
                if frame.pts is not None and frame.time_base is not None:
                    ts = float(frame.pts * frame.time_base)
                if ts is None:
                    continue
                if ts + 1e-6 < start_ts:
                    continue
                if end_ts is not None and ts - 1e-6 >= end_ts:
                    break

                arr = frame.to_ndarray(format="rgb24")
                chw = np.transpose(arr, (2, 0, 1))
                down = _auto_downsample_height_width(chw)
                pixels = np.transpose(down, (1, 2, 0)).reshape(-1, down.shape[0])
                samples.append(pixels.astype(np.float64, copy=False))

    if not samples:
        return None
    return np.concatenate(samples, axis=0)


def _auto_downsample_height_width(
    image_chw: np.ndarray,
    *,
    target_size: int = 150,
    max_size_threshold: int = 300,
) -> np.ndarray:
    _, height, width = image_chw.shape
    if max(width, height) < max_size_threshold:
        return image_chw
    factor = max(1, int((width if width > height else height) / target_size))
    return image_chw[:, ::factor, ::factor]


def _feature_stats(array: np.ndarray, *, keepdims: bool) -> dict[str, np.ndarray]:
    if array.ndim == 0:
        array = array.reshape(1)

    count = int(array.shape[0])
    axis = 0

    min_v = np.min(array, axis=axis)
    max_v = np.max(array, axis=axis)
    mean_v = np.mean(array, axis=axis)
    std_v = np.std(array, axis=axis)

    if keepdims:
        min_v = np.atleast_1d(min_v)
        max_v = np.atleast_1d(max_v)
        mean_v = np.atleast_1d(mean_v)
        std_v = np.atleast_1d(std_v)

    out: dict[str, np.ndarray] = {
        "min": np.asarray(min_v),
        "max": np.asarray(max_v),
        "mean": np.asarray(mean_v),
        "std": np.asarray(std_v),
        "count": np.array([count], dtype=np.int64),
    }

    for q in _DEFAULT_STATS_QUANTILES:
        key = f"q{int(q * 100):02d}"
        q_value = np.quantile(array, q, axis=axis)
        if keepdims:
            q_value = np.atleast_1d(q_value)
        out[key] = np.asarray(q_value)

    return out


def _flatten_stats_for_episode(
    stats: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature, feature_stats in stats.items():
        for stat_name, stat_value in feature_stats.items():
            out[f"stats/{feature}/{stat_name}"] = _jsonable_value(stat_value)
    return out


def _aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    if not stats_list:
        return {}

    feature_keys = {k for stats in stats_list for k in stats.keys()}
    out: dict[str, dict[str, np.ndarray]] = {}

    for feature in sorted(feature_keys):
        items = [stats[feature] for stats in stats_list if feature in stats]
        means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in items])
        stds = np.stack([np.asarray(item["std"], dtype=np.float64) for item in items])
        counts = np.stack(
            [np.asarray(item["count"], dtype=np.float64) for item in items]
        )
        mins = np.stack([np.asarray(item["min"], dtype=np.float64) for item in items])
        maxs = np.stack([np.asarray(item["max"], dtype=np.float64) for item in items])

        total_count = counts.sum(axis=0)
        counts_b = counts
        while counts_b.ndim < means.ndim:
            counts_b = np.expand_dims(counts_b, axis=-1)

        total_mean = (means * counts_b).sum(axis=0) / np.maximum(total_count, 1e-12)
        variances = stds**2
        delta = means - total_mean
        total_var = ((variances + delta**2) * counts_b).sum(axis=0) / np.maximum(
            total_count,
            1e-12,
        )

        agg: dict[str, np.ndarray] = {
            "min": np.min(mins, axis=0),
            "max": np.max(maxs, axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_var),
            "count": total_count,
        }

        quantile_keys = [
            k for k in items[0].keys() if k.startswith("q") and k[1:].isdigit()
        ]
        for q_key in quantile_keys:
            if not all(q_key in item for item in items):
                continue
            q_vals = np.stack(
                [np.asarray(item[q_key], dtype=np.float64) for item in items]
            )
            agg[q_key] = (q_vals * counts_b).sum(axis=0) / np.maximum(
                total_count, 1e-12
            )

        out[feature] = agg

    return out


def _serialize_stats(
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for feature, feature_stats in stats.items():
        out_feature: dict[str, Any] = {}
        for key, value in feature_stats.items():
            out_feature[key] = _jsonable_value(value)
        out[feature] = out_feature
    return out


def _cast_stats_to_numpy(raw: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for feature, feature_stats in raw.items():
        if not isinstance(feature_stats, Mapping):
            continue
        inner: dict[str, np.ndarray] = {}
        for key, value in feature_stats.items():
            inner[str(key)] = np.asarray(value)
        out[str(feature)] = inner
    return out


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not float(value).is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _probe_video_duration_s(video: Video) -> float | None:
    decoded = _decoded_video_payload(video)
    if decoded is not None:
        return decoded.duration_s

    with video.media.cached_path(suffix=".mp4") as local_path:
        return _probe_video_duration_local_path(local_path)


def _probe_video_duration_local_path(local_path: str) -> float | None:
    with av.open(local_path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    return None


def _resolve_video_fps(video: Video, default_fps: int | None) -> int:
    if video.fps is not None and int(video.fps) > 0:
        return int(video.fps)
    if default_fps is not None and int(default_fps) > 0:
        return int(default_fps)

    if _decoded_video_payload(video) is not None:
        return 30

    with video.media.cached_path(suffix=".mp4") as local_path:
        with av.open(local_path) as container:
            stream = container.streams.video[0]
            if stream.average_rate is not None:
                try:
                    rate = float(stream.average_rate)
                    fps = int(round(rate))
                    if fps > 0:
                        return fps
                except Exception:
                    pass
            if stream.base_rate is not None:
                try:
                    rate = float(stream.base_rate)
                    fps = int(round(rate))
                    if fps > 0:
                        return fps
                except Exception:
                    pass
    return 30


def _dtype_for_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    arr = np.asarray(value)
    if arr.dtype.kind in {"i", "u"}:
        return "int64"
    if arr.dtype.kind == "f":
        return "float64"
    return None


def _shape_for_value(value: Any) -> list[int]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [1]
    return [int(x) for x in arr.shape]


def _to_rel(folder: DataFolder, abs_path: str) -> str:
    root = folder.path.rstrip("/")
    prefix = f"{root}/"
    if abs_path.startswith(prefix):
        return abs_path[len(prefix) :]
    return abs_path


__all__ = [
    "LeRobotWriterConfig",
    "LeRobotWriterSinkStep",
    "LeRobotReduceSinkStep",
]
