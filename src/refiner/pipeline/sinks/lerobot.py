from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence, cast

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.asyncio.runtime import submit
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.video import VideoFile
from refiner.video.writer import VideoStreamWriter, VideoTranscodeConfig
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.lerobot import LeRobotMetaReduceSink
from refiner.robotics.lerobot_format import (
    LeRobotFeatureInfo,
    LeRobotFeatureStats,
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotVideoInfo,
    LeRobotVideoStatsAccumulator,
    compute_table_stats,
    default_feature_info_by_key,
    infer_feature_info,
)
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import register_gauge

_DEFAULT_DATA_FILE_SIZE_IN_MB = 100
_DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
_CURRENT_OUTPUT_VERSION = "v3.0"
_OUTPUT_DATA_PATH = "data/chunk-{chunk_index}/file-{file_index:03d}.parquet"
_OUTPUT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4"


@dataclass(slots=True)
class _LeRobotShardState:
    """Shard-local staged state for one LeRobot output chunk."""

    chunk_key: str
    metadata: LeRobotMetadata | None = field(default=None, init=False)
    features: dict[str, LeRobotFeatureInfo] = field(default_factory=dict, init=False)
    episode_rows: list[Row] = field(default_factory=list, init=False)
    has_videos: bool = field(default=False, init=False)
    frames: "_FramesState" = field(default_factory=lambda: _FramesState(), init=False)
    video_writers: dict[str, VideoStreamWriter] = field(
        default_factory=dict, init=False
    )


@dataclass(slots=True)
class _FramesState:
    """Running state for staged frame parquet output within one chunk."""

    total_rows: int = 0
    file_index: int = 0
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    writer_file: Any = None

    def close(self) -> None:
        """Close the current parquet writer and reset file-local state."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.writer_file is not None:
            self.writer_file.close()
            self.writer_file = None


class LeRobotWriterSink(BaseSink):
    """Skeleton sink for the new LeRobot writer implementation."""

    def __init__(
        self,
        output: DataFolderLike,
        *,
        data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        max_video_prepare_in_flight: int = 10,
        codec: str = "mpeg4",
        pix_fmt: str = "yuv420p",
        transencoding_threads: int | None = None,
        encoder_options: Mapping[str, str] | None = None,
        quantile_bins: int = 5000,
        force_recompute_video_stats: bool = False,
    ):
        check_required_dependencies("write_lerobot", ["av"], dist="robotics")
        self.output = DataFolder.resolve(output)
        self.data_files_size_in_mb = data_files_size_in_mb
        self.video_files_size_in_mb = video_files_size_in_mb
        self.max_video_prepare_in_flight = max_video_prepare_in_flight
        self.video_transcode_config = VideoTranscodeConfig(
            codec=codec,
            pix_fmt=pix_fmt,
            transencoding_threads=transencoding_threads,
            encoder_options=(
                dict(encoder_options) if encoder_options is not None else None
            ),
        )
        self.quantile_bins = quantile_bins
        self.force_recompute_video_stats = force_recompute_video_stats
        self._states: dict[str, _LeRobotShardState] = {}
        self._async_window = AsyncWindow[None](
            max_in_flight=max_video_prepare_in_flight,
            preserve_order=False,
        )
        self._episodes_in_flight_registered = False

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        """Submit one async write task per episode row in the shard-local block."""
        if not self._episodes_in_flight_registered:
            register_gauge(
                "episodes_in_flight",
                lambda: len(self._async_window),
                unit="episodes",
            )
            self._episodes_in_flight_registered = True
        state = self._state_for_shard(shard_id)
        for row in block:
            self._async_window.submit_blocking(self._write_row(state, row))

    def on_shard_complete(self, shard_id: str) -> None:
        """Flush pending async work and persist one shard-local output chunk."""
        # TODO: We don't have to flush the whole thing we just need to flush the shard rows
        self._async_window.flush()
        state = self._states.pop(shard_id, None)
        if state is not None:
            self._commit_shard(state)

    def close(self) -> None:
        """Flush any remaining shard-local work before sink shutdown."""
        self._async_window.flush()
        for state in self._states.values():
            self._commit_shard(state)
        self._states.clear()

    def _state_for_shard(self, shard_id: str) -> _LeRobotShardState:
        """Get or create the staged writer state for one output chunk."""
        state = self._states.get(shard_id)
        if state is None:
            state = _LeRobotShardState(
                chunk_key=f"{shard_id}__w{get_active_worker_token()}",
            )
            self._states[shard_id] = state
        return state

    def _video_writer(
        self,
        state: _LeRobotShardState,
        video_key: str,
    ) -> VideoStreamWriter:
        writer = state.video_writers.get(video_key)
        if writer is None:
            writer = VideoStreamWriter(
                folder=self.output,
                stream_key=video_key,
                transcode_config=self.video_transcode_config,
                video_bytes_limit=int(self.video_files_size_in_mb) * 1024 * 1024,
                output_rel_template=_OUTPUT_VIDEO_PATH.replace(
                    "{video_key}", "{stream_key}"
                ),
                output_context={"chunk_index": state.chunk_key},
            )
            state.video_writers[video_key] = writer
        return writer

    def _commit_shard(self, state: _LeRobotShardState) -> None:
        """Write staged shard-local metadata files for one completed chunk."""
        metadata = state.metadata
        if metadata is None:
            raise RuntimeError(
                "cannot commit LeRobot shard without initialized metadata"
            )

        state.frames.close()
        writers = tuple(state.video_writers.values())
        if writers:

            async def _close_all_video_writers() -> None:
                await asyncio.gather(*(writer.close_async() for writer in writers))

            submit(_close_all_video_writers()).result()
        state.video_writers.clear()

        episode_count = len(state.episode_rows)
        if state.episode_rows:
            episodes = Tabular.from_rows(state.episode_rows).table
            with self.output.open(
                f"meta/chunk-{state.chunk_key}/episodes/file-000.parquet",
                mode="wb",
            ) as out:
                pq.write_table(
                    episodes,
                    out,
                    compression="snappy",
                    use_dictionary=True,
                )
            state.episode_rows.clear()

        tasks_table = metadata.tasks.to_table()
        with self.output.open(
            f"meta/chunk-{state.chunk_key}/tasks.parquet",
            mode="wb",
        ) as out:
            pq.write_table(
                tasks_table,
                out,
                compression="snappy",
                use_dictionary=True,
            )

        info_record = LeRobotInfo(
            codebase_version=_CURRENT_OUTPUT_VERSION,
            fps=metadata.info.fps,
            robot_type=metadata.info.robot_type,
            total_episodes=episode_count,
            total_frames=state.frames.total_rows,
            total_tasks=len(metadata.tasks),
            data_files_size_in_mb=self.data_files_size_in_mb,
            video_files_size_in_mb=self.video_files_size_in_mb,
            data_path=_OUTPUT_DATA_PATH,
            video_path=_OUTPUT_VIDEO_PATH if state.has_videos else "",
            features=state.features,
            splits={"train": f"0:{int(episode_count)}"},
        )
        info_record = info_record.to_json_dict()
        with self.output.open(
            f"meta/chunk-{state.chunk_key}/info.json",
            mode="wt",
            encoding="utf-8",
        ) as out:
            out.write(json.dumps(info_record, sort_keys=True))

    async def _write_row(self, state: _LeRobotShardState, row: Row) -> None:
        """Write one LeRobot episode into staged shard-local outputs."""
        metadata = row.metadata if isinstance(row, LeRobotRow) else row.get("metadata")
        if not isinstance(metadata, LeRobotMetadata):
            raise ValueError("LeRobot writer requires LeRobot metadata on each row")

        if isinstance(row, LeRobotRow):
            frames = (
                row.frames
                if isinstance(row.frames, Tabular)
                else Tabular.from_rows(cast(Sequence[Row], row.frames))
            )
        else:
            frames = Tabular.from_rows(
                [
                    frame if isinstance(frame, Row) else DictRow(frame)
                    for frame in row["frames"]
                ]
            )
        if frames.num_rows <= 0:
            # empty frames
            return

        if isinstance(row, LeRobotRow):
            base_row = row._row
            video_inputs = [
                (video_ref.key, video_ref.video) for video_ref in row.videos.values()
            ]
            source_stats_by_key = {key: row.stats.get(key) for key, _ in video_inputs}
            frame_count = row.length
        else:
            base_row = DictRow(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"frames", "metadata"}
                    and not isinstance(value, VideoFile)
                },
                shard_id=row.shard_id,
            )
            video_inputs = [
                (key, value)
                for key, value in row.items()
                if isinstance(value, VideoFile)
            ]
            source_stats_by_key: dict[str, LeRobotFeatureStats | None] = {
                key: None for key, _ in video_inputs
            }
            frame_count = frames.num_rows

        if state.metadata is None:
            state.metadata = metadata
        elif (
            metadata.info.fps != state.metadata.info.fps
            or metadata.info.robot_type != state.metadata.info.robot_type
        ):
            raise ValueError(
                "LeRobot writer requires stable fps and robot_type within each shard"
            )
        elif metadata.tasks.index_to_task != state.metadata.tasks.index_to_task:
            raise ValueError(
                "LeRobot writer encountered mismatched task metadata across episodes"
            )

        if video_inputs and self.video_transcode_config.videos_in_row == 1:
            self.video_transcode_config = (
                self.video_transcode_config.with_videos_in_row(len(video_inputs))
            )

        # compute features
        features: dict[str, LeRobotFeatureInfo] = {}
        if not state.features:
            for key, value in base_row.items():
                spec = infer_feature_info(value)
                if spec is not None:
                    features.setdefault(key, spec)

            frame_sample = next(iter(frames), None)
            if frame_sample is not None:
                for key, value in frame_sample.items():
                    spec = infer_feature_info(
                        value,
                        fps=(
                            None
                            if key
                            in {
                                "timestamp",
                                "frame_index",
                                "episode_index",
                                "index",
                                "task_index",
                            }
                            else metadata.info.fps
                        ),
                    )
                    if spec is not None:
                        features.setdefault(key, spec)

            for key, spec in default_feature_info_by_key().items():
                features.setdefault(key, spec)

        episode_row_patch: dict[str, Any] = {
            "meta/episodes/chunk_index": state.chunk_key,
            "meta/episodes/file_index": 0,
        }

        # write frames. get frame information ready to patch episode
        episode_row_patch.update(self._write_frames(state, frames=frames))
        if "task_index" in frames.table.schema.names:
            task_indices = sorted(
                {
                    int(task_index)
                    for task_index in frames.table.column("task_index").to_pylist()
                    if task_index is not None
                }
            )
            episode_row_patch["tasks"] = [
                metadata.tasks.index_to_task[task_index] for task_index in task_indices
            ]

        # video work
        video_tasks = [
            asyncio.create_task(
                self._write_video(
                    state,
                    video_key=video_key,
                    video=video,
                    frame_count=frame_count,
                    source_stats=source_stats_by_key[video_key],
                )
            )
            for video_key, video in video_inputs
        ]
        video_features: dict[str, LeRobotFeatureInfo] = {}

        try:
            # get video results
            if video_tasks:
                state.has_videos = True
                for video_key, row_patch, feature in await asyncio.gather(*video_tasks):
                    episode_row_patch.update(row_patch)
                    if feature is not None:
                        video_features[video_key] = feature
        except Exception:
            for task in video_tasks:
                if not task.done():
                    task.cancel()
            if video_tasks:
                await asyncio.gather(*video_tasks, return_exceptions=True)
            raise

        written_row = base_row.update(episode_row_patch)

        if not state.features:
            state.features = features
            state.features.update(video_features)
        state.episode_rows.append(written_row)
        state.frames.total_rows += frames.num_rows
        if row.shard_id is not None:
            row.log_throughput("episodes_written", 1, unit="episodes")
            row.log_histogram("frames", frame_count, unit="frames", per="episode")
            if video_inputs:
                row.log_throughput("videos_written", len(video_inputs), unit="videos")

    async def _write_video(
        self,
        state: _LeRobotShardState,
        *,
        video_key: str,
        video: VideoFile,
        frame_count: int,
        source_stats: LeRobotFeatureStats | None,
    ) -> tuple[str, dict[str, Any], LeRobotFeatureInfo | None]:
        """Write one episode video and return the patch it produced."""
        accumulator = LeRobotVideoStatsAccumulator(
            frame_count=frame_count,
            quantile_bins=self.quantile_bins,
        )
        written = await self._video_writer(state, video_key).write_video(
            # source video
            video,
            # callback to not shove lerobot logic inside video writer
            frame_observer=accumulator.observe_rgb_frame,
            # if this episode has no video stats, or if we have been force to manually recompute all episodes, use transcode
            force_transcode=(
                self.force_recompute_video_stats
                or source_stats is None
                or source_stats == LeRobotFeatureStats()
            ),
        )
        segment = written.segment
        computed_stats = accumulator.stats()
        stats = (
            source_stats.to_json_dict()
            if (
                written.mode == "remux"
                and source_stats is not None
                and source_stats != LeRobotFeatureStats()
                and not self.force_recompute_video_stats
            )
            else computed_stats.to_json_dict()
            if computed_stats is not None
            else {}
        )
        row_patch: dict[str, Any] = {
            f"videos/{segment.stream_key}/chunk_index": state.chunk_key,
            f"videos/{segment.stream_key}/file_index": segment.file_index,
            f"videos/{segment.stream_key}/from_timestamp": segment.from_timestamp,
            f"videos/{segment.stream_key}/to_timestamp": segment.to_timestamp,
            **{
                f"stats/{segment.stream_key}/{name}": value
                for name, value in stats.items()
            },
        }
        feature = LeRobotFeatureInfo(
            dtype="video",
            shape=(segment.height, segment.width, 3),
            names=["height", "width", "channels"],
            video_info=LeRobotVideoInfo(
                codec=segment.codec,
                pix_fmt=segment.pix_fmt,
                is_depth_map=False,
                fps=segment.fps,
                has_audio=False,
            ),
        )
        return video_key, row_patch, feature

    def _write_frames(
        self,
        state: _LeRobotShardState,
        *,
        frames: Tabular,
    ) -> dict[str, Any]:
        """Write one episode frame table and return the patch it produced."""
        frames_state = state.frames
        table = frames.table
        row_count = int(table.num_rows)
        # update indexes
        index_column = pa.array(
            [frames_state.total_rows + i for i in range(row_count)],
            type=pa.int64(),
        )
        table = set_or_append_column(table, "index", index_column)

        if "frame_index" not in table.schema.names:
            table = set_or_append_column(
                table,
                "frame_index",
                pa.array(range(row_count), type=pa.int64()),
            )

        # file switch
        if (
            frames_state.writer is not None
            and frames_state.writer_file is not None
            and int(frames_state.writer_file.tell())
            >= int(self.data_files_size_in_mb) * 1024 * 1024
        ):
            frames_state.close()
            frames_state.file_index += 1

        current_file_index = frames_state.file_index
        dataset_from_index = frames_state.total_rows
        dataset_to_index = dataset_from_index + row_count

        if frames_state.schema is None:
            frames_state.schema = table.schema
        elif frames_state.schema != table.schema:
            raise ValueError(
                "LeRobot writer requires a stable frame schema across episodes"
            )

        if frames_state.writer is None:
            rel = _OUTPUT_DATA_PATH.format(
                chunk_index=state.chunk_key,
                file_index=frames_state.file_index,
            )
            frames_state.writer_file = self.output.open(rel, mode="wb")
            frames_state.writer = pq.ParquetWriter(
                frames_state.writer_file,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )

        frames_state.writer.write_table(table)

        return {
            "length": frames.num_rows,
            "data/chunk_index": state.chunk_key,
            "data/file_index": current_file_index,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            **compute_table_stats(table).flatten_fields(),
        }

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_lerobot",
            "writer",
            {
                "path": self.output.abs_path(),
                "data_files_size_in_mb": self.data_files_size_in_mb,
                "video_files_size_in_mb": self.video_files_size_in_mb,
                "max_video_prepare_in_flight": self.max_video_prepare_in_flight,
            },
        )

    def build_reducer(self) -> BaseSink:
        return LeRobotMetaReduceSink(output=self.output)


__all__ = [
    "LeRobotWriterSink",
]
