from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.asyncio.window import AsyncWindow
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.robotics.lerobot_format import (
    compute_table_stats,
    LeRobotFeatureInfo,
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotVideoRef,
    default_feature_info_by_key,
    infer_feature_info,
)
from refiner.worker.context import get_active_run_handle

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


@dataclass(slots=True)
class _FramesState:
    """Running state for staged frame parquet output within one chunk."""

    total_rows: int = 0
    file_index: int = 0
    bytes_written: int = 0
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
        self.bytes_written = 0


@dataclass(frozen=True, slots=True)
class _CompletedVideo:
    """Per-video write result merged back into the episode row."""

    video_key: str
    row_patch: Mapping[str, Any]
    feature: LeRobotFeatureInfo | None = None


class LeRobotWriterSink(BaseSink):
    """Skeleton sink for the new LeRobot writer implementation."""

    def __init__(
        self,
        output: DataFolderLike,
        data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        max_video_prepare_in_flight: int = 10,
    ):
        self.output = DataFolder.resolve(output)
        self.data_files_size_in_mb = data_files_size_in_mb
        self.video_files_size_in_mb = video_files_size_in_mb
        self.max_video_prepare_in_flight = max_video_prepare_in_flight
        self._states: dict[str, _LeRobotShardState] = {}
        self._async_window = AsyncWindow[None](
            max_in_flight=max_video_prepare_in_flight,
            preserve_order=False,
        )

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        """Submit one async write task per episode row in the shard-local block."""
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
            worker_token = get_active_run_handle().worker_token
            state = _LeRobotShardState(
                chunk_key=f"{shard_id}__w{worker_token}",
            )
            self._states[shard_id] = state
        return state

    def _commit_shard(self, state: _LeRobotShardState) -> None:
        """Write staged shard-local metadata files for one completed chunk."""
        metadata = state.metadata
        if metadata is None:
            return

        state.frames.close()
        # TODO: finalize and close shard-local video writers here once _write_video()
        # starts emitting real media segments.

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
            splits=metadata.info.splits,
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
        if not isinstance(row, LeRobotRow):
            raise TypeError("LeRobotWriterSink requires LeRobotRow inputs")

        if not isinstance(row.metadata, LeRobotMetadata):
            raise ValueError("LeRobot writer requires LeRobot metadata on each row")

        frames = (
            row.frames
            if isinstance(row.frames, Tabular)
            else Tabular.from_rows(row.frames)
        )
        if frames.num_rows <= 0:
            # empty frames
            return

        if state.metadata is None:
            state.metadata = row.metadata
        elif (
            row.metadata.info.fps != state.metadata.info.fps
            or row.metadata.info.robot_type != state.metadata.info.robot_type
        ):
            raise ValueError(
                "LeRobot writer requires stable fps and robot_type within each shard"
            )

        # compute features
        if not state.features:
            for key, value in row._row.items():
                spec = infer_feature_info(value)
                if spec is not None:
                    state.features.setdefault(key, spec)

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
                            else row.metadata.info.fps
                        ),
                    )
                    if spec is not None:
                        state.features.setdefault(key, spec)

            for key, spec in default_feature_info_by_key().items():
                state.features.setdefault(key, spec)

        episode_row_patch: dict[str, Any] = {
            "meta/episodes/chunk_index": state.chunk_key,
            "meta/episodes/file_index": 0,
        }

        # video work
        video_tasks = [
            asyncio.create_task(self._write_video(state, video_ref))
            for video_ref in row.videos.values()
        ]
        video_features: dict[str, LeRobotFeatureInfo] = {}

        # write frames hopefully while videos are inflight. get frame information ready to patch episode
        episode_row_patch.update(self._write_frames(state, frames=frames))

        # get video results
        if video_tasks:
            state.has_videos = True
            for completed in await asyncio.gather(*video_tasks):
                episode_row_patch.update(completed.row_patch)
                if completed.feature is not None:
                    video_features[completed.video_key] = completed.feature

        written_row = row._row.update(episode_row_patch)

        state.features.update(video_features)
        state.episode_rows.append(written_row)
        state.frames.total_rows += frames.num_rows

    async def _write_video(
        self,
        state: _LeRobotShardState,
        video_ref: LeRobotVideoRef,
    ) -> _CompletedVideo:
        """Write one episode video and return the patch it produced."""
        # Hand the source video off to media/video once that module owns the
        # mux/remux/transcode details for LeRobot-compatible outputs.
        #
        # Append the prepared segment into the shard-local video files, rotating
        # files when they hit the configured byte limit.
        #
        # Return the segment metadata and any flattened stats directly in row_patch,
        # plus inferred feature info for the shard-level info file.
        _ = state
        return _CompletedVideo(video_key=video_ref.key, row_patch={})

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
        if "index" in table.schema.names:
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                index_column,
            )
        else:
            table = table.append_column("index", index_column)

        if "frame_index" not in table.schema.names:
            table = table.append_column(
                "frame_index",
                pa.array(range(row_count), type=pa.int64()),
            )

        # file switch
        if (
            frames_state.writer is not None
            and frames_state.bytes_written
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
        if frames_state.writer_file is not None:
            frames_state.bytes_written = int(frames_state.writer_file.tell())

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


__all__ = ["LeRobotWriterSink"]
