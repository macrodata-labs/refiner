from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from fsspec.spec import AbstractFileSystem

from refiner.io import DataFolder
from refiner.io.datafolder import DataFolderLike
from refiner.io.fileset import DataFileSet
from refiner.pipeline.data.shard import FilePartsDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
import json
from functools import cached_property
import pyarrow as pa
import pyarrow.parquet as pq
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotTasks,
    LeRobotStatsFile,
    LeRobotMetadata,
    LeRobotTabular,
    remap_task_index_table,
)
from refiner.robotics.lerobot_format.metadata.metadata import merge_metadata
from refiner.worker.context import logger
from refiner.worker.metrics.api import log_throughput

_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes.jsonl"
_INFO_JSON = "meta/info.json"
_STATS_JSONL = "meta/episodes_stats.jsonl"
_TASKS_JSONL = "meta/tasks.jsonl"


class LeRobotV2_1EpisodeReader(BaseSource):
    """Read LeRobot v2.1 episode datasets."""

    name = "read_lerobot_v2_1"

    def __init__(
        self,
        inputs: DataFolderLike | Sequence[DataFolderLike],
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        skip_malformed_rows: bool = False,
    ) -> None:
        self._root_fileset = DataFileSet.resolve(
            inputs,
            fs=fs,
            storage_options=storage_options,
            expect_type="folder",
        )
        if not self._root_fileset.entries:
            raise ValueError("LeRobot v2.1 reader requires at least one dataset root")

        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.skip_malformed_rows = skip_malformed_rows

        self._roots: tuple[DataFolder, ...] | None = None
        self._episode_reader = None
        self._warned_malformed_row = False

    def describe(self) -> dict[str, Any]:
        inputs = [entry.abs_path() for entry in self._root_fileset.entries]
        return {"path": ", ".join(inputs), "inputs": inputs}

    @property
    def roots(self) -> tuple[DataFolder, ...]:
        if self._roots is None:
            roots = self._root_fileset.datafolders
            if not roots:
                raise ValueError(
                    "LeRobot v2.1 reader requires at least one dataset root"
                )
            self._roots = roots
        return self._roots

    def _json_reader(self):
        if self._episode_reader is None:
            from refiner.pipeline.sources.readers.json import JsonReader

            self._episode_reader = JsonReader(
                inputs=tuple(
                    (str(root.file(_DEFAULT_EPISODES_GLOB_ROOT).abs_path()), root.fs)
                    for root in self.roots
                ),
                recursive=False,
                target_shard_bytes=self.target_shard_bytes,
                num_shards=self.num_shards,
                file_path_column=None,
                lines=True,
            )
        return self._episode_reader

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self._root_fileset.required_refiner_extras()

    @cached_property
    def _metadata_bundle(
        self,
    ) -> tuple[
        tuple[LeRobotMetadata, ...],
        tuple[dict[int, int], ...],
        tuple[dict[int, LeRobotStatsFile], ...],
    ]:
        """Load and merge metadata for all dataset roots once per reader."""
        loaded = [self._load_metadata(root) for root in self.roots]
        metadata_list, stats_maps_list = zip(*loaded)
        merged_metadata, remaps = merge_metadata(metadata_list)
        return merged_metadata, remaps, tuple(stats_maps_list)

    def _load_metadata(
        self, root: DataFolder
    ) -> tuple[LeRobotMetadata, dict[int, LeRobotStatsFile]]:
        """Load one dataset root's info, tasks, and episode-level stats files."""

        with root.file(_INFO_JSON).open("rb") as handle:
            info = LeRobotInfo.from_json_dict(json.loads(handle.read()))

        with root.file(_TASKS_JSONL).open("rb") as handle:
            task_rows = [json.loads(line) for line in handle]
            tasks = LeRobotTasks.from_rows(task_rows)

        episode_stats_map: dict[int, LeRobotStatsFile] = {}
        with root.file(_STATS_JSONL).open("rb") as handle:
            for line in handle:
                record = json.loads(line)
                episode_idx = int(record["episode_index"])
                if episode_idx in episode_stats_map:
                    raise ValueError(
                        f"Duplicate episode_index {episode_idx} in episodes_stats.jsonl"
                    )
                episode_stats_map[episode_idx] = LeRobotStatsFile.from_json_dict(
                    record["stats"]
                )
        stats = LeRobotStatsFile.aggregate(list(episode_stats_map.values()))

        return LeRobotMetadata(info=info, stats=stats, tasks=tasks), episode_stats_map

    def list_shards(self) -> list[Shard]:
        return self._json_reader().list_shards()

    def _read_episode_frame_table(
        self,
        *,
        root: DataFolder,
        metadata: LeRobotMetadata,
        episode_idx: int,
        remap: Mapping[int, int],
    ) -> pa.Table:
        """Read the full frame parquet table for a single v2.1 episode."""
        chunks_size = metadata.info.chunks_size
        if chunks_size is None or chunks_size <= 0:
            raise ValueError(
                "LeRobot v2.1 info.json is missing required 'chunks_size' or it is <= 0"
            )

        episode_chunk = episode_idx // chunks_size

        rel_path = metadata.info.data_path.format(
            episode_chunk=episode_chunk, episode_index=episode_idx
        )

        with root.file(rel_path).open("rb") as handle:
            table = pq.read_table(handle)

        if remap:
            table = remap_task_index_table(table, remap)

        return table

    def _attach_episode_stats(
        self,
        batch: Tabular,
        episode_stats_map: dict[int, LeRobotStatsFile],
    ) -> Tabular:
        """Append flattened per-episode stats as columns to the episode batch."""
        episode_indices = batch.table.column("episode_index")
        stats_columns: dict[str, list[Any]] = {}

        for row_idx in range(batch.num_rows):
            episode_idx = episode_indices[row_idx].as_py()
            stats_file = episode_stats_map.get(episode_idx)
            if stats_file is None:
                raise ValueError(
                    f"Missing stats for episode {episode_idx} in episodes_stats.jsonl"
                )

            flat_stats = stats_file.flatten_fields()
            for key, value in flat_stats.items():
                if key not in stats_columns:
                    stats_columns[key] = [None] * batch.num_rows
                stats_columns[key][row_idx] = value

        table = batch.table
        for key, values in stats_columns.items():
            if all(value is None for value in values):
                continue
            table = set_or_append_column(table, key, pa.array(values))

        return batch.with_table(table)

    def _attach_video_info(
        self,
        batch: Tabular,
        metadata: LeRobotMetadata,
    ) -> Tabular:
        video_keys = metadata.info.video_keys
        if not video_keys or batch.num_rows == 0:
            return batch

        chunks_size = metadata.info.chunks_size
        fps = metadata.info.fps
        if chunks_size is None or chunks_size <= 0:
            raise ValueError(
                "LeRobot v2.1 info.json is missing required 'chunks_size' or it is <= 0"
            )
        if fps is None or fps <= 0:
            raise ValueError(
                "LeRobot v2.1 info.json is missing required 'fps' or it is <= 0"
            )

        video_path = metadata.info.video_path
        if video_path is None or video_path == "None" or not video_path.strip():
            raise ValueError("LeRobot v2.1 info.json is missing required 'video_path'")

        table = batch.table
        lengths = table.column("length")
        episode_indices = table.column("episode_index")

        for camera_key in video_keys:
            uris = []
            from_timestamps = []
            to_timestamps = []

            for row_idx in range(batch.num_rows):
                episode_idx = episode_indices[row_idx].as_py()
                length = lengths[row_idx].as_py()

                uri = video_path.format(
                    episode_chunk=episode_idx // chunks_size,
                    episode_index=episode_idx,
                    video_key=camera_key,
                )
                uris.append(uri)
                from_timestamps.append(0.0)
                to_timestamps.append(length / fps)

            table = set_or_append_column(
                table, f"videos/{camera_key}/uri", pa.array(uris, type=pa.string())
            )
            table = set_or_append_column(
                table,
                f"videos/{camera_key}/from_timestamp",
                pa.array(from_timestamps, type=pa.float64()),
            )
            table = set_or_append_column(
                table,
                f"videos/{camera_key}/to_timestamp",
                pa.array(to_timestamps, type=pa.float64()),
            )

        return batch.with_table(table)

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        metadata, remaps, stats_maps = self._metadata_bundle

        for part in descriptor.parts:
            part_shard = Shard.from_file_parts((part,))
            for batch in self._json_reader().read_shard(part_shard):
                if not isinstance(batch, Tabular):
                    raise TypeError(
                        "LeRobotV2_1EpisodeReader requires Tabular batches from JsonReader"
                    )

                root = self.roots[part.source_index]
                metadata_for_source = metadata[part.source_index]
                remap = remaps[part.source_index]
                episode_stats_map = stats_maps[part.source_index]

                if batch.num_rows <= 0:
                    continue

                batch = self._attach_episode_stats(batch, episode_stats_map)
                batch = self._attach_video_info(batch, metadata_for_source)

                episode_indices = batch.table.column("episode_index")
                frames_by_row = tuple(
                    self._read_episode_frame_table(
                        root=root,
                        metadata=metadata_for_source,
                        episode_idx=episode_indices[row_idx].as_py(),
                        remap=remap,
                    )
                    for row_idx in range(batch.num_rows)
                )

                lengths = batch.table.column("length")
                keep = [
                    table.num_rows == int(lengths[row_idx].as_py())
                    for row_idx, table in enumerate(frames_by_row)
                ]

                skipped = keep.count(False)
                if skipped and not self.skip_malformed_rows:
                    row_idx = keep.index(False)
                    expected = int(lengths[row_idx].as_py())
                    actual = frames_by_row[row_idx].num_rows
                    episode_index = int(episode_indices[row_idx].as_py())
                    error = (
                        f"episode {episode_index} expected {expected} "
                        f"frames, got {actual}"
                    )
                    raise ValueError(error)

                if skipped:
                    if not self._warned_malformed_row:
                        logger.warning("Skipping malformed LeRobot v2.1 episodes")
                        self._warned_malformed_row = True
                    log_throughput(
                        "malformed_lerobot_episodes_skipped",
                        skipped,
                        shard_id=shard.id,
                        unit="episodes",
                    )
                    batch = batch.with_table(
                        batch.table.filter(pa.array(keep, type=pa.bool_()))
                    )
                    frames_by_row = tuple(
                        table
                        for table, should_keep in zip(frames_by_row, keep, strict=True)
                        if should_keep
                    )
                    if batch.num_rows == 0:
                        continue

                yield LeRobotTabular(
                    batch,
                    metadata_by_row=(metadata_for_source,) * batch.num_rows,
                    frames_by_row=tuple(Tabular(table) for table in frames_by_row),
                    roots_by_row=(root,) * batch.num_rows,
                )
