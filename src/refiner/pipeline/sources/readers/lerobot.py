from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fsspec.spec import AbstractFileSystem

from refiner.io import DataFolder
from refiner.io.datafolder import DataFolderLike
from refiner.io.fileset import DataFileSet
from refiner.pipeline.data.shard import FilePartsDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.parquet import ParquetReader
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotStatsFile,
    LeRobotTabular,
    LeRobotTasks,
    merge_metadata,
    remap_task_index_table,
)
from refiner.worker.context import logger
from refiner.worker.metrics.api import log_throughput

_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
_TASKS_PARQUET = "meta/tasks.parquet"


class LeRobotEpisodeReader(BaseSource):
    """Read LeRobot episode datasets into `LeRobotTabular` blocks.

    Episode parquet shards are loaded through `ParquetReader`, then hydrated
    with dataset metadata and per-episode frame tabulars from the corresponding
    LeRobot dataset roots.
    """

    name = "read_lerobot"

    def __init__(
        self,
        inputs: DataFolderLike | Sequence[DataFolderLike],
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        arrow_batch_size: int = 65536,
        split_row_groups: bool = True,
        skip_malformed_rows: bool = False,
        limit: int | None = None,
    ) -> None:
        """Create a LeRobot episode reader over one or more dataset roots.

        The shard-planning arguments apply to the episode parquet files under
        each dataset root's `meta/episodes` directory.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        self._root_fileset = DataFileSet.resolve(
            inputs,
            fs=fs,
            storage_options=storage_options,
            expect_type="folder",
        )
        if not self._root_fileset.entries:
            raise ValueError("LeRobot reader requires at least one dataset root")
        self._roots: tuple[DataFolder, ...] | None = None
        self._episode_reader: ParquetReader | None = None
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.arrow_batch_size = arrow_batch_size
        self.split_row_groups = split_row_groups
        self.dtypes = None
        self._last_frame_table: tuple[tuple[int, Any, Any], pa.Table] | None = None
        self.skip_malformed_rows = skip_malformed_rows
        self._warned_malformed_row = False
        self._limit = limit
        self._emitted_rows = 0

    def describe(self) -> dict[str, Any]:
        inputs = [entry.abs_path() for entry in self._root_fileset.entries]
        description: dict[str, Any] = {"path": ", ".join(inputs), "inputs": inputs}
        if self._limit is not None:
            description["limit"] = self._limit
        return description

    def list_shards(self) -> list[Shard]:
        if self._limit == 0:
            return []
        return self._parquet_reader().list_shards()

    def read(self) -> Iterator[SourceUnit]:
        self._emitted_rows = 0
        yield from super().read()

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one planned episode shard and emit `LeRobotTabular` blocks."""
        if self._remaining_limit() == 0:
            return
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        metadata, remaps = self._metadata_bundle
        for part in descriptor.parts:
            if self._remaining_limit() == 0:
                return
            part_shard = Shard.from_file_parts((part,))
            for batch in self._parquet_reader().read_shard(part_shard):
                remaining = self._remaining_limit()
                if remaining == 0:
                    return
                if not isinstance(batch, Tabular):
                    raise TypeError(
                        "LeRobotEpisodeReader requires Tabular batches from ParquetReader"
                    )
                if (
                    remaining is not None
                    and not self.skip_malformed_rows
                    and batch.num_rows > remaining
                ):
                    batch = batch.with_table(batch.table.slice(0, remaining))

                root = self.roots[part.source_index]
                metadata_for_source = metadata[part.source_index]
                remap = remaps[part.source_index]
                frame_tables = self._load_frame_tables(
                    source_index=part.source_index,
                    tabular=batch,
                    root=root,
                    metadata=metadata_for_source,
                    remap=remap,
                )
                if batch.num_rows <= 0:
                    continue

                frames_by_row = tuple(
                    self._slice_episode_frame_table(
                        row_idx=row_idx,
                        tabular=batch,
                        frame_tables=frame_tables,
                    )
                    for row_idx in range(batch.num_rows)
                )
                lengths = batch.columns[batch.index_by_name["length"]]
                keep = [
                    table.num_rows == int(lengths[row_idx].as_py())
                    for row_idx, table in enumerate(frames_by_row)
                ]
                skipped = keep.count(False)
                if skipped and not self.skip_malformed_rows:
                    row_idx = keep.index(False)
                    expected = int(lengths[row_idx].as_py())
                    actual = frames_by_row[row_idx].num_rows
                    episode_index = int(
                        self._episode_value(batch, row_idx, "episode_index")
                    )
                    error = (
                        f"episode {episode_index} expected {expected} "
                        f"frames, got {actual}"
                    )
                    raise ValueError(error)

                if skipped:
                    if not self._warned_malformed_row:
                        logger.warning("Skipping malformed LeRobot episodes")
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

                remaining = self._remaining_limit()
                if remaining == 0:
                    return
                if remaining is not None and batch.num_rows > remaining:
                    batch = batch.with_table(batch.table.slice(0, remaining))
                    frames_by_row = frames_by_row[:remaining]

                self._emitted_rows += batch.num_rows
                yield LeRobotTabular(
                    batch,
                    metadata_by_row=(metadata_for_source,) * batch.num_rows,
                    frames_by_row=tuple(Tabular(table) for table in frames_by_row),
                    roots_by_row=(root,) * batch.num_rows,
                )

    @property
    def roots(self) -> tuple[DataFolder, ...]:
        if self._roots is None:
            roots = self._root_fileset.datafolders
            if not roots:
                raise ValueError("LeRobot reader requires at least one dataset root")
            self._roots = roots
        return self._roots

    def _parquet_reader(self) -> ParquetReader:
        if self._episode_reader is None:
            self._episode_reader = ParquetReader(
                inputs=tuple(
                    (str(root.abs_paths(_DEFAULT_EPISODES_GLOB_ROOT)), root.fs)
                    for root in self.roots
                ),
                recursive=True,
                target_shard_bytes=self.target_shard_bytes,
                num_shards=self.num_shards,
                arrow_batch_size=self.arrow_batch_size,
                split_row_groups=self.split_row_groups,
                file_path_column=None,
            )
        return self._episode_reader

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self._root_fileset.required_refiner_extras()

    def _remaining_limit(self) -> int | None:
        if self._limit is None:
            return None
        return max(0, self._limit - self._emitted_rows)

    @cached_property
    def _metadata_bundle(
        self,
    ) -> tuple[tuple[LeRobotMetadata, ...], tuple[dict[int, int], ...]]:
        """Load and merge metadata for all dataset roots once per reader."""
        loaded = [self._load_metadata(root) for root in self.roots]
        return merge_metadata(loaded)

    def _load_metadata(self, root: DataFolder) -> LeRobotMetadata:
        """Load one dataset root's info, stats, and task metadata files."""
        with root.file(_INFO_JSON).open("rb") as handle:
            info = LeRobotInfo.from_json_dict(json.loads(handle.read()))

        stats_file = root.file(_STATS_JSON)
        if stats_file.exists():
            with stats_file.open("rb") as handle:
                stats = LeRobotStatsFile.from_json_dict(json.loads(handle.read()))
        else:
            stats = LeRobotStatsFile.from_json_dict({})

        with root.file(_TASKS_PARQUET).open("rb") as handle:
            tasks = LeRobotTasks.from_rows(pq.read_table(handle).to_pylist())

        return LeRobotMetadata(info=info, stats=stats, tasks=tasks)

    def _load_frame_tables(
        self,
        *,
        source_index: int,
        tabular: Tabular,
        root: DataFolder,
        metadata: LeRobotMetadata,
        remap: Mapping[int, int],
    ) -> dict[tuple[Any, Any], pa.Table]:
        """Load one reduced frame table per referenced `(chunk, file)` pair.

        Each table is narrowed to the enclosing dataset-index span needed by
        the current episode batch and task indices are remapped once at the
        shared-table level.
        """
        request_ranges = (
            tabular.table.select(
                [
                    "data/chunk_index",
                    "data/file_index",
                    "dataset_from_index",
                    "dataset_to_index",
                ]
            )
            .group_by(["data/chunk_index", "data/file_index"])
            .aggregate(
                [
                    ("dataset_from_index", "min"),
                    ("dataset_to_index", "max"),
                ]
            )
        )

        tables: dict[tuple[Any, Any], pa.Table] = {}
        range_chunks = request_ranges.column("data/chunk_index")
        range_files = request_ranges.column("data/file_index")
        range_from_indices = request_ranges.column("dataset_from_index_min")
        range_to_indices = request_ranges.column("dataset_to_index_max")
        for row_idx in range(request_ranges.num_rows):
            chunk = range_chunks[row_idx].as_py()
            file_idx = range_files[row_idx].as_py()
            from_idx = int(range_from_indices[row_idx].as_py())
            to_idx = int(range_to_indices[row_idx].as_py())
            table = self._get_frame_file_table(
                source_index=source_index,
                root=root,
                metadata=metadata,
                chunk=chunk,
                file_idx=file_idx,
            )
            if "index" not in table.schema.names:
                raise ValueError(
                    "LeRobot frame parquet is missing required 'index' column"
                )
            table = table.filter(
                (pc.field("index") >= pa.scalar(from_idx, type=pa.int64()))
                & (pc.field("index") < pa.scalar(to_idx, type=pa.int64()))
            )
            table = remap_task_index_table(table, remap)
            tables[(chunk, file_idx)] = table
        return tables

    def _get_frame_file_table(
        self,
        *,
        source_index: int,
        root: DataFolder,
        metadata: LeRobotMetadata,
        chunk: Any,
        file_idx: Any,
    ) -> pa.Table:
        """Read or reuse the last full frame parquet table for one source file."""
        cache_key = (source_index, chunk, file_idx)
        cached = self._last_frame_table
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        rel = metadata.info.data_path.format(chunk_index=chunk, file_index=file_idx)
        with root.file(rel).open("rb") as handle:
            table = pq.read_table(handle)
        self._last_frame_table = (cache_key, table)
        return table

    def _slice_episode_frame_table(
        self,
        *,
        row_idx: int,
        tabular: Tabular,
        frame_tables: Mapping[tuple[Any, Any], pa.Table],
    ) -> pa.Table:
        """Extract one episode's frame rows from the preloaded shared tables."""
        chunk = self._episode_value(tabular, row_idx, "data/chunk_index")
        file_idx = self._episode_value(tabular, row_idx, "data/file_index")
        from_idx = int(self._episode_value(tabular, row_idx, "dataset_from_index"))
        to_idx = int(self._episode_value(tabular, row_idx, "dataset_to_index"))
        table = frame_tables[(chunk, file_idx)].filter(
            (pc.field("index") >= pa.scalar(from_idx, type=pa.int64()))
            & (pc.field("index") < pa.scalar(to_idx, type=pa.int64()))
        )
        episode_index = int(self._episode_value(tabular, row_idx, "episode_index"))
        episode_index_column = pa.array(
            [episode_index] * table.num_rows, type=pa.int64()
        )
        return set_or_append_column(table, "episode_index", episode_index_column)

    @staticmethod
    def _episode_value(
        tabular: Tabular,
        row_idx: int,
        key: str,
    ) -> Any:
        """Read one scalar value from an episode batch without row materialization."""
        return tabular.columns[tabular.index_by_name[key]][row_idx].as_py()


__all__ = ["LeRobotEpisodeReader"]
