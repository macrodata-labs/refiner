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
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import SourceUnit
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

_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
_TASKS_PARQUET = "meta/tasks.parquet"


class LeRobotEpisodeReader(ParquetReader):
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
    ) -> None:
        """Create a LeRobot episode reader over one or more dataset roots.

        The shard-planning arguments apply to the episode parquet files under
        each dataset root's `meta/episodes` directory.
        """
        self.roots = self._resolve_roots(
            inputs,
            fs=fs,
            storage_options=storage_options,
        )
        self._last_frame_table: tuple[tuple[int, Any, Any], pa.Table] | None = None

        super().__init__(
            inputs=tuple(
                (str(root.abs_paths(_DEFAULT_EPISODES_GLOB_ROOT)), root.fs)
                for root in self.roots
            ),
            fs=fs,
            storage_options=storage_options,
            recursive=True,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            arrow_batch_size=arrow_batch_size,
            split_row_groups=split_row_groups,
        )

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one planned episode shard and emit `LeRobotTabular` blocks."""
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        metadata, remaps = self._metadata_bundle
        for part in descriptor.parts:
            part_shard = Shard.from_file_parts((part,))
            for batch in super().read_shard(part_shard):
                if not isinstance(batch, Tabular):
                    raise TypeError(
                        "LeRobotEpisodeReader requires Tabular batches from ParquetReader"
                    )

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
                if batch.num_rows > 0:
                    yield LeRobotTabular(
                        batch,
                        metadata_by_row=(metadata_for_source,) * batch.num_rows,
                        frames_by_row=tuple(
                            Tabular(
                                self._slice_episode_frame_table(
                                    row_idx=row_idx,
                                    tabular=batch,
                                    frame_tables=frame_tables,
                                )
                            )
                            for row_idx in range(batch.num_rows)
                        ),
                        roots_by_row=(root,) * batch.num_rows,
                    )

    @staticmethod
    def _resolve_roots(
        inputs: DataFolderLike | Sequence[DataFolderLike],
        *,
        fs: AbstractFileSystem | None,
        storage_options: Mapping[str, Any] | None,
    ) -> tuple[DataFolder, ...]:
        """Resolve reader inputs into concrete LeRobot dataset roots.

        Inputs may be single paths, `(path, fs)` pairs, `DataFolder`s, or
        sequences of those values.
        """
        fileset = DataFileSet.resolve(
            inputs,
            fs=fs,
            storage_options=storage_options,
            expect_type="folder",
        )
        roots = fileset.datafolders
        if not roots:
            raise ValueError("LeRobot reader requires at least one dataset root")
        return roots

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
        for row in request_ranges.to_pylist():
            chunk = row["data/chunk_index"]
            file_idx = row["data/file_index"]
            from_idx = int(row["dataset_from_index_min"])
            to_idx = int(row["dataset_to_index_max"])
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
        if "episode_index" in table.schema.names:
            return table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                episode_index_column,
            )
        return table.append_column("episode_index", episode_index_column)

    @staticmethod
    def _episode_value(
        tabular: Tabular,
        row_idx: int,
        key: str,
    ) -> Any:
        """Read one scalar value from an episode batch without row materialization."""
        return tabular.columns[tabular.index_by_name[key]][row_idx].as_py()


__all__ = ["LeRobotEpisodeReader"]
