from __future__ import annotations

import concurrent.futures
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from refiner.io import DataFolder
from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.tabular import set_or_append_column
from refiner.pipeline.sinks.base import BaseSink
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotStatsFile,
    LeRobotTasks,
)
from refiner.worker.context import (
    RunHandle,
    get_active_run_handle,
    get_active_runtime_lifecycle,
)


class LeRobotMetaReduceSink(BaseSink):
    """Reduce stage-1 LeRobot shard outputs into final dataset metadata."""

    def __init__(self, output: DataFolderLike):
        self.output = DataFolder.resolve(output)

    def write_shard_block(self, _shard_id: str, _block: Block) -> None:
        finalized_chunk_keys = self._finalized_chunk_keys()
        finalized_chunk_key_set = set(finalized_chunk_keys)

        listed_paths = sorted(
            path for path in self.output.find("", withdirs=True) if path and path != "."
        )
        listed_path_set = set(listed_paths)

        episode_paths = [
            path
            for chunk_key in finalized_chunk_keys
            if (path := f"meta/chunk-{chunk_key}/episodes/file-000.parquet")
            in listed_path_set
        ]
        tasks_paths = [
            path
            for chunk_key in finalized_chunk_keys
            if (path := f"meta/chunk-{chunk_key}/tasks.parquet") in listed_path_set
        ]
        info_paths = [
            path
            for chunk_key in finalized_chunk_keys
            if (path := f"meta/chunk-{chunk_key}/info.json") in listed_path_set
        ]

        # fetch everything in one go. we might have to change this later
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(
                1,
                min(32, len(episode_paths) + len(tasks_paths) + len(info_paths)),
            )
        ) as pool:
            episode_futures = [
                pool.submit(self._read_parquet_table, rel_path)
                for rel_path in episode_paths
            ]
            tasks_futures = [
                pool.submit(self._read_parquet_table, rel_path)
                for rel_path in tasks_paths
            ]
            info_futures = [
                pool.submit(self._read_info_json, rel_path) for rel_path in info_paths
            ]
            episode_tables = [future.result() for future in episode_futures]
            tasks_tables = [future.result() for future in tasks_futures]
            info_items = [future.result() for future in info_futures]

        if not episode_tables:
            return None

        # episodes
        episodes_table = pa.concat_tables(episode_tables)
        episode_indices = np.asarray(
            episodes_table.column("episode_index")
            .combine_chunks()
            .to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        # we keep all unique episode ids, and remap repeated ones to max(episode_ids)+1...
        episode_ids_in_sync = True
        duplicate_mask = np.zeros(episode_indices.shape[0], dtype=bool)
        if episode_indices.size > 1:
            _, first_indices = np.unique(episode_indices, return_index=True)
            duplicate_mask = np.ones(episode_indices.shape[0], dtype=bool)
            duplicate_mask[first_indices] = False
        if duplicate_mask.any():
            # episode ids between metadata and frames will no longer be in sync until the next read using refiner, since we're
            # only updating the episodes meta row and not frames (would be a pain)
            episode_ids_in_sync = False
            next_episode_index = int(episode_indices.max()) + 1
            episode_indices[duplicate_mask] = np.arange(
                next_episode_index,
                next_episode_index + int(duplicate_mask.sum()),
                dtype=np.int64,
            )
            episodes_table = set_or_append_column(
                episodes_table,
                "episode_index",
                pa.array(episode_indices, type=pa.int64()),
            )
        chunk_index = pa.array([0] * episodes_table.num_rows, type=pa.int64())
        episodes_table = set_or_append_column(
            episodes_table,
            "meta/episodes/chunk_index",
            chunk_index,
        )
        file_index = pa.array([0] * episodes_table.num_rows, type=pa.int64())
        episodes_table = set_or_append_column(
            episodes_table,
            "meta/episodes/file_index",
            file_index,
        )

        # tasks
        for current in tasks_tables[1:]:
            if not current.equals(tasks_tables[0]):
                raise ValueError(
                    "LeRobot reduce encountered mismatched canonical task tables "
                    "across stage-1 shard outputs"
                )
        tasks = (
            LeRobotTasks({})
            if not tasks_tables
            else LeRobotTasks.from_table(tasks_tables[0])
        )

        # info jsons
        if not info_items:
            raise ValueError("LeRobot reduce is missing required stage-1 info metadata")
        info = info_items[0]
        ref_info_fields = self._stable_info_fields(info)
        for current in info_items[1:]:
            if ref_info_fields != self._stable_info_fields(current):
                raise ValueError(
                    "LeRobot reduce encountered mismatched stage-1 info metadata"
                )

        info = replace(
            info,
            total_episodes=int(episodes_table.num_rows),
            total_frames=int(pc.sum(episodes_table.column("length")).as_py() or 0),
            total_tasks=len(tasks),
            splits={"train": f"0:{int(episodes_table.num_rows)}"},
            episode_ids_in_sync=episode_ids_in_sync,
        )

        stats_file = LeRobotStatsFile.aggregate_flat_table(episodes_table)

        # save final metadata
        with self.output.open(
            "meta/episodes/chunk-000/file-000.parquet", mode="wb"
        ) as out:
            pq.write_table(
                episodes_table,
                out,
                compression="snappy",
                use_dictionary=True,
            )

        with self.output.open("meta/tasks.parquet", mode="wb") as out:
            pq.write_table(
                tasks.to_table(),
                out,
                compression="snappy",
                use_dictionary=True,
            )

        with self.output.open("meta/stats.json", mode="wt", encoding="utf-8") as out:
            out.write(
                orjson.dumps(
                    stats_file.to_json_dict(),
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                ).decode("utf-8")
            )

        with self.output.open("meta/info.json", mode="wt", encoding="utf-8") as out:
            out.write(
                orjson.dumps(
                    info.to_json_dict(),
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                ).decode("utf-8")
            )

        # clean up metadata chunks + failed shards
        for rel_path in listed_paths:
            if rel_path.startswith("meta/chunk-"):
                pass
            elif rel_path.startswith("data/chunk-"):
                chunk_key = self._chunk_key_from_path(rel_path)
                if chunk_key in finalized_chunk_key_set:
                    continue
            elif rel_path.startswith("videos/") and "/chunk-" in rel_path:
                chunk_key = self._chunk_key_from_path(rel_path)
                if chunk_key in finalized_chunk_key_set:
                    continue
            else:
                continue
            try:
                self.output.rm(rel_path, recursive=True)
            except FileNotFoundError:
                continue

    def describe(self) -> tuple[str, str, dict[str, str]]:
        return (
            "write_lerobot_meta_reduce",
            "writer",
            {"path": self.output.abs_path()},
        )

    def _read_parquet_table(self, rel_path: str) -> pa.Table:
        with self.output.open(rel_path, mode="rb") as src:
            return pq.read_table(src)

    def _read_info_json(self, rel_path: str) -> LeRobotInfo:
        with self.output.open(rel_path, mode="rt", encoding="utf-8") as src:
            item = orjson.loads(src.read())
        if not isinstance(item, Mapping):
            raise ValueError("LeRobot reduce encountered invalid stage-1 info")
        return LeRobotInfo.from_json_dict(item)

    @staticmethod
    def _stable_info_fields(info: LeRobotInfo) -> tuple[object, ...]:
        return (
            info.codebase_version,
            info.fps,
            info.robot_type,
            info.data_files_size_in_mb,
            info.video_files_size_in_mb,
            info.data_path,
            info.video_path,
            dict(info.features),
        )

    @staticmethod
    def _chunk_key_from_path(rel_path: str) -> str | None:
        for part in rel_path.split("/"):
            if part.startswith("chunk-"):
                return part[6:]
        return None

    @staticmethod
    def _finalized_chunk_keys() -> list[str]:
        runtime_lifecycle = get_active_runtime_lifecycle()
        stage_index = get_active_run_handle().stage_index
        if runtime_lifecycle is None or stage_index is None or stage_index <= 0:
            raise ValueError("LeRobot stage-2 reduce requires active runtime context")
        return [
            f"{row.shard_id}__w{RunHandle.worker_token_for(row.worker_id)}"
            for row in runtime_lifecycle.finalized_workers(stage_index=stage_index - 1)
        ]


__all__ = ["LeRobotMetaReduceSink"]
