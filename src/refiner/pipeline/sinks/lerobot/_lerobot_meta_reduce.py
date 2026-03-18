from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import json
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.tracking.shards import count_block_by_shard
from refiner.io import DataFolder
from refiner.pipeline.sinks.base import (
    BaseSink,
    Block,
    ShardCounts,
    describe_datafolder_path,
)

from refiner.pipeline.sinks.lerobot._lerobot_stats import (
    _aggregate_stats,
    _extract_episode_stats,
    _serialize_stats,
)
from refiner.worker.context import (
    RunHandle,
    get_active_run_handle,
    get_active_runtime_lifecycle,
)


__all__ = ["LeRobotMetaReduceSink"]


@dataclass(slots=True)
class LeRobotMetaReduceSink(BaseSink):
    """Stage 2 sink: reduce chunked metadata and clean intermediate chunks."""

    config: Any

    _reduced: bool = False

    def write_block(self, block: Block) -> ShardCounts:
        counts = count_block_by_shard(block)
        if not self._reduced:
            _LeRobotMetaReducer(config=self.config).reduce()
            self._reduced = True
        return counts

    def close(self) -> None:
        if self._reduced:
            return
        _LeRobotMetaReducer(config=self.config).reduce()
        self._reduced = True

    def describe(self) -> tuple[str, str, dict[str, str]]:
        return (
            "write_lerobot_meta_reduce",
            "writer",
            {"path": describe_datafolder_path(self.config.output)},
        )


@dataclass(slots=True)
class _LeRobotMetaReducer:
    config: Any
    folder: DataFolder = field(init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(self.config.output)

    def reduce(self) -> None:
        finalized_chunk_keys = self._finalized_chunk_keys()
        episodes_rows = self._load_stage1_episode_rows(finalized_chunk_keys)
        if not episodes_rows:
            return

        task_to_index = self._load_stage1_tasks(finalized_chunk_keys)
        stats_list = []
        for row in episodes_rows:
            episode_stats = _extract_episode_stats(row)
            if episode_stats:
                stats_list.append(episode_stats)
        info = self._load_stage1_info(finalized_chunk_keys)

        for row in episodes_rows:
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0

        episodes_table = pa.Table.from_pylist(episodes_rows)
        with self.folder.open(
            "meta/episodes/chunk-000/file-000.parquet", mode="wb"
        ) as out:
            pq.write_table(
                episodes_table,
                out,
                compression="snappy",
                use_dictionary=True,
            )

        ordered_tasks = sorted(
            task_to_index.items(), key=lambda item: (item[1], item[0])
        )
        tasks_table = pa.Table.from_pydict(
            {
                "task": [task for task, _ in ordered_tasks],
                "task_index": [task_index for _, task_index in ordered_tasks],
            }
        )
        with self.folder.open("meta/tasks.parquet", mode="wb") as out:
            pq.write_table(
                tasks_table,
                out,
                compression="snappy",
                use_dictionary=True,
            )

        merged_stats = _aggregate_stats(stats_list)
        with self.folder.open("meta/stats.json", mode="wt", encoding="utf-8") as out:
            json.dump(_serialize_stats(merged_stats), out, indent=2, sort_keys=True)

        info.update(
            total_episodes=len(episodes_rows),
            total_frames=sum(
                max(
                    0,
                    int(row["dataset_to_index"]) - int(row["dataset_from_index"]),
                )
                for row in episodes_rows
            ),
            total_tasks=len(task_to_index),
            splits={"train": f"0:{len(episodes_rows)}"},
        )
        with self.folder.open("meta/info.json", mode="wt", encoding="utf-8") as out:
            json.dump(info, out, indent=2, sort_keys=True)

        self._cleanup_stage1_chunks(finalized_chunk_keys)

    def _load_stage1_episode_rows(
        self, finalized_chunk_keys: set[str]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rel in self._iter_stage1_episode_files(finalized_chunk_keys):
            with self.folder.open(rel, mode="rb") as src:
                table = pq.read_table(src)
            rows.extend(table.to_pylist())
        rows.sort(key=lambda row: int(row["episode_index"]))
        return rows

    def _load_stage1_tasks(self, finalized_chunk_keys: set[str]) -> dict[str, int]:
        task_to_index: dict[str, int] | None = None
        for rel in self._iter_stage1_task_files(finalized_chunk_keys):
            with self.folder.open(rel, mode="rb") as src:
                table = pq.read_table(src)
            current = {row["task"]: int(row["task_index"]) for row in table.to_pylist()}
            if task_to_index is not None and current != task_to_index:
                raise ValueError(
                    "LeRobot reduce encountered mismatched canonical task tables "
                    "across stage-1 shard outputs"
                )
            task_to_index = current
        return {} if task_to_index is None else task_to_index

    def _load_stage1_info(self, finalized_chunk_keys: set[str]) -> dict[str, Any]:
        info: dict[str, Any] | None = None
        for rel in self._iter_stage1_jsonl_files(
            finalized_chunk_keys, filename="info.jsonl"
        ):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                payloads = [line.strip() for line in src if line.strip()]
            if len(payloads) != 1:
                raise ValueError(
                    "LeRobot reduce requires exactly one stage-1 info record per shard"
                )
            item = json.loads(payloads[0])
            if not isinstance(item, Mapping):
                raise ValueError("LeRobot reduce encountered invalid stage-1 info")
            current = dict(item)
            if info is not None and current != info:
                raise ValueError(
                    "LeRobot reduce encountered mismatched stage-1 info metadata"
                )
            info = current
        if info is None:
            raise ValueError("LeRobot reduce is missing required stage-1 info metadata")
        return info

    def _cleanup_stage1_chunks(self, finalized_chunk_keys: set[str]) -> None:
        for rel_path in self._stage1_matches("meta/chunk-*"):
            try:
                self.folder.rm(rel_path, recursive=True)
            except FileNotFoundError:
                continue

        keep = {f"data/chunk-{chunk_key}" for chunk_key in finalized_chunk_keys}
        for rel_path in self._stage1_matches("data/chunk-*"):
            if rel_path in keep:
                continue
            try:
                self.folder.rm(rel_path, recursive=True)
            except FileNotFoundError:
                continue

        for rel_path in self._stage1_matches("videos/*/chunk-*"):
            if self._chunk_key_from_path(rel_path) in finalized_chunk_keys:
                continue
            try:
                self.folder.rm(rel_path, recursive=True)
            except FileNotFoundError:
                continue

    def _iter_stage1_episode_files(self, finalized_chunk_keys: set[str]) -> list[str]:
        return self._stage1_matches(
            "meta/chunk-*/episodes/file-*.parquet", finalized_chunk_keys
        )

    def _iter_stage1_jsonl_files(
        self, finalized_chunk_keys: set[str], *, filename: str
    ) -> list[str]:
        return self._stage1_matches(f"meta/chunk-*/{filename}", finalized_chunk_keys)

    def _iter_stage1_task_files(self, finalized_chunk_keys: set[str]) -> list[str]:
        return self._stage1_matches(
            "meta/chunk-*/tasks.parquet",
            finalized_chunk_keys,
        )

    def _stage1_matches(
        self, pattern: str, finalized_chunk_keys: set[str] | None = None
    ) -> list[str]:
        root = self.folder.path.rstrip("/")
        prefix = f"{root}/"
        matches = sorted(
            path[len(prefix) :] if path.startswith(prefix) else path
            for path in self.folder.fs.glob(self.folder._join(pattern))
        )
        if finalized_chunk_keys is None:
            return matches
        return [
            rel_path
            for rel_path in matches
            if self._chunk_key_from_path(rel_path) in finalized_chunk_keys
        ]

    @staticmethod
    def _chunk_key_from_path(rel_path: str) -> str | None:
        for part in rel_path.split("/"):
            if part.startswith("chunk-"):
                return part[6:]
        return None

    def _finalized_chunk_keys(self) -> set[str]:
        runtime_lifecycle = get_active_runtime_lifecycle()
        stage_index = get_active_run_handle().stage_index
        if runtime_lifecycle is None or stage_index is None or stage_index <= 0:
            raise ValueError("LeRobot stage-2 reduce requires active runtime context")
        rows = runtime_lifecycle.finalized_workers(stage_index=stage_index - 1)
        return {
            f"{row.shard_id}__w{RunHandle.worker_token_for(row.worker_id)}"
            for row in rows
        }
