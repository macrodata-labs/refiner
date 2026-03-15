from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import json
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.tracking.shards import count_block_by_shard
from refiner.io import DataFolder
from refiner.pipeline.sinks.base import BaseSink, Block, ShardCounts

from refiner.pipeline.sinks.lerobot._lerobot_stats import (
    _aggregate_stats,
    _cast_stats_to_numpy,
    _serialize_stats,
)
from refiner.pipeline.sinks.lerobot._lerobot_writer_shard import (
    _DEFAULT_CODEBASE_VERSION,
)
from refiner.worker.metrics.context import (
    get_active_runtime_lifecycle,
    get_active_runtime_stage_index,
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


@dataclass(slots=True)
class _LeRobotMetaReducer:
    config: Any
    folder: DataFolder = field(init=False)

    def __post_init__(self) -> None:
        self.folder = DataFolder.resolve(
            self.config.root,
            fs=self.config.fs,
            storage_options=self.config.storage_options,
        )

    def reduce(self) -> None:
        finalized_chunk_keys = self._finalized_chunk_keys()
        episodes_rows = self._load_stage1_episode_rows(finalized_chunk_keys)
        if not episodes_rows:
            return

        tasks = self._load_stage1_tasks(finalized_chunk_keys)
        stats_list = self._load_stage1_stats(finalized_chunk_keys)
        infos = self._load_stage1_infos(finalized_chunk_keys)

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

        tasks_table = pa.table(
            {
                "task_index": list(range(len(tasks))),
                "task": tasks,
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

        info = self._merge_infos(infos=infos, tasks=tasks, episodes_rows=episodes_rows)
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

    def _load_stage1_tasks(self, finalized_chunk_keys: set[str]) -> list[str]:
        task_to_index: dict[str, int] = {}
        for rel in self._iter_stage1_jsonl_files(
            finalized_chunk_keys, filename="tasks.jsonl"
        ):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                for line in src:
                    payload = line.strip()
                    if not payload:
                        continue
                    item = json.loads(payload)
                    task = item["task"]
                    if not isinstance(task, str) or not task:
                        continue
                    raw_idx = int(item["task_index"]) if "task_index" in item else None
                    if raw_idx is None:
                        task_to_index.setdefault(task, len(task_to_index))
                        continue
                    existing = task_to_index.get(task)
                    if existing is None:
                        task_to_index[task] = raw_idx
                    else:
                        task_to_index[task] = min(existing, raw_idx)
        if not task_to_index:
            return []

        ordered = sorted(task_to_index.items(), key=lambda kv: (kv[1], kv[0]))
        return [task for task, _ in ordered]

    def _load_stage1_stats(
        self, finalized_chunk_keys: set[str]
    ) -> list[dict[str, dict[str, Any]]]:
        out: list[dict[str, dict[str, Any]]] = []
        for rel in self._iter_stage1_jsonl_files(
            finalized_chunk_keys, filename="stats.jsonl"
        ):
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

    def _load_stage1_infos(
        self, finalized_chunk_keys: set[str]
    ) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for rel in self._iter_stage1_jsonl_files(
            finalized_chunk_keys, filename="info.jsonl"
        ):
            with self.folder.open(rel, mode="rt", encoding="utf-8") as src:
                for line in src:
                    payload = line.strip()
                    if not payload:
                        continue
                    item = json.loads(payload)
                    if isinstance(item, Mapping):
                        infos.append(dict(item))
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
                int(row["dataset_to_index"]) - int(row["dataset_from_index"]),
            )
            for row in episodes_rows
        )

        return {
            "codebase_version": str(
                first["codebase_version"]
                if infos and first.get("codebase_version") is not None
                else _DEFAULT_CODEBASE_VERSION
            ),
            "chunks_size": int(first["chunks_size"])
            if infos
            else self.config.chunk_size,
            "data_files_size_in_mb": int(first["data_files_size_in_mb"])
            if infos
            else self.config.data_files_size_in_mb,
            "video_files_size_in_mb": int(first["video_files_size_in_mb"])
            if infos
            else self.config.video_files_size_in_mb,
            "data_path": str(first["data_path"]) if infos else None,
            "video_path": first["video_path"] if infos else None,
            "fps": int(first["fps"])
            if infos and first.get("fps") is not None
            else None,
            "robot_type": first["robot_type"] if infos else None,
            "features": first["features"] if infos else {},
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "splits": {"train": f"0:{total_episodes}"},
        }

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
        stage_index = get_active_runtime_stage_index()
        if runtime_lifecycle is None or stage_index is None or stage_index <= 0:
            raise ValueError("LeRobot stage-2 reduce requires active runtime context")
        rows = runtime_lifecycle.finalized_workers(stage_index=stage_index - 1)
        return {f"{row.shard_id}__w{row.worker_id}" for row in rows}
