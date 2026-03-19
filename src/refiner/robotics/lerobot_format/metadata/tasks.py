from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True, slots=True)
class LeRobotTasks(Mapping[int, str]):
    index_to_task: Mapping[int, str]

    def __getitem__(self, key: int) -> str:
        return self.index_to_task[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self.index_to_task)

    def __len__(self) -> int:
        return len(self.index_to_task)

    @property
    def task_to_index(self) -> dict[str, int]:
        return {task: task_index for task_index, task in self.index_to_task.items()}


def parse_tasks_rows(rows: Sequence[Mapping[str, Any]]) -> LeRobotTasks:
    index_field = (
        "__index_level_0__" if rows and "__index_level_0__" in rows[0] else "task"
    )
    return LeRobotTasks(
        {
            int(row["task_index"]): str(row[index_field])
            for row in rows
            if row.get("task_index") is not None
            and isinstance(row.get(index_field), str)
        }
    )


def merge_tasks(
    tasks_by_dataset: Sequence[LeRobotTasks],
) -> tuple[LeRobotTasks, tuple[dict[int, int], ...]]:
    if not tasks_by_dataset:
        return LeRobotTasks({}), ()

    index_to_task = dict(tasks_by_dataset[0].index_to_task)
    task_to_index = {task: task_index for task_index, task in index_to_task.items()}
    next_index = max(index_to_task, default=-1) + 1
    remaps: list[dict[int, int]] = [{}]

    for source_tasks in tasks_by_dataset[1:]:
        remap: dict[int, int] = {}
        for source_index, task in source_tasks.items():
            merged_index = task_to_index.get(task)
            if merged_index is None:
                merged_index = next_index
                next_index += 1
                task_to_index[task] = merged_index
                index_to_task[merged_index] = task
            remap[int(source_index)] = merged_index
        remaps.append(
            {} if all(source == target for source, target in remap.items()) else remap
        )

    return LeRobotTasks(index_to_task), tuple(remaps)


def remap_task_index_table(
    table: pa.Table,
    remap: Mapping[int, int],
) -> pa.Table:
    if not remap:
        return table
    if "task_index" not in table.schema.names:
        raise ValueError(
            "LeRobot frame parquet is missing required 'task_index' column"
        )
    task_index_column = table.column("task_index").combine_chunks()
    task_index_type = task_index_column.type
    source_indices = pa.array(list(remap), type=task_index_type)
    target_indices = pa.array(
        [remap[source_index] for source_index in remap],
        type=task_index_type,
    )
    matches = pc.call_function(
        "index_in",
        [task_index_column],
        options=pc.SetLookupOptions(source_indices),
    )
    mapped_values = pc.take(target_indices, matches)
    unmatched = pc.call_function(
        "and_kleene",
        [
            pc.call_function("is_valid", [task_index_column]),
            pc.call_function("is_null", [mapped_values]),
        ],
    )
    if pc.call_function("any", [unmatched]).as_py():
        invalid_indices = pc.call_function(
            "unique", [task_index_column.filter(unmatched)]
        ).to_pylist()
        raise ValueError(
            "LeRobot frame parquet contains task_index values missing from "
            f"source task metadata: {invalid_indices}"
        )
    return table.set_column(
        table.schema.get_field_index("task_index"),
        "task_index",
        mapped_values,
    )


__all__ = [
    "LeRobotTasks",
    "merge_tasks",
    "parse_tasks_rows",
    "remap_task_index_table",
]
