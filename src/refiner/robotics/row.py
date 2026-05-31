from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

import pyarrow as pa

from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.datatype import asset_storage, asset_type
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.tabular import set_or_append_column
from refiner.video import VideoSource, video_from_storage_value

if TYPE_CHECKING:
    from refiner.robotics.tabular import RoboticsTabular


RoboticsRowT = TypeVar("RoboticsRowT", bound="RoboticsRow")
_FrameSource = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RoboticsRowSpec:
    episode_id_key: str | None = None
    task_key: str | None = None
    fps: float | None = None
    fps_key: str | None = None
    robot_type: str | None = None
    robot_type_key: str | None = None
    nested_frames_key: str | None = None
    frame_sources: Mapping[str, _FrameSource] | None = None
    video_sources: Mapping[str, tuple[str, str | None]] | None = None
    stats_key: str | None = "stats"
    stats_prefix: str = "stats/"

    @classmethod
    def from_options(
        cls,
        *,
        episode_id_key: str | None = None,
        task_key: str | None = None,
        fps: float | None = None,
        fps_key: str | None = None,
        robot_type: str | None = None,
        robot_type_key: str | None = None,
        nested_frames_key: str | None = None,
        timestamp_key: str | None = "timestamp",
        action_key: str | None = "action",
        state_key: str | Sequence[str] | None = "observation.state",
        extra_observation_keys: Mapping[str, str] | Iterable[str] | None = None,
        video_keys: Mapping[str, str] | Iterable[str] | None = None,
        schema: pa.Schema | None = None,
        stats_key: str | None = "stats",
        stats_prefix: str = "stats/",
    ) -> "_RoboticsRowSpec":
        return cls(
            episode_id_key=episode_id_key,
            task_key=task_key,
            fps=fps,
            fps_key=fps_key,
            robot_type=robot_type,
            robot_type_key=robot_type_key,
            nested_frames_key=nested_frames_key,
            frame_sources=_semantic_source_map(
                timestamp_key=timestamp_key,
                action_key=action_key,
                state_key=state_key,
                extra_observation_keys=extra_observation_keys,
            ),
            video_sources=_video_sources(schema=schema, video_keys=video_keys),
            stats_key=stats_key,
            stats_prefix=stats_prefix,
        )

    @property
    def frame_source_map(self) -> Mapping[str, _FrameSource]:
        return self.frame_sources or {}

    @property
    def video_source_map(self) -> Mapping[str, tuple[str, str | None]]:
        return self.video_sources or {}

    def wrap(self, row: Row) -> RoboticsRow:
        return _RoboticsRowView(row, self)


@runtime_checkable
class RoboticsRow(Protocol):
    """Common semantic view over one robotics episode."""

    @property
    def shard_id(self) -> str | None: ...

    @property
    def episode_id(self) -> str: ...

    @property
    def num_frames(self) -> int: ...

    @property
    def task(self) -> str | None: ...

    @property
    def fps(self) -> float | None: ...

    @property
    def robot_type(self) -> str | None: ...

    @property
    def videos(self) -> Mapping[str, VideoSource]: ...

    @property
    def stats(self) -> Mapping[str, Any]: ...

    @property
    def timestamps(self) -> Any: ...

    @property
    def actions(self) -> Any: ...

    @property
    def states(self) -> Any: ...

    def observations(self, name: str | None = None) -> Any: ...

    def with_timestamps(self: RoboticsRowT, values: Any) -> RoboticsRowT: ...

    def with_actions(self: RoboticsRowT, values: Any) -> RoboticsRowT: ...

    def with_observation(
        self: RoboticsRowT,
        key: str,
        values: Any,
    ) -> RoboticsRowT: ...

    def with_video(
        self: RoboticsRowT,
        key: str,
        video: VideoSource,
    ) -> RoboticsRowT: ...

    def select_frames(
        self: RoboticsRowT,
        indices: Sequence[int],
    ) -> RoboticsRowT: ...

    def to_frame_table(self) -> Tabular: ...

    def drop_stats(self: RoboticsRowT, feature: str) -> RoboticsRowT: ...

    def update(
        self: RoboticsRowT,
        patch: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> RoboticsRowT: ...


class _RoboticsRowView(Row, RoboticsRow):
    def __init__(
        self,
        row: Row,
        spec: _RoboticsRowSpec,
    ) -> None:
        self._row = row
        self._spec = spec

    def __getitem__(self, key: str) -> Any:
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        hidden = set(self._source_frame_keys())
        for key in tuple(hidden):
            if "/" in key and key not in self._row:
                hidden.add(key.split("/", 1)[0])
        nested_frames_key = _valid_nested_frames_key(
            self._row,
            self._spec.nested_frames_key,
        )
        if nested_frames_key is not None:
            hidden.add(nested_frames_key)
        hidden.update(
            source_key for source_key, _ in self._spec.video_source_map.values()
        )
        for key in tuple(hidden):
            if "/" in key and key not in self._row:
                hidden.add(key.split("/", 1)[0])
        for key in self._row:
            if key not in hidden:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def tabular_type(self) -> type["RoboticsTabular"]:
        from refiner.robotics.tabular import RoboticsTabular

        return RoboticsTabular

    @property
    def shard_id(self) -> str | None:
        return self._row.shard_id

    @property
    def episode_id(self) -> str:
        if self._spec.episode_id_key is None:
            return "-1"
        value = self._row.get(self._spec.episode_id_key)
        return str(value) if value is not None else "-1"

    @property
    def num_frames(self) -> int:
        table = self._nested_frame_table()
        if table is not None:
            return table.num_rows
        first_key = self._first_source_frame_key()
        if first_key is None:
            return -1
        return len(_get_path(self._row, first_key))

    @property
    def task(self) -> str | None:
        if self._spec.task_key is None:
            return None
        value = self._row.get(self._spec.task_key)
        return value if isinstance(value, str) else None

    @property
    def fps(self) -> float | None:
        if self._spec.fps is not None:
            return self._spec.fps
        if self._spec.fps_key is None:
            return None
        value = self._row.get(self._spec.fps_key)
        return float(value) if value is not None else None

    @property
    def robot_type(self) -> str | None:
        if self._spec.robot_type is not None:
            return self._spec.robot_type
        if self._spec.robot_type_key is None:
            return None
        value = self._row.get(self._spec.robot_type_key)
        return str(value) if value is not None else None

    @property
    def videos(self) -> Mapping[str, VideoSource]:
        videos: dict[str, VideoSource] = {}
        for key, (source_key, storage) in self._spec.video_source_map.items():
            value = _get_path(self._row, source_key, default=None)
            video = video_from_storage_value(storage, value, fps=self.fps or 30.0)
            if video is not None:
                videos[key] = video
        return videos

    @property
    def stats(self) -> Mapping[str, Any]:
        stats: dict[str, Any] = {}
        if self._spec.stats_key is not None:
            value = self._row.get(self._spec.stats_key)
            if isinstance(value, Mapping):
                stats.update(value)
        if self._spec.stats_prefix:
            for key in self._row:
                if not key.startswith(self._spec.stats_prefix):
                    continue
                rest = key[len(self._spec.stats_prefix) :]
                feature, _, stat_name = rest.partition("/")
                if not feature or not stat_name:
                    continue
                feature_stats = stats.setdefault(feature, {})
                if isinstance(feature_stats, dict):
                    feature_stats[stat_name] = self._row[key]
        return stats

    def _frame_values(self, key: str) -> Any:
        table = self._nested_frame_table()
        if table is not None:
            source_key = self._source_frame_key(key, table_names=table.names)
            if isinstance(source_key, tuple):
                return _concat_frame_values([table.column(key) for key in source_key])
            return table.column(source_key)
        source_key = self._source_frame_key(key)
        if isinstance(source_key, tuple):
            return _concat_frame_values(
                [_get_path(self._row, key) for key in source_key]
            )
        return _get_path(self._row, source_key)

    @property
    def timestamps(self) -> Any:
        return self._optional_frame_values("timestamp")

    @property
    def actions(self) -> Any:
        return self._optional_frame_values("action")

    @property
    def states(self) -> Any:
        return self._optional_frame_values("observation.state")

    def observations(self, name: str | None = None) -> Any:
        values: dict[str, Any] = {}
        states = self.states
        if states is not None:
            values["state"] = states
        for semantic_key in self._spec.frame_source_map:
            observation_key = _strip_observation_prefix(semantic_key)
            if observation_key == semantic_key or observation_key == "state":
                continue
            try:
                values[observation_key] = self._frame_values(semantic_key)
            except KeyError:
                continue
        values.update({f"videos/{key}": video for key, video in self.videos.items()})
        if name is None:
            return values
        normalized_name = _strip_observation_prefix(name)
        return values[normalized_name]

    def _optional_frame_values(self, key: str) -> Any:
        try:
            return self._frame_values(key)
        except KeyError:
            return None

    def with_timestamps(self, values: Any) -> "_RoboticsRowView":
        return self._with_frame_values("timestamp", values)

    def with_actions(self, values: Any) -> "_RoboticsRowView":
        return self._with_frame_values("action", values)

    def with_observation(self, key: str, values: Any) -> "_RoboticsRowView":
        return self._with_frame_values(_observation_semantic_key(key), values)

    def _with_frame_values(self, key: str, values: Any) -> "_RoboticsRowView":
        table = self._nested_frame_table()
        if table is None:
            mapped_source_key = self._spec.frame_source_map.get(key, key)
            source_key = (
                mapped_source_key if isinstance(mapped_source_key, str) else key
            )
            next_mapping = dict(self._spec.frame_source_map)
            next_mapping[key] = source_key
            return self._replace(
                self._row.update(_set_path(self._row, source_key, values)),
                frame_sources=next_mapping,
            )
        mapped_source_key = self._spec.frame_source_map.get(key, key)
        source_key = mapped_source_key if isinstance(mapped_source_key, str) else key
        frame_table = set_or_append_column(
            table.table,
            source_key,
            _as_arrow_column(
                values,
                length=table.num_rows,
                value_type=table.table.schema.field(source_key).type
                if source_key in table.table.column_names
                else None,
            ),
        )
        nested_frames_key = self._require_nested_frames_key()
        next_mapping = dict(self._spec.frame_source_map)
        next_mapping[key] = source_key
        return self._replace(
            self._row.update({nested_frames_key: table.with_table(frame_table)}),
            frame_sources=next_mapping,
        )

    def select_frames(self, indices: Sequence[int]) -> "_RoboticsRowView":
        table = self._nested_frame_table()
        if table is not None:
            selected = _select_frame_table(table, indices)
            nested_frames_key = self._require_nested_frames_key()
            return self._replace(
                self._row.update(
                    {nested_frames_key: selected, "length": selected.num_rows}
                )
            )

        patch: dict[str, Any] = {
            source_key: _take_values(_get_path(self._row, source_key), indices)
            for source_key in self._source_frame_keys()
            if _has_path(self._row, source_key)
        }
        if "length" in self._row:
            patch["length"] = len(indices)
        if "frame_index" in patch:
            patch["frame_index"] = list(range(len(indices)))
        return self._replace(self._row.update(patch))

    def to_frame_table(self) -> Tabular:
        table = self._nested_frame_table()
        if table is not None:
            return self._semantic_frame_table(table)
        return Tabular(
            pa.table(
                {
                    semantic_key: _source_values(self._row, source_key)
                    for semantic_key, source_key in self._spec.frame_source_map.items()
                    if _has_source(self._row, source_key)
                }
            )
        )

    def drop_stats(self, feature: str) -> "_RoboticsRowView":
        row = self._row
        if self._spec.stats_key is not None:
            value = row.get(self._spec.stats_key)
            if isinstance(value, Mapping) and feature in value:
                next_stats = dict(value)
                next_stats.pop(feature, None)
                row = row.update({self._spec.stats_key: next_stats})
        if self._spec.stats_prefix:
            prefix = f"{self._spec.stats_prefix}{feature}/"
            row = row.drop(*[key for key in row if key.startswith(prefix)])
        return self._replace(row)

    def with_video(self, key: str, video: VideoSource) -> "_RoboticsRowView":
        source_key = self._spec.video_source_map.get(key, (key, None))[0]
        next_sources = dict(self._spec.video_source_map)
        next_sources[key] = (source_key, None)
        return self._replace(
            self._row.update(_set_path(self._row, source_key, video)),
            video_sources=next_sources,
        )

    def update(
        self,
        patch: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> "_RoboticsRowView":
        return self._replace(self._row.update(patch, **kwargs))

    def drop(self, *keys: str) -> "_RoboticsRowView":
        return self._replace(self._row.drop(*keys))

    def with_shard_id(self, shard_id: str) -> "_RoboticsRowView":
        return self._replace(self._row.with_shard_id(shard_id))

    def _nested_frame_table(self) -> Tabular | None:
        nested_frames_key = _valid_nested_frames_key(
            self._row,
            self._spec.nested_frames_key,
        )
        if nested_frames_key is None:
            return None
        value = self._row.get(nested_frames_key)
        if value is None:
            return None
        if isinstance(value, Tabular):
            return value
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return None
        return Tabular.from_rows(_frame_rows(value))

    def _require_nested_frames_key(self) -> str:
        if self._spec.nested_frames_key is None:
            raise ValueError("row has no nested frame key")
        return self._spec.nested_frames_key

    def _replace(
        self,
        row: Row,
        *,
        frame_sources: Mapping[str, _FrameSource] | None = None,
        video_sources: Mapping[str, tuple[str, str | None]] | None = None,
    ) -> "_RoboticsRowView":
        spec = dataclass_replace(
            self._spec,
            frame_sources=(
                frame_sources
                if frame_sources is not None
                else self._spec.frame_source_map
            ),
            video_sources=(
                video_sources
                if video_sources is not None
                else self._spec.video_source_map
            ),
        )
        return _RoboticsRowView(row, spec)

    def _source_frame_key(
        self,
        semantic_key: str,
        *,
        table_names: Sequence[str] | None = None,
    ) -> _FrameSource:
        source_key = self._spec.frame_source_map.get(semantic_key, semantic_key)
        if isinstance(source_key, tuple):
            if table_names is not None and not all(
                key in table_names for key in source_key
            ):
                raise KeyError(semantic_key)
            if table_names is None and not all(
                _has_path(self._row, key) for key in source_key
            ):
                raise KeyError(semantic_key)
            return source_key
        if table_names is not None and source_key not in table_names:
            raise KeyError(semantic_key)
        if table_names is None and not _has_path(self._row, source_key):
            raise KeyError(semantic_key)
        return source_key

    def _source_frame_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for source_key in self._spec.frame_source_map.values():
            if isinstance(source_key, tuple):
                keys.extend(source_key)
            else:
                keys.append(source_key)
        return tuple(dict.fromkeys(keys))

    def _first_source_frame_key(self) -> str | None:
        for source_key in self._source_frame_keys():
            if _has_path(self._row, source_key):
                return source_key
        return None

    def _semantic_frame_table(self, table: Tabular) -> Tabular:
        columns: dict[str, pa.Array | pa.ChunkedArray] = {}
        for semantic_key, source_key in self._spec.frame_source_map.items():
            if isinstance(source_key, tuple):
                if all(key in table.names for key in source_key):
                    columns[semantic_key] = _concat_frame_values(
                        [table.column(key) for key in source_key]
                    )
            elif source_key in table.names:
                columns[semantic_key] = table.column(source_key)
        return Tabular(pa.table(columns))


def _robot_row_converter(
    *,
    episode_id_key: str | None = None,
    task_key: str | None = None,
    fps: float | None = None,
    fps_key: str | None = None,
    robot_type: str | None = None,
    robot_type_key: str | None = None,
    nested_frames_key: str | None = None,
    timestamp_key: str | None = "timestamp",
    action_key: str | None = "action",
    state_key: str | Sequence[str] | None = "observation.state",
    extra_observation_keys: Mapping[str, str] | Iterable[str] | None = None,
    video_keys: Mapping[str, str] | Iterable[str] | None = None,
    stats_key: str | None = "stats",
    stats_prefix: str = "stats/",
    schema: pa.Schema | None = None,
) -> Callable[[Row], Row]:
    spec = _RoboticsRowSpec.from_options(
        episode_id_key=episode_id_key,
        task_key=task_key,
        fps=fps,
        fps_key=fps_key,
        robot_type=robot_type,
        robot_type_key=robot_type_key,
        nested_frames_key=nested_frames_key,
        timestamp_key=timestamp_key,
        action_key=action_key,
        state_key=state_key,
        extra_observation_keys=extra_observation_keys,
        video_keys=video_keys,
        schema=schema,
        stats_key=stats_key,
        stats_prefix=stats_prefix,
    )

    def map_row(row: Row) -> Row:
        return cast(Row, spec.wrap(row))

    return map_row


def _valid_nested_frames_key(row: Row, key: str | None) -> str | None:
    if key is None or key not in row:
        return None
    value = row[key]
    if isinstance(value, Tabular):
        return key
    if isinstance(value, str | bytes):
        return None
    return key if isinstance(value, Sequence) else None


def _video_sources(
    *,
    schema: pa.Schema | None,
    video_keys: Mapping[str, str] | Iterable[str] | None,
) -> dict[str, tuple[str, str | None]]:
    sources: dict[str, tuple[str, str | None]] = {}
    if schema is not None:
        for field in schema:
            if asset_type(field) == "video":
                sources[field.name] = (field.name, asset_storage(field))
    if video_keys is None:
        return sources
    if isinstance(video_keys, Mapping):
        mapped_keys = cast(Mapping[str, str], video_keys)
        sources.update(
            {key: (source_key, None) for key, source_key in mapped_keys.items()}
        )
    else:
        sources.update({key: (key, None) for key in video_keys})
    return sources


def _semantic_source_map(
    *,
    timestamp_key: str | None = "timestamp",
    action_key: str | None = "action",
    state_key: str | Sequence[str] | None = "observation.state",
    extra_observation_keys: Mapping[str, str] | Iterable[str] | None = None,
) -> dict[str, _FrameSource]:
    fields: dict[str, _FrameSource] = {}
    if timestamp_key is not None:
        fields["timestamp"] = timestamp_key
    if action_key is not None:
        fields["action"] = action_key
    if state_key is not None:
        fields["observation.state"] = _state_source_key(state_key)
    fields.update(_observation_source_map(extra_observation_keys))
    return fields


def _state_source_key(state_key: str | Sequence[str]) -> _FrameSource:
    if isinstance(state_key, str):
        return state_key
    keys = tuple(state_key)
    if not keys:
        raise ValueError("state_key sequence cannot be empty")
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("state_key sequence values must be strings")
    return keys


def _observation_source_map(
    extra_observation_keys: Mapping[str, str] | Iterable[str] | None,
) -> dict[str, str]:
    if extra_observation_keys is None:
        return {}
    if isinstance(extra_observation_keys, Mapping):
        mapped_keys = cast(Mapping[str, str], extra_observation_keys)
        return {
            _observation_semantic_key(key): source_key
            for key, source_key in mapped_keys.items()
        }
    return {_observation_semantic_key(key): key for key in extra_observation_keys}


def _observation_semantic_key(key: str) -> str:
    return key if key.startswith("observation.") else f"observation.{key}"


def _strip_observation_prefix(key: str) -> str:
    prefix = "observation."
    return key[len(prefix) :] if key.startswith(prefix) else key


def _frame_rows(value: Sequence[Any]) -> list[Row]:
    return [
        item
        if isinstance(item, Row)
        else DictRow(_flatten_mapping(item))
        if isinstance(item, Mapping)
        else DictRow({"value": item})
        for item in value
    ]


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_mapping(item, path))
        else:
            flat[path] = item
    return flat


_MISSING = object()


def _get_path(row: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    if key in row:
        return row[key]
    current: Any = row
    for part in key.split("/"):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            if default is _MISSING:
                raise KeyError(key)
            return default
    return current


def _has_path(row: Mapping[str, Any], key: str) -> bool:
    try:
        _get_path(row, key)
    except KeyError:
        return False
    return True


def _source_keys(source_key: _FrameSource) -> tuple[str, ...]:
    return source_key if isinstance(source_key, tuple) else (source_key,)


def _has_source(row: Mapping[str, Any], source_key: _FrameSource) -> bool:
    return all(_has_path(row, key) for key in _source_keys(source_key))


def _source_values(row: Mapping[str, Any], source_key: _FrameSource) -> Any:
    if isinstance(source_key, tuple):
        return _concat_frame_values([_get_path(row, key) for key in source_key])
    values = _get_path(row, source_key)
    if isinstance(values, pa.ChunkedArray | pa.Array):
        return values
    return _frame_value_list(values)


def _set_path(row: Mapping[str, Any], key: str, values: Any) -> dict[str, Any]:
    if "/" not in key or key in row:
        return {key: values}
    head, *tail = key.split("/")
    current = dict(row.get(head, {})) if isinstance(row.get(head), Mapping) else {}
    root = current
    for part in tail[:-1]:
        child = current.get(part)
        next_child = dict(child) if isinstance(child, Mapping) else {}
        current[part] = next_child
        current = next_child
    current[tail[-1]] = values
    return {head: root}


def _select_frame_table(table: Tabular, indices: Sequence[int]) -> Tabular:
    selected = table.table.take(pa.array(indices, type=pa.int64()))
    if "frame_index" in selected.column_names:
        selected = set_or_append_column(
            selected,
            "frame_index",
            pa.array(range(selected.num_rows), type=pa.int64()),
        )
    return table.with_table(selected)


def _as_arrow_column(
    values: Any,
    *,
    length: int,
    value_type: pa.DataType | None = None,
) -> pa.Array | pa.ChunkedArray:
    if isinstance(values, pa.ChunkedArray):
        return values
    if isinstance(values, pa.Array):
        return values
    if isinstance(values, pa.Scalar):
        return pa.repeat(values, length)
    return pa.array(values, type=value_type)


def _concat_frame_values(values: Sequence[Any]) -> Any:
    columns = [_frame_value_list(value) for value in values]
    if not columns:
        return []
    length = len(columns[0])
    if any(len(column) != length for column in columns):
        raise ValueError("source keys must have matching frame counts")
    concatenated = [
        [
            item
            for column in columns
            for item in _frame_feature_values(column[frame_idx])
        ]
        for frame_idx in range(length)
    ]
    if any(isinstance(value, pa.ChunkedArray | pa.Array) for value in values):
        return pa.array(concatenated)
    return concatenated


def _frame_value_list(values: Any) -> list[Any]:
    if isinstance(values, pa.ChunkedArray | pa.Array):
        return values.to_pylist()
    if isinstance(values, str | bytes):
        raise TypeError("frame-aligned values must be a sequence of frames")
    if hasattr(values, "tolist"):
        value_list = values.tolist()
        return value_list if isinstance(value_list, list) else [value_list]
    return list(values)


def _frame_feature_values(value: Any) -> list[Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str | bytes):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _take_values(values: Any, indices: Sequence[int]) -> Any:
    if isinstance(values, pa.ChunkedArray):
        return values.take(pa.array(indices, type=pa.int64()))
    if isinstance(values, pa.Array):
        return values.take(pa.array(indices, type=pa.int64()))
    return [values[index] for index in indices]


__all__ = [
    "RoboticsRow",
]
