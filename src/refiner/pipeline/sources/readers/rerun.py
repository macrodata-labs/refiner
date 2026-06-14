from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fsspec import AbstractFileSystem

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    PathSelection,
    path_selection_map,
)
from refiner.utils import check_required_dependencies
from refiner.video import VideoFrameSequence

RerunOutputMode = Literal["recording", "robotics"]

_RERUN_SEGMENT_ID = "rerun_segment_id"
_RERUN_COMPONENT_METADATA = b"rerun:component"
_RERUN_ENTITY_PATH_METADATA = b"rerun:entity_path"
_ROBOTICS_ROW_COLUMNS = frozenset(
    {"episode_id", "rerun", "frames", "fps", "robot_type"}
)
_RECORDING_ROW_COLUMNS = frozenset({"episode_id", "rerun"})


@dataclass(frozen=True, slots=True)
class RerunRecording:
    """Columnar Rerun recording data loaded from one RRD segment."""

    segment_id: str
    source_path: str
    tables: Mapping[str, Tabular]
    static: Tabular | None = None
    source_file: DataFile | None = None
    application_id: str | None = None
    recording_id: str | None = None
    contents: tuple[str, ...] | None = None
    timelines: tuple[str, ...] | None = None
    include_static: bool = True
    use_source_chunks: bool = True


class RerunReader(BaseReader):
    """Read Rerun RRD files as columnar recording rows or robotics episode rows."""

    name = "read_rerun"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: str | None = "file_path",
        output: RerunOutputMode = "recording",
        contents: str | Sequence[str] | None = None,
        timelines: Sequence[str] | None = None,
        primary_timeline: str | None = None,
        include_static: bool = True,
        materialize_tables: bool = True,
        include_recording: bool | None = None,
        fill_latest_at: bool = False,
        action_prefix: str = "/action",
        state_prefix: str = "/observation/state",
        camera_prefix: str = "/cam",
        actions: PathSelection | None = None,
        states: PathSelection | None = None,
        videos: PathSelection | None = None,
        fps: float | None = None,
        robot_type: str | None = None,
    ) -> None:
        if output not in ("recording", "robotics"):
            raise ValueError("output must be 'recording' or 'robotics'")
        if output == "robotics" and not materialize_tables:
            raise ValueError(
                "materialize_tables=False is only supported for recording output"
            )
        if output == "recording" and include_recording is False:
            raise ValueError(
                "include_recording=False is only supported for robotics output"
            )
        if fps is not None:
            fps = float(fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be > 0")
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".rrd",),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
        )
        self.output = output
        self.contents = _contents(contents)
        self.timelines = tuple(timelines) if timelines is not None else None
        self.primary_timeline = primary_timeline
        self.include_static = include_static
        self.materialize_tables = materialize_tables
        self.include_recording = (
            output == "recording" if include_recording is None else include_recording
        )
        self.fill_latest_at = fill_latest_at
        self.action_prefix = _normalize_entity_prefix(action_prefix)
        self.state_prefix = _normalize_entity_prefix(state_prefix)
        self.camera_prefix = _normalize_entity_prefix(camera_prefix)
        self.actions_explicit = actions is not None
        self.states_explicit = states is not None
        self.videos_explicit = videos is not None
        self.actions = _selection_map(
            actions,
            format_name="Rerun actions",
            derive_names_from_paths=False,
        )
        self.states = _selection_map(
            states,
            format_name="Rerun states",
            derive_names_from_paths=False,
        )
        self.videos = _selection_map(videos, format_name="Rerun videos")
        reserved_row_columns = (
            _ROBOTICS_ROW_COLUMNS if output == "robotics" else _RECORDING_ROW_COLUMNS
        )
        if file_path_column in reserved_row_columns:
            raise ValueError(
                f"file_path_column cannot use reserved Rerun {output} row "
                f"column {file_path_column!r}"
            )
        if output == "robotics":
            reserved_video_names = set(reserved_row_columns)
            if file_path_column is not None:
                reserved_video_names.add(file_path_column)
            video_collisions = set(self.videos).intersection(reserved_video_names)
            if video_collisions:
                raise ValueError(
                    "Rerun video output names cannot use reserved robotics row "
                    "columns: " + ", ".join(sorted(video_collisions))
                )
        self.fps = fps
        self.robot_type = robot_type

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("rerun",)

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "output": self.output,
                "contents": self.contents,
                "timelines": self.timelines,
                "primary_timeline": self.primary_timeline,
                "include_static": self.include_static,
                "materialize_tables": self.materialize_tables,
                "include_recording": self.include_recording,
                "fill_latest_at": self.fill_latest_at,
                "action_prefix": self.action_prefix,
                "state_prefix": self.state_prefix,
                "camera_prefix": self.camera_prefix,
                "actions": dict(self.actions) if self.actions_explicit else None,
                "states": dict(self.states) if self.states_explicit else None,
                "videos": dict(self.videos) if self.videos_explicit else None,
                "fps": self.fps,
                "robot_type": self.robot_type,
            }
        )
        return description

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        with ExitStack() as stack:
            local_files = []
            for part in descriptor.parts:
                source = self.fileset.resolve_file(part.source_index, part.path)
                local_files.append((source, stack.enter_context(_local_rrd(source))))
            yield from self._read_files(local_files)

    def _read_files(
        self,
        local_files: Sequence[tuple[DataFile, Path]],
    ) -> Iterator[SourceUnit]:
        if self.output == "recording" and not self.materialize_tables:
            check_required_dependencies(
                "read_rerun",
                [("rerun", "rerun-sdk")],
                dist="rerun",
            )
            server_fallback: list[tuple[DataFile, Path]] = []
            for source, local_path in local_files:
                rows = self._read_metadata_only_recording_rows(source, local_path)
                if rows:
                    yield from rows
                else:
                    server_fallback.append((source, local_path))
            if server_fallback:
                yield from self._read_files_with_server(server_fallback)
            return

        yield from self._read_files_with_server(local_files)

    def _read_files_with_server(
        self,
        local_files: Sequence[tuple[DataFile, Path]],
    ) -> Iterator[SourceUnit]:
        check_required_dependencies(
            "read_rerun",
            [("rerun", "rerun-sdk"), "datafusion"],
            dist="rerun",
        )
        import rerun as rr

        datasets = {
            f"recording_{index}": (str(local_path),)
            for index, (_source, local_path) in enumerate(local_files)
        }
        with rr.server.Server(datasets=cast(Any, datasets)) as server:
            client = server.client()
            for dataset_name, (source, local_path) in zip(
                datasets, local_files, strict=True
            ):
                dataset = client.get_dataset(dataset_name)
                yield from self._read_dataset(source, local_path, dataset)

    def _read_dataset(
        self,
        source: DataFile,
        local_path: Path,
        dataset: Any,
    ) -> Iterator[SourceUnit]:
        store_entries = (
            _recording_entries(local_path)
            if self.output == "recording" or self.include_recording
            else []
        )
        entries_by_recording_id = {entry.recording_id: entry for entry in store_entries}
        timelines = self.timelines
        if timelines is None:
            timelines = self._timelines(dataset.schema())
        for segment_id in dataset.segment_ids():
            store = entries_by_recording_id.get(segment_id)
            application_id = store.application_id if store is not None else None
            recording_id = store.recording_id if store is not None else segment_id
            view = dataset.filter_segments([segment_id])
            if self.output == "robotics":
                yield self._robotics_row(
                    view,
                    segment_id=segment_id,
                    source_path=source.abs_path(),
                    source_file=source,
                    application_id=application_id,
                    recording_id=recording_id,
                    timelines=timelines,
                )
            else:
                tables: dict[str, Tabular] = {}
                static = None
                if self.materialize_tables:
                    content_view = self._view_for_contents(view)
                    static = (
                        _collect_table(content_view.reader(index=None))
                        if self.include_static
                        else None
                    )
                    tables = {
                        timeline: Tabular(
                            _collect_table(
                                content_view.reader(
                                    index=timeline,
                                    fill_latest_at=self.fill_latest_at,
                                )
                            )
                        )
                        for timeline in timelines
                    }
                yield self._recording_row(
                    segment_id=segment_id,
                    source=source,
                    local_path=local_path,
                    tables=tables,
                    static=Tabular(static) if static is not None else None,
                    application_id=application_id,
                    recording_id=recording_id,
                )

    def _read_metadata_only_recording_rows(
        self,
        source: DataFile,
        local_path: Path,
    ) -> list[DictRow]:
        rows = []
        for store in _recording_entries(local_path):
            recording_id = str(store.recording_id)
            rows.append(
                self._recording_row(
                    segment_id=recording_id,
                    source=source,
                    local_path=local_path,
                    tables={},
                    static=None,
                    application_id=store.application_id,
                    recording_id=recording_id,
                )
            )
        return rows

    def _recording_row(
        self,
        *,
        segment_id: str,
        source: DataFile,
        local_path: Path,
        tables: Mapping[str, Tabular],
        static: Tabular | None,
        application_id: str | None,
        recording_id: str | None,
    ) -> DictRow:
        data: dict[str, Any] = {
            "episode_id": segment_id,
            "rerun": RerunRecording(
                segment_id=segment_id,
                source_path=source.abs_path(),
                tables=tables,
                static=static,
                source_file=source,
                application_id=application_id,
                recording_id=recording_id,
                contents=self.contents,
                timelines=self.timelines,
                include_static=self.include_static,
            ),
        }
        self._with_file_path(data, source)
        return DictRow(data)

    def _timelines(self, schema: Any) -> tuple[str, ...]:
        if self.timelines is not None:
            return self.timelines
        return tuple(str(index.name) for index in schema.index_columns())

    def _primary_timeline(self, timelines: Sequence[str]) -> str:
        if self.primary_timeline is not None:
            return self.primary_timeline
        for timeline in timelines:
            if timeline not in {"log_tick", "log_time", "real_time"}:
                return timeline
        if not timelines:
            raise ValueError("Rerun recording has no timelines")
        return timelines[0]

    def _view_for_contents(self, dataset_or_view: Any) -> Any:
        if self.contents is None:
            return dataset_or_view
        return dataset_or_view.filter_contents(self.contents)

    def _robotics_contents(self) -> tuple[str, ...]:
        if self.contents is not None:
            return self.contents
        contents: list[str] = []
        if self.actions_explicit:
            contents.extend(self.actions.values())
        else:
            contents.append(_prefix_contents(self.action_prefix))
        if self.states_explicit:
            contents.extend(self.states.values())
        else:
            contents.append(_prefix_contents(self.state_prefix))
        if self.videos_explicit:
            contents.extend(self.videos.values())
        else:
            contents.append(_prefix_contents(self.camera_prefix))
        return tuple(dict.fromkeys(contents))

    def _robotics_row(
        self,
        view: Any,
        *,
        segment_id: str,
        source_path: str,
        source_file: DataFile,
        application_id: str | None,
        recording_id: str,
        timelines: Sequence[str],
    ) -> DictRow:
        timeline = self._primary_timeline(timelines)
        contents = self._robotics_contents()
        content_view = view.filter_contents(contents)
        table = _collect_table(
            content_view.reader(
                index=timeline,
                fill_latest_at=self.fill_latest_at,
            )
        )
        frames = _robotics_frame_table(table, timeline=timeline)
        row: dict[str, Any] = {
            "episode_id": segment_id,
        }
        if self.include_recording:
            static = (
                _collect_table(content_view.reader(index=None))
                if self.include_static
                else None
            )
            row["rerun"] = RerunRecording(
                segment_id=segment_id,
                source_path=source_path,
                tables={timeline: Tabular(table)},
                static=Tabular(static) if static is not None else None,
                source_file=source_file,
                application_id=application_id,
                recording_id=recording_id,
                contents=tuple(contents),
                timelines=(timeline,),
                include_static=self.include_static,
            )
        if self.fps is not None:
            row["fps"] = self.fps
        if self.robot_type is not None:
            row["robot_type"] = self.robot_type

        component_columns = _component_column_maps(table)
        scalar_columns = component_columns.get("Scalars:scalars", {})
        action_columns = (
            _selected_columns(
                scalar_columns,
                self.actions,
                format_name="Rerun action",
            )
            if self.actions_explicit
            else _prefixed_columns(scalar_columns, self.action_prefix)
        )
        state_columns = (
            _selected_columns(
                scalar_columns,
                self.states,
                format_name="Rerun state",
            )
            if self.states_explicit
            else _prefixed_columns(scalar_columns, self.state_prefix)
        )
        if action_columns:
            frames = frames.append_column(
                "action",
                _list_column(_singleton_scalar_matrix(table, action_columns)),
            )
        if state_columns:
            frames = frames.append_column(
                "observation.state",
                _list_column(_singleton_scalar_matrix(table, state_columns)),
            )
        row["frames"] = Tabular(frames)

        image_columns = component_columns.get("EncodedImage:blob", {})
        camera_columns = (
            _selected_camera_columns(image_columns, self.videos)
            if self.videos_explicit
            else _camera_columns(image_columns, self.camera_prefix)
        )
        reserved_video_names = set(_ROBOTICS_ROW_COLUMNS)
        if self.file_path_column is not None:
            reserved_video_names.add(self.file_path_column)
        _validate_video_output_names(camera_columns, reserved=reserved_video_names)
        for name, column in camera_columns.items():
            values = table.column(column).combine_chunks()
            _require_dense_encoded_images(values, video_name=name)
            row[name] = VideoFrameSequence(
                lambda values=values: _iter_encoded_images(values),
                fps=self.fps or 30.0,
                frame_count=len(values),
            )

        if self.file_path_column is not None:
            row[self.file_path_column] = source_path
        return DictRow(row)


def _contents(contents: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if contents is None:
        return None
    if isinstance(contents, str):
        return (contents,)
    return tuple(contents)


def _normalize_entity_prefix(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Rerun entity prefixes must be non-empty")
    stripped = value.strip("/")
    return "/" if not stripped else "/" + stripped


def _prefix_contents(prefix: str) -> str:
    return "/**" if prefix == "/" else f"{prefix}/**"


def _selection_map(
    value: PathSelection | None,
    *,
    format_name: str,
    derive_names_from_paths: bool = True,
) -> dict[str, str]:
    return {
        name: _normalize_entity_prefix(path)
        for name, path in path_selection_map(
            value,
            format_name=format_name,
            derive_names_from_paths=derive_names_from_paths,
        ).items()
    }


def _collect_table(df: Any) -> pa.Table:
    table = df.to_arrow_table()
    if _RERUN_SEGMENT_ID in table.column_names and table.num_rows > 0:
        segment_ids = table.column(_RERUN_SEGMENT_ID)
        if segment_ids.null_count:
            table = table.filter(_is_valid(segment_ids))
    return table


def _recording_entries(local_path: Path) -> list[Any]:
    import rerun as rr

    try:
        return list(rr.experimental.RrdReader(local_path).recordings())
    except Exception:
        return []


class _local_rrd:
    def __init__(self, source: DataFile) -> None:
        self.source = source
        self.tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if self.source.is_local:
            return Path(self.source.abs_path())
        self.tmpdir = tempfile.TemporaryDirectory(prefix="refiner-rerun-")
        name = os.path.basename(self.source.path) or "recording.rrd"
        self.path = Path(self.tmpdir.name) / name
        self.source.copy(str(self.path))
        return self.path

    def __exit__(self, *args: object) -> None:
        if self.tmpdir is not None:
            self.tmpdir.cleanup()


def _metadata_text(metadata: Mapping[bytes, bytes], key: bytes) -> str | None:
    value = metadata.get(key)
    return value.decode("utf-8") if value is not None else None


def _component_column_maps(table: pa.Table) -> dict[str, dict[str, str]]:
    by_component: dict[str, dict[str, str]] = {}
    for field in table.schema:
        metadata = field.metadata or {}
        field_component = _metadata_text(metadata, _RERUN_COMPONENT_METADATA)
        if field_component is None:
            continue
        entity_path = _metadata_text(metadata, _RERUN_ENTITY_PATH_METADATA)
        if entity_path is not None:
            by_component.setdefault(field_component, {})[entity_path] = field.name
    return by_component


def _prefixed_columns(columns: Mapping[str, str], prefix: str) -> list[tuple[str, str]]:
    return sorted(
        (
            (path, column)
            for path, column in columns.items()
            if _matches_entity_prefix(path, prefix)
        ),
        key=lambda item: item[0],
    )


def _selected_columns(
    columns: Mapping[str, str],
    selected: Mapping[str, str],
    *,
    format_name: str,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _name, path in selected.items():
        column = columns.get(path)
        if column is None:
            raise KeyError(f"{format_name} entity path not found: {path}")
        out.append((path, column))
    return out


def _matches_entity_prefix(entity_path: str, prefix: str) -> bool:
    if prefix == "/":
        return entity_path.startswith("/")
    return entity_path == prefix or entity_path.startswith(f"{prefix}/")


def _camera_columns(columns: Mapping[str, str], prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entity_path, column in columns.items():
        if not _matches_entity_prefix(entity_path, prefix):
            continue
        out[entity_path.strip("/").replace("/", ".")] = column
    return out


def _selected_camera_columns(
    by_entity_path: Mapping[str, str],
    selected: Mapping[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, path in selected.items():
        column = by_entity_path.get(path)
        if column is None:
            raise KeyError(f"Rerun video entity path not found: {path}")
        out[name] = column
    return out


def _validate_video_output_names(
    selected: Mapping[str, str],
    *,
    reserved: set[str],
) -> None:
    collisions = set(selected).intersection(reserved)
    if collisions:
        raise ValueError(
            "Rerun video output names cannot use reserved robotics row columns: "
            + ", ".join(sorted(collisions))
        )


def _robotics_frame_table(table: pa.Table, *, timeline: str) -> pa.Table:
    columns: dict[str, pa.ChunkedArray] = {}
    if timeline in table.column_names:
        columns["frame_index"] = table.column(timeline)
    return pa.table(columns)


def _singleton_scalar_matrix(
    table: pa.Table,
    columns: Sequence[tuple[str, str]],
) -> np.ndarray:
    values = np.full((table.num_rows, len(columns)), np.nan, dtype=np.float64)
    for index, (_, column) in enumerate(columns):
        _fill_singleton_list_array(
            table.column(column).combine_chunks(), values[:, index]
        )
    return values


def _list_column(values: np.ndarray) -> pa.Array:
    if values.ndim != 2:
        raise ValueError("Rerun vector columns must be 2D")
    width = int(values.shape[1])
    if width <= 0:
        offsets = pa.array(np.zeros(values.shape[0] + 1, dtype=np.int32))
        return pa.ListArray.from_arrays(offsets, pa.array([], type=pa.float64()))
    flat_values = pa.array(
        np.ascontiguousarray(values).reshape(-1),
        type=pa.float64(),
    )
    offsets = pa.array(
        np.arange(0, len(flat_values) + width, width, dtype=np.int32),
    )
    return pa.ListArray.from_arrays(offsets, flat_values)


def _fill_singleton_list_array(array: pa.Array, out: np.ndarray) -> None:
    if len(array) != len(out):
        raise ValueError("Rerun vector column length mismatch")
    if len(array) == 0:
        return
    if not pa.types.is_list(array.type) and not pa.types.is_large_list(array.type):
        raise TypeError(f"Expected a Rerun list component column, got {array.type}")
    offsets = np.asarray(array.offsets)
    starts = offsets[:-1]
    ends = offsets[1:]
    valid = np.asarray(_is_valid(array), dtype=bool) & (ends > starts)
    if not valid.any():
        return
    values = np.asarray(array.values)
    out[valid] = values[starts[valid]]


def _require_dense_encoded_images(values: pa.Array, *, video_name: str) -> None:
    if not pa.types.is_list(values.type) and not pa.types.is_large_list(values.type):
        raise TypeError(
            f"Expected a Rerun encoded image list column, got {values.type}"
        )
    offsets = np.asarray(values.offsets)
    missing = np.asarray(_is_valid(values), dtype=bool) == 0
    if len(offsets) > 1:
        missing |= offsets[1:] <= offsets[:-1]
    if missing.any():
        raise ValueError(
            f"Rerun video {video_name!r} has missing frames on the primary timeline; "
            "use fill_latest_at=True or select a denser timeline"
        )


def _iter_encoded_images(values: pa.Array) -> Iterator[np.ndarray]:
    from io import BytesIO

    from PIL import Image

    for index in range(len(values)):
        data = _encoded_image_bytes(values, index)
        if data is None:
            continue
        with Image.open(BytesIO(data)) as image:
            yield np.asarray(image.convert("RGB"), dtype=np.uint8)


def _encoded_image_bytes(values: pa.Array, index: int) -> bytes | None:
    if not pa.types.is_list(values.type) and not pa.types.is_large_list(values.type):
        raise TypeError(
            f"Expected a Rerun encoded image list column, got {values.type}"
        )
    if not values[index].is_valid:
        return None
    outer_offsets = np.asarray(values.offsets)
    outer_start = int(outer_offsets[index])
    outer_end = int(outer_offsets[index + 1])
    if outer_end <= outer_start:
        return None
    inner = values.values
    if not pa.types.is_list(inner.type) and not pa.types.is_large_list(inner.type):
        value = values[index].as_py()
        if not value:
            return None
        return bytes(cast(bytes | bytearray | list[int], value[0]))
    inner_offsets = np.asarray(inner.offsets)
    byte_start = int(inner_offsets[outer_start])
    byte_end = int(inner_offsets[outer_start + 1])
    payload = inner.values.slice(byte_start, byte_end - byte_start)
    return np.asarray(payload).tobytes()


def _is_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.call_function("is_valid", [values])


__all__ = ["RerunReader", "RerunRecording", "RerunOutputMode"]
