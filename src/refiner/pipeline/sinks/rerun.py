from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from string import Formatter
from typing import Any, Callable

import pyarrow as pa

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline._rerun_io import LocalRrd, RerunRecording
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput

_DEFAULT_FILENAME_TEMPLATE = "{shard_id}__w{worker_id}/{row_index}.rrd"


class RerunSink(BaseSink):
    """Write Rerun recording rows as distributed shard-local RRD files."""

    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = _DEFAULT_FILENAME_TEMPLATE,
        app_id: str = "refiner",
        write_footer: bool = True,
    ) -> None:
        template_fields = _validate_filename_template(filename_template)
        self.output = DataFolder.resolve(output)
        self._local_output_root = (
            self.output.abs_path() if self.output.is_local else None
        )
        self.filename_template = filename_template
        self._uses_row_index = "row_index" in template_fields
        self._uses_segment_id = "segment_id" in template_fields
        self._render_relpath = _compile_relpath_renderer(
            filename_template,
            uses_segment_id=self._uses_segment_id,
        )
        self.app_id = app_id
        self.write_footer = write_footer
        self._row_indices: dict[str, int] = {}
        self._written_relpaths: dict[str, set[str]] = {}
        self._created_local_parents: set[str] = set()

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("rerun",)

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        worker_id = get_active_worker_token()
        row_index = self._row_indices.get(shard_id, 0)
        local_output_root = self._local_output_root
        if (
            local_output_root is not None
            and self.filename_template == _DEFAULT_FILENAME_TEMPLATE
        ):
            parent = f"{local_output_root}/{shard_id}__w{worker_id}"
            self._ensure_local_parent(parent)
            for row in block:
                recording = _recording_from_row(row)
                self._write_recording(
                    recording,
                    f"{parent}/{row_index}.rrd",
                )
                row_index += 1
                count += 1
        else:
            written_relpaths = (
                None
                if self._uses_row_index
                else self._written_relpaths.setdefault(shard_id, set())
            )
            for row in block:
                recording = _recording_from_row(row)
                relpath = self._render_relpath(
                    shard_id=shard_id,
                    worker_id=worker_id,
                    row_index=row_index,
                    segment_id=recording.segment_id,
                )
                if written_relpaths is not None:
                    if relpath in written_relpaths:
                        raise ValueError(
                            "write_rerun filename_template rendered duplicate output path "
                            f"{relpath!r}; include {{row_index}} or another unique row field"
                        )
                    written_relpaths.add(relpath)
                self._write_recording(recording, relpath)
                row_index += 1
                count += 1
        self._row_indices[shard_id] = row_index
        if count:
            log_throughput("files_written", count, shard_id=shard_id, unit="files")
        return count

    def _write_recording(self, recording: RerunRecording, relpath: str) -> None:
        target = self.output.file(relpath)
        if (
            self.write_footer
            and recording.use_source_chunks
            and recording.source_file is not None
        ):
            if _can_copy_source_rrd(recording):
                recording.source_file.copy(target)
                return

        def write_local(path: Path | str) -> None:
            if (
                self.write_footer
                and recording.use_source_chunks
                and recording.source_file is not None
            ):
                _write_source_chunks(recording, path, application_id=self.app_id)
                return
            _write_recording_tables(
                recording,
                path,
                application_id=self.app_id,
                write_footer=self.write_footer,
            )

        local_output_root = self._local_output_root
        if local_output_root is not None:
            local_path = f"{local_output_root}/{relpath}"
            self._ensure_local_parent(os.path.dirname(local_path))
            write_local(local_path)
            return

        with tempfile.TemporaryDirectory(prefix="refiner-rerun-write-") as tmpdir:
            local_path = Path(tmpdir) / os.path.basename(relpath)
            write_local(local_path)
            DataFile.resolve(str(local_path)).copy(target)

    def on_shard_complete(self, shard_id: str) -> None:
        self._row_indices.pop(shard_id, None)
        if not self._uses_row_index:
            self._written_relpaths.pop(shard_id, None)

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_rerun",
            "writer",
            {
                "path": self.output.abs_path(),
                "filename_template": self.filename_template,
                "app_id": self.app_id,
                "write_footer": self.write_footer,
            },
        )

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_rerun_reduce",
        )

    def _ensure_local_parent(self, parent: str) -> None:
        if parent not in self._created_local_parents:
            Path(parent).mkdir(parents=True, exist_ok=True)
            self._created_local_parents.add(parent)


def _recording_from_row(row: Row) -> RerunRecording:
    value = row.get("rerun")
    if not isinstance(value, RerunRecording):
        raise ValueError("write_rerun requires rows with a RerunRecording in 'rerun'")
    return value


def _write_source_chunks(
    recording: RerunRecording,
    path: Path | str,
    *,
    application_id: str,
) -> None:
    local_source = recording.local_source
    local_source_path = local_source.path if local_source is not None else None
    if local_source_path is not None:
        _write_source_chunks_from_path(
            recording,
            path,
            local_path=local_source_path,
            application_id=application_id,
        )
        return
    source = recording.source_file
    if source is None:
        raise ValueError("Rerun source chunk write requires source_file")
    with LocalRrd(source) as local_path:
        _write_source_chunks_from_path(
            recording,
            path,
            local_path=local_path,
            application_id=application_id,
        )


def _write_source_chunks_from_path(
    recording: RerunRecording,
    path: Path | str,
    *,
    local_path: Path,
    application_id: str,
) -> None:
    if _can_copy_source_rrd(recording):
        try:
            os.link(local_path, path)
        except OSError:
            shutil.copyfile(local_path, path)
        return

    check_required_dependencies(
        "write_rerun",
        [("rerun", "rerun-sdk")],
        dist="rerun",
    )
    import rerun as rr

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="RRD file has no footer/manifest:.*",
        )
        reader = rr.experimental.RrdReader(local_path)
    store = _matching_store(reader, recording)
    stream = reader.stream(store=store)
    if recording.contents is not None:
        stream = stream.filter(content=recording.contents)
    if not recording.include_static:
        stream = stream.drop(is_static=True)
    stream = _filter_timelines(
        stream,
        reader=reader,
        store=store,
        recording=recording,
    )
    stream.write_rrd(
        path,
        application_id=recording.application_id or application_id,
        recording_id=recording.recording_id or recording.segment_id,
    )


def _can_copy_source_rrd(recording: RerunRecording) -> bool:
    return (
        recording.source_recording_count == 1
        and recording.contents is None
        and recording.timelines is None
        and recording.include_static
    )


def _filter_timelines(
    stream: Any,
    *,
    reader: Any,
    store: Any,
    recording: RerunRecording,
) -> Any:
    timelines = recording.timelines
    if timelines is None:
        return stream
    if len(timelines) == 1:
        dynamic = stream.filter(has_timeline=timelines[0])
        if not recording.include_static:
            return dynamic
        static = reader.stream(store=store).filter(is_static=True)
        if recording.contents is not None:
            static = static.filter(content=recording.contents)
        import rerun as rr

        return rr.experimental.LazyChunkStream.merge(static, dynamic)

    selected = set(timelines)

    def keep_selected(chunk: Any) -> tuple[Any, ...]:
        if chunk.is_static:
            return (chunk,) if recording.include_static else ()
        return (chunk,) if selected.intersection(chunk.timeline_names) else ()

    return stream.flat_map(keep_selected)


def _matching_store(reader: Any, recording: RerunRecording) -> Any:
    stores = list(reader.recordings())
    if not stores:
        return None
    for store in stores:
        if (
            recording.recording_id is not None
            and store.recording_id == recording.recording_id
            and (
                recording.application_id is None
                or store.application_id == recording.application_id
            )
        ):
            return store
    for store in stores:
        if store.recording_id == recording.segment_id:
            return store
    return stores[0]


def _write_recording_tables(
    recording: RerunRecording,
    path: Path | str,
    *,
    application_id: str,
    write_footer: bool,
) -> None:
    import rerun as rr

    static = (
        _sendable_static_table(recording.static.table)
        if recording.static is not None
        else None
    )
    dynamic_tables = [
        dynamic
        for table in recording.tables.values()
        if (dynamic := _sendable_dynamic_table(table.table)).num_columns > 0
    ]
    if (static is None or static.num_columns == 0) and not dynamic_tables:
        raise ValueError(
            "write_rerun cannot write a RerunRecording without materialized "
            "Rerun table columns; use write_footer=True for raw source chunk "
            "writes or read_rerun(..., materialize_tables=True)"
        )

    with rr.RecordingStream(
        recording.application_id or application_id,
        recording_id=recording.recording_id or recording.segment_id,
    ) as rec:
        rec.save(path, write_footer=write_footer)
        if static is not None and static.num_columns > 0:
            rec.send_dataframe(static)
        for dynamic in dynamic_tables:
            rec.send_dataframe(dynamic)


def _sendable_static_table(table: pa.Table) -> pa.Table:
    keep = [
        field.name
        for field in table.schema
        if _is_data_column(field) and _is_static_column(field)
    ]
    return table.select(keep) if keep else pa.table({})


def _sendable_dynamic_table(table: pa.Table) -> pa.Table:
    keep = [
        field.name
        for field in table.schema
        if field.name != "rerun_segment_id" and not _is_static_column(field)
    ]
    return table.select(keep) if keep else pa.table({})


def _is_data_column(field: pa.Field) -> bool:
    metadata = field.metadata or {}
    return metadata.get(b"rerun:kind") == b"data" or b"rerun:entity_path" in metadata


def _is_static_column(field: pa.Field) -> bool:
    return (field.metadata or {}).get(b"rerun:is_static") == b"true"


def _validate_filename_template(filename_template: str) -> set[str]:
    fields: set[str] = set()
    for _literal_text, field_name, format_spec, conversion in Formatter().parse(
        filename_template
    ):
        if field_name is None:
            continue
        if conversion is not None or format_spec:
            raise ValueError("filename_template only supports plain named fields")
        if field_name not in {"shard_id", "worker_id", "row_index", "segment_id"}:
            raise ValueError(
                "filename_template only supports shard_id, worker_id, "
                "row_index, and segment_id"
            )
        fields.add(field_name)
    missing = {"shard_id", "worker_id"}.difference(fields)
    if missing:
        raise ValueError(
            "filename_template requires fields: "
            + ", ".join(f"{{{field}}}" for field in sorted(missing))
        )
    if not fields.intersection({"row_index", "segment_id"}):
        raise ValueError(
            "filename_template requires {row_index} or {segment_id} so each "
            "Rerun row writes a distinct file"
        )
    _normalize_relpath(
        filename_template.format(
            shard_id="shard",
            worker_id="worker",
            row_index=0,
            segment_id="segment",
        ),
        "filename_template",
    )
    return fields


def _compile_relpath_renderer(
    filename_template: str,
    *,
    uses_segment_id: bool,
) -> Callable[..., str]:
    parts: list[tuple[str, str | None]] = []
    for literal_text, field_name, format_spec, conversion in Formatter().parse(
        filename_template
    ):
        if conversion is not None or format_spec:
            raise ValueError("filename_template only supports plain named fields")
        parts.append((literal_text, field_name))

    def render(
        *,
        shard_id: str,
        worker_id: str,
        row_index: int,
        segment_id: str,
    ) -> str:
        normalized_segment_id = (
            _normalize_path_segment(segment_id, "segment_id")
            if uses_segment_id
            else segment_id
        )
        pieces: list[str] = []
        for literal_text, field_name in parts:
            pieces.append(literal_text)
            if field_name is None:
                continue
            if field_name == "shard_id":
                pieces.append(shard_id)
            elif field_name == "worker_id":
                pieces.append(worker_id)
            elif field_name == "row_index":
                pieces.append(str(row_index))
            elif field_name == "segment_id":
                pieces.append(normalized_segment_id)
            else:
                raise AssertionError(f"unexpected filename field {field_name!r}")
        return _normalize_relpath("".join(pieces), "rendered filename")

    return render


def _render_relpath(
    filename_template: str,
    *,
    shard_id: str,
    worker_id: str,
    row_index: int,
    segment_id: str,
    uses_segment_id: bool,
) -> str:
    return _compile_relpath_renderer(
        filename_template,
        uses_segment_id=uses_segment_id,
    )(
        shard_id=shard_id,
        worker_id=worker_id,
        row_index=row_index,
        segment_id=segment_id,
    )


def _normalize_path_segment(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single relative path segment")
    return value


def _normalize_relpath(path: str, label: str) -> str:
    if path.startswith("/"):
        raise ValueError(f"{label} must be relative")
    parts = [part for part in path.split("/") if part]
    if not parts:
        raise ValueError(f"{label} must not be empty")
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"{label} must not contain '.' or '..' segments")
    return "/".join(parts)


__all__ = ["RerunSink"]
