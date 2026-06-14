from __future__ import annotations

import os
import tempfile
from pathlib import Path
from string import Formatter

import pyarrow as pa

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.pipeline.sources.readers.rerun import RerunRecording
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token

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
        _validate_filename_template(filename_template)
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.app_id = app_id
        self.write_footer = write_footer
        self._row_indices: dict[str, int] = {}

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("rerun",)

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        for row in block:
            recording = _recording_from_row(row)
            row_index = self._row_indices.get(shard_id, 0)
            self._row_indices[shard_id] = row_index + 1
            relpath = _render_relpath(
                self.filename_template,
                shard_id=shard_id,
                worker_id=get_active_worker_token(),
                row_index=row_index,
                segment_id=recording.segment_id,
            )
            self._write_recording(recording, relpath)
            count += 1
        return count

    def _write_recording(self, recording: RerunRecording, relpath: str) -> None:
        check_required_dependencies(
            "write_rerun",
            [("rerun", "rerun-sdk")],
            dist="rerun",
        )
        import rerun as rr

        def write_local(path: Path) -> None:
            with rr.RecordingStream(
                self.app_id,
                recording_id=recording.segment_id,
            ) as rec:
                rec.save(path, write_footer=self.write_footer)
                if recording.static is not None:
                    static = _sendable_table(recording.static.table)
                    if static.num_rows > 0 or static.num_columns > 0:
                        rec.send_dataframe(static)
                for table in recording.tables.values():
                    rec.send_dataframe(_sendable_table(table.table))

        target = self.output.file(relpath)
        if target.is_local:
            local_path = Path(target.abs_path())
            local_path.parent.mkdir(parents=True, exist_ok=True)
            write_local(local_path)
            return

        with tempfile.TemporaryDirectory(prefix="refiner-rerun-write-") as tmpdir:
            local_path = Path(tmpdir) / os.path.basename(relpath)
            write_local(local_path)
            DataFile.resolve(str(local_path)).copy(target)

    def on_shard_complete(self, shard_id: str) -> None:
        self._row_indices.pop(shard_id, None)

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


def _recording_from_row(row: Row) -> RerunRecording:
    value = row.get("rerun")
    if not isinstance(value, RerunRecording):
        raise ValueError("write_rerun requires rows with a RerunRecording in 'rerun'")
    return value


def _sendable_table(table: pa.Table) -> pa.Table:
    if "rerun_segment_id" in table.column_names:
        return table.drop(["rerun_segment_id"])
    return table


def _validate_filename_template(filename_template: str) -> None:
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
    _normalize_relpath(
        filename_template.format(
            shard_id="shard",
            worker_id="worker",
            row_index=0,
            segment_id="segment",
        ),
        "filename_template",
    )


def _render_relpath(
    filename_template: str,
    *,
    shard_id: str,
    worker_id: str,
    row_index: int,
    segment_id: str,
) -> str:
    return _normalize_relpath(
        filename_template.format(
            shard_id=shard_id,
            worker_id=worker_id,
            row_index=row_index,
            segment_id=segment_id,
        ),
        "rendered filename",
    )


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
