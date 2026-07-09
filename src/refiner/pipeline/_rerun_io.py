from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import ClassVar
from typing import cast

from refiner.io import DataFile
from refiner.pipeline.data.tabular import Tabular


class LocalRrd:
    def __init__(self, source: DataFile) -> None:
        self.source = source
        self.tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def open(self) -> Path:
        if self.path is not None:
            return self.path
        if self.source.is_local:
            self.path = Path(self.source.abs_path())
            return self.path
        self.tmpdir = tempfile.TemporaryDirectory(prefix="refiner-rerun-")
        name = os.path.basename(self.source.path) or "recording.rrd"
        self.path = Path(self.tmpdir.name) / name
        self.source.copy(str(self.path))
        return self.path

    def close(self) -> None:
        if self.tmpdir is not None:
            self.tmpdir.cleanup()
            self.tmpdir = None
        self.path = None

    def __enter__(self) -> Path:
        return self.open()

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, object]:
        return {
            "source": self.source,
            "path": str(self.path) if self.path is not None else None,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.source = cast(DataFile, state["source"])
        self.tmpdir = None
        path = state.get("path")
        self.path = Path(path) if isinstance(path, str) else None


@dataclass(frozen=True, slots=True)
class RerunRecording:
    """Columnar Rerun recording data loaded from one RRD segment."""

    __refiner_side_data__: ClassVar[bool] = True

    segment_id: str
    source_path: str
    tables: Mapping[str, Tabular]
    static: Tabular | None = None
    source_file: DataFile | None = None
    local_source: LocalRrd | None = None
    application_id: str | None = None
    recording_id: str | None = None
    contents: tuple[str, ...] | None = None
    timelines: tuple[str, ...] | None = None
    include_static: bool = True
    use_source_chunks: bool = True
    source_recording_count: int | None = None


__all__ = ["LocalRrd", "RerunRecording"]
