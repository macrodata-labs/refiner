from __future__ import annotations

import re
from string import Formatter

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.sinks.assets import ASSET_ATTEMPT_DIR_RE
from refiner.pipeline.sinks.base import BaseSink
from refiner.worker.context import get_active_stage_index, get_finalized_workers

_REQUIRED_TEMPLATE_FIELDS = {"shard_id", "worker_id"}
_TOKEN_PATTERN = r"[0-9a-f]{12}"
_FIELD_PATTERNS = {
    # row.shard_id is not theoretically restricted to the planned Shard.id format,
    # but launched deterministic file sinks are expected to name outputs from
    # planned shard ids in practice, so cleanup matches that 12-hex shape here.
    "shard_id": _TOKEN_PATTERN,
    "worker_id": _TOKEN_PATTERN,
}
_DEFAULT_FIELD_PATTERN = r"[^/]+"


def _compile_output_path_patterns(filename_template: str) -> list[re.Pattern[str]]:
    path_parts: list[str] = []
    patterns: list[re.Pattern[str]] = []
    seen_fields: set[str] = set()

    for segment in (part for part in filename_template.split("/") if part):
        segment_parts: list[str] = []
        for literal_text, field_name, format_spec, conversion in Formatter().parse(
            segment
        ):
            segment_parts.append(re.escape(literal_text))
            if field_name is None:
                continue
            if conversion is not None or format_spec:
                raise ValueError(
                    "filename_template reducer matching only supports plain "
                    "named fields without conversion or format specifiers"
                )
            if not field_name.isidentifier():
                raise ValueError(
                    "filename_template reducer matching only supports plain named fields"
                )
            if field_name in seen_fields:
                segment_parts.append(f"(?P={field_name})")
                continue
            pattern = _FIELD_PATTERNS.get(field_name, _DEFAULT_FIELD_PATTERN)
            segment_parts.append(f"(?P<{field_name}>{pattern})")
            seen_fields.add(field_name)
        path_parts.append("".join(segment_parts))
        patterns.append(re.compile("^" + "/".join(path_parts) + "$"))

    missing_fields = sorted(_REQUIRED_TEMPLATE_FIELDS.difference(seen_fields))
    if missing_fields:
        raise ValueError(
            "filename_template reducer matching requires fields: "
            + ", ".join(f"{{{field_name}}}" for field_name in missing_fields)
        )

    return patterns


class FileCleanupReducerSink(BaseSink):
    """Delete non-finalized deterministic file-sink outputs."""

    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str,
        reducer_name: str,
        assets_subdir: str | None = None,
    ) -> None:
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.reducer_name = reducer_name
        self.assets_subdir = assets_subdir
        self._output_path_patterns = _compile_output_path_patterns(filename_template)
        self._cleanup_ran = False

    def write_shard_block(self, shard_id, block) -> None:
        del shard_id, block
        self._run_cleanup()

    @property
    def counts_output_rows(self) -> bool:
        return False

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.assets_subdir is not None:
            args["assets_subdir"] = self.assets_subdir
        return (
            self.reducer_name,
            "writer",
            args,
        )

    def _run_cleanup(self) -> None:
        if self._cleanup_ran:
            return
        self._cleanup_ran = True

        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                f"{self.reducer_name} requires an active reducer stage with a prior writer stage"
            )

        keep_pairs = {
            (row.shard_id, row.worker_token)
            for row in get_finalized_workers(stage_index=stage_index - 1)
        }

        literal_prefix = ""
        for literal_text, field_name, _format_spec, _conversion in Formatter().parse(
            self.filename_template
        ):
            literal_prefix += literal_text
            if field_name is not None:
                break
        listing_prefix = (
            "" if "/" not in literal_prefix else literal_prefix.rsplit("/", 1)[0]
        )
        if (
            self.assets_subdir is None
            and listing_prefix == ""
            and len(self._output_path_patterns) == 2
        ):
            try:
                root_entries = self.output.ls(listing_prefix, detail=False)
            except (FileNotFoundError, NotADirectoryError):
                root_entries = []
            paths_to_delete: set[str] = set()
            for rel_path in root_entries:
                if len(rel_path) != 27 or rel_path[12:15] != "__w":
                    continue
                shard_id = rel_path[:12]
                worker_id = rel_path[15:]
                if (shard_id, worker_id) not in keep_pairs:
                    paths_to_delete.add(rel_path)
            for path in sorted(paths_to_delete):
                try:
                    self.output.rm(path, recursive=True)
                except FileNotFoundError:
                    continue
            return
        paths = [listing_prefix]
        prefix_parts = [part for part in listing_prefix.split("/") if part]
        for pattern in self._output_path_patterns[len(prefix_parts) :]:
            next_paths: list[str] = []
            for path in paths:
                try:
                    next_paths.extend(
                        item
                        for item in self.output.ls(path, detail=False)
                        if pattern.fullmatch(item)
                    )
                except (FileNotFoundError, NotADirectoryError):
                    continue
            paths = next_paths

        paths_to_delete: set[str] = set()
        # Extra template fields are structure only. Authority is decided from
        # the finalized (shard_id, worker_id) pair extracted from the path.
        for rel_path in paths:
            match = self._output_path_patterns[-1].fullmatch(rel_path)
            if match is None:
                continue
            if (match.group("shard_id"), match.group("worker_id")) not in keep_pairs:
                paths_to_delete.add(rel_path)

        if self.assets_subdir is not None:
            asset_prefix = f"{self.assets_subdir.rstrip('/')}/"
            try:
                asset_paths = self.output.find(self.assets_subdir)
            except FileNotFoundError:
                asset_paths = []
            for rel_path in asset_paths:
                if not rel_path.startswith(asset_prefix):
                    continue
                attempt_dir = rel_path[len(asset_prefix) :].split("/", maxsplit=1)[0]
                match = ASSET_ATTEMPT_DIR_RE.fullmatch(attempt_dir)
                if match is None:
                    continue
                if (
                    match.group("shard_id"),
                    match.group("worker_id"),
                ) not in keep_pairs:
                    paths_to_delete.add(f"{asset_prefix}{attempt_dir}")

        for path in sorted(paths_to_delete):
            try:
                self.output.rm(path, recursive=True)
            except FileNotFoundError:
                continue


__all__ = ["FileCleanupReducerSink"]
