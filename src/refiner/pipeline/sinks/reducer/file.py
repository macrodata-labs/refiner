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


def _compile_managed_path_pattern(filename_template: str) -> re.Pattern[str]:
    parts: list[str] = []
    seen_fields: set[str] = set()

    for literal_text, field_name, format_spec, conversion in Formatter().parse(
        filename_template
    ):
        parts.append(re.escape(literal_text))
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
            # Repeated fields in the template must resolve to the same path segment.
            parts.append(f"(?P={field_name})")
            continue
        pattern = _FIELD_PATTERNS.get(field_name, _DEFAULT_FIELD_PATTERN)
        parts.append(f"(?P<{field_name}>{pattern})")
        seen_fields.add(field_name)

    missing_fields = sorted(_REQUIRED_TEMPLATE_FIELDS.difference(seen_fields))
    if missing_fields:
        raise ValueError(
            "filename_template reducer matching requires fields: "
            + ", ".join(f"{{{field_name}}}" for field_name in missing_fields)
        )

    return re.compile("^" + "".join(parts) + "$")


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
        self._managed_path_pattern = _compile_managed_path_pattern(filename_template)
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
            (
                row.shard_id,
                row.worker_token,
            )
            for row in get_finalized_workers(stage_index=stage_index - 1)
        }

        try:
            listed_paths = self.output.find("")
        except FileNotFoundError:
            listed_paths = []

        assets_prefix = (
            f"{self.assets_subdir.rstrip('/')}/"
            if self.assets_subdir is not None
            else None
        )

        removed_asset_attempts: set[str] = set()
        # Extra template fields are treated as structure only. Authority is decided
        # solely from the finalized (shard_id, worker_id) pair extracted from each
        # managed path.
        for rel_path in listed_paths:
            if not isinstance(rel_path, str) or not rel_path or rel_path == ".":
                continue
            if assets_prefix is not None and (
                rel_path == self.assets_subdir or rel_path.startswith(assets_prefix)
            ):
                attempt_dir = rel_path[len(assets_prefix) :].split("/", maxsplit=1)[0]
                match = ASSET_ATTEMPT_DIR_RE.fullmatch(attempt_dir)
                if match is None:
                    continue
                if (match.group("shard_id"), match.group("worker_id")) in keep_pairs:
                    continue
                if attempt_dir in removed_asset_attempts:
                    continue
                removed_asset_attempts.add(attempt_dir)
                try:
                    self.output.rm(f"{assets_prefix}{attempt_dir}", recursive=True)
                except FileNotFoundError:
                    continue
                continue

            match = self._managed_path_pattern.fullmatch(rel_path)
            if match is None:
                continue
            if (match.group("shard_id"), match.group("worker_id")) in keep_pairs:
                continue
            try:
                self.output.rm(rel_path)
            except FileNotFoundError:
                continue


__all__ = ["FileCleanupReducerSink"]
