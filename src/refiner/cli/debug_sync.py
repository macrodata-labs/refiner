from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import ModuleType

import cloudpickle

MAX_SOURCE_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_DEBUG_SYNC_BUNDLE_BYTES = 64 * 1024 * 1024
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
EXCLUDED_FILES = frozenset({".env"})


@dataclass(frozen=True, slots=True)
class SourceArchive:
    payload: bytes
    sha256: str
    file_count: int


@dataclass(frozen=True, slots=True)
class DebugSyncBundle:
    payload: bytes
    sha256: str
    source_sha256: str
    pipeline_sha256: str
    file_count: int


def _included_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not path.is_symlink()
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and relative.name not in EXCLUDED_FILES
        and not relative.name.startswith(".env.")
    )


def build_source_archive(path: str | Path) -> SourceArchive:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"source path is not a directory: {root}")
    files = sorted(
        (candidate for candidate in root.rglob("*") if _included_file(root, candidate)),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", compresslevel=6) as archive:
        for file_path in files:
            archive.add(
                file_path,
                arcname=file_path.relative_to(root).as_posix(),
                recursive=False,
            )
    payload = output.getvalue()
    if not payload:
        raise ValueError("source archive is empty")
    if len(payload) > MAX_SOURCE_ARCHIVE_BYTES:
        raise ValueError("compressed source archive exceeds 16 MiB")
    return SourceArchive(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        file_count=len(files),
    )


def find_project_root(script: str | Path) -> Path:
    script_path = Path(script).expanduser().resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    return script_path.parent


def _module_is_within(module: ModuleType, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    try:
        relative = Path(module_file).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return not any(part in EXCLUDED_PARTS for part in relative.parts)


@contextmanager
def pickle_project_modules_by_value(path: str | Path) -> Iterator[None]:
    """Make a debug payload independent of imports from synchronized source."""

    root = Path(path).expanduser().resolve()
    already_registered = set(cloudpickle.list_registry_pickle_by_value())
    registered: list[ModuleType] = []
    for module in tuple(sys.modules.values()):
        if (
            isinstance(module, ModuleType)
            and module.__name__ not in already_registered
            and _module_is_within(module, root)
        ):
            cloudpickle.register_pickle_by_value(module)
            registered.append(module)
    try:
        yield
    finally:
        for module in registered:
            cloudpickle.unregister_pickle_by_value(module)


def build_debug_sync_bundle(
    *,
    source_root: str | Path,
    pipeline_payload: bytes,
    pipeline_sha256: str,
    allocation_fingerprint: str,
) -> DebugSyncBundle:
    actual_pipeline_sha256 = hashlib.sha256(pipeline_payload).hexdigest()
    if actual_pipeline_sha256 != pipeline_sha256:
        raise ValueError("pipeline payload sha256 mismatch")
    if len(allocation_fingerprint) != 64 or not all(
        character in "0123456789abcdef" for character in allocation_fingerprint.lower()
    ):
        raise ValueError("allocation fingerprint must be 64 hexadecimal characters")
    source = build_source_archive(source_root)
    metadata = json.dumps(
        {
            "schema_version": 1,
            "allocation_fingerprint": allocation_fingerprint,
            "source_sha256": source.sha256,
            "pipeline_sha256": pipeline_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, content in (
            ("sync.json", metadata),
            ("source.tar.gz", source.payload),
            ("pipeline.cloudpickle", pipeline_payload),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(content))
    payload = output.getvalue()
    if len(payload) > MAX_DEBUG_SYNC_BUNDLE_BYTES:
        raise ValueError("debug sync bundle exceeds 64 MiB")
    return DebugSyncBundle(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_sha256=source.sha256,
        pipeline_sha256=pipeline_sha256,
        file_count=source.file_count,
    )
