from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import tarfile

MAX_SOURCE_ARCHIVE_BYTES = 16 * 1024 * 1024
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
