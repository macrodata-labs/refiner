from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from refiner.platform.client import MacrodataClient


@dataclass(frozen=True, slots=True)
class DebugSessionRecord:
    script_path: str
    project_root: str
    script_args: list[str]
    job_id: str
    base_url: str
    workspace: str


def debug_sessions_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "macrodata" / "debug_sessions.json"


def _scope(client: MacrodataClient) -> tuple[str, str]:
    identity = client.verify_api_key()
    if identity.workspace is not None:
        workspace = identity.workspace.slug
    else:
        workspace = identity.key_id or identity.name
    return client.base_url.rstrip("/"), workspace


@contextmanager
def session_creation_lock(
    *, script: str | Path, client: MacrodataClient
) -> Iterator[None]:
    script_path = str(Path(script).expanduser().resolve())
    base_url, workspace = _scope(client)
    lock_key = hashlib.sha256(
        json.dumps([script_path, base_url, workspace], separators=(",", ":")).encode()
    ).hexdigest()
    lock_directory = debug_sessions_path().parent / "debug_session_locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    os.chmod(lock_directory, 0o700)
    lock_path = lock_directory / f"{lock_key}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


@contextmanager
def _locked_registry() -> Iterator[Path]:
    path = debug_sessions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield path


def _read(path: Path) -> list[DebugSessionRecord]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid debug session registry: {path}") from error
    rows = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid debug session registry: {path}")
    try:
        return [DebugSessionRecord(**row) for row in rows if isinstance(row, dict)]
    except TypeError as error:
        raise RuntimeError(f"invalid debug session registry: {path}") from error


def _write(path: Path, records: list[DebugSessionRecord]) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "sessions": [asdict(record) for record in records],
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".debug_sessions.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        temporary_path.unlink(missing_ok=True)


def find_session(
    *, script: str | Path, client: MacrodataClient
) -> DebugSessionRecord | None:
    script_path = str(Path(script).expanduser().resolve())
    base_url, workspace = _scope(client)
    with _locked_registry() as path:
        return next(
            (
                record
                for record in _read(path)
                if record.script_path == script_path
                and record.base_url == base_url
                and record.workspace == workspace
            ),
            None,
        )


def save_session(record: DebugSessionRecord) -> None:
    with _locked_registry() as path:
        records = [
            existing
            for existing in _read(path)
            if not (
                existing.script_path == record.script_path
                and existing.base_url == record.base_url
                and existing.workspace == record.workspace
            )
        ]
        records.append(record)
        _write(path, records)


def remove_session(*, script: str | Path, client: MacrodataClient) -> None:
    script_path = str(Path(script).expanduser().resolve())
    base_url, workspace = _scope(client)
    with _locked_registry() as path:
        records = [
            record
            for record in _read(path)
            if not (
                record.script_path == script_path
                and record.base_url == base_url
                and record.workspace == workspace
            )
        ]
        _write(path, records)


def new_session_record(
    *,
    script: str | Path,
    project_root: str | Path,
    script_args: list[str],
    job_id: str,
    client: MacrodataClient,
) -> DebugSessionRecord:
    base_url, workspace = _scope(client)
    return DebugSessionRecord(
        script_path=str(Path(script).expanduser().resolve()),
        project_root=str(Path(project_root).expanduser().resolve()),
        script_args=list(script_args),
        job_id=job_id,
        base_url=base_url,
        workspace=workspace,
    )


__all__ = [
    "DebugSessionRecord",
    "debug_sessions_path",
    "find_session",
    "new_session_record",
    "remove_session",
    "save_session",
    "session_creation_lock",
]
