from __future__ import annotations

import re
from pathlib import Path


_RE_SHARD_FILENAME = re.compile(
    r"^(?P<shardid>[0-9a-f]+)(?:__w(?P<workerid>\d+))?\.json$"
)


def pending_filename(shard_id: str) -> str:
    return f"{shard_id}.json"


def leased_filename(shard_id: str, worker_id: int) -> str:
    return f"{shard_id}__w{int(worker_id)}.json"


def parse_shard_filename(filename: str) -> tuple[str, int | None]:
    match = _RE_SHARD_FILENAME.match(filename)
    if not match:
        raise ValueError(f"Unrecognized shard filename: {filename!r}")
    worker_id = match.group("workerid")
    return match.group("shardid"), int(worker_id) if worker_id is not None else None


def safe_unlink(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def safe_replace(src: str, dst: str) -> bool:
    try:
        Path(src).replace(dst)
        return True
    except OSError:
        return False
