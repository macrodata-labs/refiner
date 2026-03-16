from __future__ import annotations

import re
from pathlib import Path

from refiner.worker.id import worker_token


_RE_SHARD_FILENAME = re.compile(
    r"^(?P<shardid>[0-9a-f]+)(?:__w(?P<workerid>[^./]+))?\.json$"
)


def pending_filename(shard_id: str) -> str:
    return f"{shard_id}.json"


def leased_filename(shard_id: str, worker_id: str) -> str:
    return f"{shard_id}__w{worker_token(worker_id)}.json"


def parse_shard_filename(filename: str) -> tuple[str, str | None]:
    match = _RE_SHARD_FILENAME.match(filename)
    if not match:
        raise ValueError(f"Unrecognized shard filename: {filename!r}")
    return match.group("shardid"), match.group("workerid")


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
