from __future__ import annotations

import json
import os
import time
import hashlib
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from refiner.pipeline.data.shard import (
    Shard,
    format_leased_filename,
    format_pending_filename,
    parse_shard_filename,
    strip_worker_suffix,
)
from refiner.worker.workdir import resolve_workdir


def _h_int(*parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


class ClaimPolicy:
    BLOCK_SIZE = 8

    def __init__(self, *, job_id: str, worker_id: int):
        self.job_id = str(job_id)
        self.worker_id = int(worker_id)

    @dataclass(frozen=True, slots=True)
    class _ShardKey:
        file_key: str
        start: int
        end: int
        shard_id: str

    def claim_key(
        self,
        *,
        previous: Shard | None,
        all_keys: Iterable[_ShardKey],
        pending_ids: set[str],
        try_claim: Callable[[_ShardKey], bool],
    ) -> _ShardKey | None:
        by_file_all: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)
        by_file_pending: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)
        by_file_pending_by_start: dict[tuple[str, int], ClaimPolicy._ShardKey] = {}

        for k in all_keys:
            by_file_all[k.file_key].append(k)
        for fk in by_file_all:
            by_file_all[fk].sort(key=lambda r: (r.start, r.end, r.shard_id))

        for fk, all_list in by_file_all.items():
            pending_for_file = [k for k in all_list if k.shard_id in pending_ids]
            if not pending_for_file:
                continue
            by_file_pending[fk] = pending_for_file
            for k in pending_for_file:
                by_file_pending_by_start[(fk, int(k.start))] = k
        for fk in by_file_pending:
            by_file_pending[fk].sort(key=lambda r: (r.start, r.end, r.shard_id))

        if not by_file_pending:
            return None

        def _try_file(
            file_key: str, prev: Shard | None
        ) -> ClaimPolicy._ShardKey | None:
            pending_list = by_file_pending.get(file_key, [])
            if not pending_list:
                return None

            if prev is not None and prev.file_key == file_key:
                cand = by_file_pending_by_start.get((file_key, int(prev.end)))
                if cand is not None and try_claim(cand):
                    return cand

            all_list = by_file_all[file_key]
            n = len(all_list)
            if n <= 0:
                return None
            bs = ClaimPolicy.BLOCK_SIZE
            num_blocks = (n + bs - 1) // bs
            offset = _h_int(self.job_id, str(self.worker_id), file_key) % max(
                1, num_blocks
            )
            for j in range(num_blocks):
                k = (offset + j) % num_blocks
                idx = k * bs
                if idx >= n:
                    continue
                shard = all_list[idx]
                if shard.shard_id not in pending_ids:
                    continue
                if try_claim(shard):
                    return shard

            ordered = sorted(
                pending_list,
                key=lambda s: _h_int(self.job_id, str(self.worker_id), s.shard_id),
            )
            for shard in ordered:
                if try_claim(shard):
                    return shard
            return None

        file_keys = sorted(by_file_pending.keys())

        tried: set[str] = set()
        if previous is not None:
            fk_prev = previous.file_key
            if fk_prev in by_file_pending:
                tried.add(fk_prev)
                out = _try_file(fk_prev, previous)
                if out is not None:
                    return out

        file_keys_sorted = sorted(
            (fk for fk in file_keys if fk not in tried),
            key=lambda fk: _h_int(self.job_id, str(self.worker_id), fk),
        )
        for fk in file_keys_sorted:
            out = _try_file(fk, None)
            if out is not None:
                return out

        return None


def _runtime_lease_seconds() -> int:
    raw = os.environ.get("REFINER_RUNTIME_LEASE_SECONDS", "")
    if not raw:
        return 10 * 60
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid REFINER_RUNTIME_LEASE_SECONDS={raw!r}; expected integer"
        ) from exc
    if value <= 0:
        raise ValueError("REFINER_RUNTIME_LEASE_SECONDS must be > 0")
    return value


class FileRuntimeLifecycle:
    """Filesystem-backed runtime shard lifecycle.

    Layout (under `<workdir>/runs/<job_id>/lifecycle/`):
      - pending/<pathhash>__<start>__<end>__<shardid>.json
      - leased/<same>__w<workerid>.json          (mtime is lease freshness)
      - done/<same>.json
      - failed/<same>.json (+ optional `<same>.error`)
    """

    def __init__(
        self,
        *,
        job_id: str,
        stage_index: int = 0,
        worker_id: int | str | None,
        workdir: str | None = None,
        lease_seconds: int | None = None,
    ):
        if not job_id:
            raise ValueError("job_id must be non-empty")
        if stage_index < 0:
            raise ValueError("stage_index must be >= 0")
        self.job_id = str(job_id)
        self.stage_index = int(stage_index)
        self.worker_id = str(worker_id) if worker_id is not None else None
        self.workdir = resolve_workdir(workdir)
        self.lease_seconds = lease_seconds or _runtime_lease_seconds()

        self._root = (
            Path(self.workdir)
            / "runs"
            / self.job_id
            / "lifecycle"
            / f"stage-{self.stage_index}"
        )
        self._pending_dir = self._root / "pending"
        self._leased_dir = self._root / "leased"
        self._done_dir = self._root / "done"
        self._failed_dir = self._root / "failed"

        self._root.mkdir(parents=True, exist_ok=True)
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._leased_dir.mkdir(parents=True, exist_ok=True)
        self._done_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)

    def _require_worker_id(self) -> int:
        if self.worker_id is None:
            raise ValueError("worker_id is required for this operation")
        try:
            return int(self.worker_id)
        except ValueError as exc:
            raise ValueError("file runtime worker_id must be an integer") from exc

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        for directory in (
            self._pending_dir,
            self._leased_dir,
            self._done_dir,
            self._failed_dir,
        ):
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                except Exception:
                    pass

        for shard in shards:
            payload = {
                "path": shard.path,
                "start": int(shard.start),
                "end": int(shard.end),
            }
            (self._pending_dir / shard.pending_filename()).write_text(
                json.dumps(payload, sort_keys=True)
            )

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except Exception:
            pass

    @staticmethod
    def _safe_replace(src: str, dst: str) -> bool:
        try:
            os.replace(src, dst)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    @staticmethod
    def _key_from_filename(name: str) -> ClaimPolicy._ShardKey | None:
        try:
            pathhash, start, end, shard_id, worker_id = parse_shard_filename(name)
        except Exception:
            return None
        if worker_id is not None:
            return None
        return ClaimPolicy._ShardKey(
            file_key=pathhash, start=int(start), end=int(end), shard_id=shard_id
        )

    @staticmethod
    def _pending_name_from_key(key: ClaimPolicy._ShardKey) -> str:
        return format_pending_filename(
            pathhash=key.file_key, start=key.start, end=key.end, shard_id=key.shard_id
        )

    @staticmethod
    def _leased_name_from_key(key: ClaimPolicy._ShardKey, worker_id: int) -> str:
        return format_leased_filename(
            pathhash=key.file_key,
            start=key.start,
            end=key.end,
            shard_id=key.shard_id,
            worker_id=worker_id,
        )

    def claim(self, previous: Shard | None = None) -> Shard | None:
        worker_id = self._require_worker_id()
        now = int(time.time())

        done_names = {path.name for path in self._done_dir.iterdir() if path.is_file()}
        failed_names = {
            path.name
            for path in self._failed_dir.iterdir()
            if path.is_file() and not path.name.endswith(".error")
        }

        leased_names_effective: set[str] = set()
        for entry in os.scandir(self._leased_dir):
            if not entry.is_file():
                continue
            name = entry.name
            try:
                st = entry.stat()
            except FileNotFoundError:
                continue
            except Exception:
                leased_names_effective.add(name)
                continue

            age = now - int(st.st_mtime)
            if age <= self.lease_seconds:
                leased_names_effective.add(name)
                continue

            try:
                base = strip_worker_suffix(name)
            except Exception:
                self._safe_unlink(entry.path)
                continue

            if base in done_names or base in failed_names:
                self._safe_unlink(entry.path)
                continue

            dst = self._pending_dir / base
            if dst.exists():
                self._safe_unlink(entry.path)
                continue
            if not self._safe_replace(entry.path, str(dst)):
                self._safe_unlink(entry.path)

        pending_names = [
            path.name for path in self._pending_dir.iterdir() if path.is_file()
        ]

        pending_ids: set[str] = set()
        all_keys: set[ClaimPolicy._ShardKey] = set()

        for name in pending_names:
            key = self._key_from_filename(name)
            if key is None:
                continue
            pending_ids.add(key.shard_id)
            all_keys.add(key)

        for name in done_names | failed_names:
            key = self._key_from_filename(name)
            if key is not None:
                all_keys.add(key)

        for name in leased_names_effective:
            try:
                base = strip_worker_suffix(name)
            except Exception:
                continue
            key = self._key_from_filename(base)
            if key is not None:
                all_keys.add(key)

        policy = ClaimPolicy(job_id=self.job_id, worker_id=worker_id)

        def _try_claim(key: ClaimPolicy._ShardKey) -> bool:
            base = self._pending_name_from_key(key)
            src = self._pending_dir / base
            dst = self._leased_dir / self._leased_name_from_key(key, worker_id)
            if dst.exists():
                return False
            return self._safe_replace(str(src), str(dst))

        picked = policy.claim_key(
            previous=previous,
            all_keys=all_keys,
            pending_ids=pending_ids,
            try_claim=_try_claim,
        )
        if picked is None:
            return None

        leased_name = self._leased_name_from_key(picked, worker_id)
        leased_path = self._leased_dir / leased_name
        payload = json.loads(leased_path.read_text())
        return Shard(
            path=payload["path"], start=int(payload["start"]), end=int(payload["end"])
        )

    def heartbeat(self, shards: Iterable[Shard]) -> None:
        worker_id = self._require_worker_id()
        now = int(time.time())
        for shard in shards:
            lease_path = self._leased_dir / shard.leased_filename(worker_id)
            try:
                os.utime(lease_path, (now, now))
            except Exception:
                pass

    def complete(self, shard: Shard) -> None:
        worker_id = self._require_worker_id()
        leased_path = self._leased_dir / shard.leased_filename(worker_id)
        done_path = self._done_dir / shard.pending_filename()
        try:
            os.replace(leased_path, done_path)
        except Exception:
            pass

    def fail(self, shard: Shard, error: str | None = None) -> None:
        worker_id = self._require_worker_id()
        leased_path = self._leased_dir / shard.leased_filename(worker_id)
        failed_base = shard.pending_filename()
        failed_path = self._failed_dir / failed_base
        try:
            os.replace(leased_path, failed_path)
        except Exception:
            pass
        if error:
            try:
                (self._failed_dir / f"{failed_base}.error").write_text(error)
            except Exception:
                pass
