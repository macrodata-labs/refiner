from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

from ..config import load_ledger_config_from_env, resolve_workdir
from ..policy import ClaimPolicy
from ..shard import (
    Shard,
    format_leased_filename,
    format_pending_filename,
    parse_shard_filename,
    strip_worker_suffix,
)
from .base import BaseLedger, LedgerConfig


class FsLedger(BaseLedger):
    """Filesystem-backed shard ledger.

    Layout (under `<workdir>/runs/<run_id>/ledger/`):
      - pending/<pathhash>__<start>__<end>__<shardid>.json
      - leased/<same>__w<workerid>.json          (mtime is heartbeat freshness)
      - done/<same>.json
      - failed/<same>.json (+ optional `<same>.error`)
    """

    def __init__(
        self,
        *,
        run_id: str,
        worker_id: int | None = None,
        workdir: str | None = None,
        config: LedgerConfig | None = None,
    ):
        cfg = config or load_ledger_config_from_env()
        super().__init__(run_id=run_id, worker_id=worker_id, config=cfg)
        self.workdir = resolve_workdir(workdir)

        self._root = Path(self.workdir) / "runs" / self.run_id / "ledger"
        self._pending_dir = self._root / "pending"
        self._leased_dir = self._root / "leased"
        self._done_dir = self._root / "done"
        self._failed_dir = self._root / "failed"

        self._root.mkdir(parents=True, exist_ok=True)
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._leased_dir.mkdir(parents=True, exist_ok=True)
        self._done_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        # Overwrite run shards by clearing dirs and populating pending/.
        for d in (
            self._pending_dir,
            self._leased_dir,
            self._done_dir,
            self._failed_dir,
        ):
            for p in d.iterdir():
                if not p.is_file():
                    continue
                try:
                    p.unlink()
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
            ph, start, end, shard_id, w = parse_shard_filename(name)
        except Exception:
            return None
        if w is not None:
            return None
        return ClaimPolicy._ShardKey(
            file_key=ph, start=int(start), end=int(end), shard_id=shard_id
        )

    @staticmethod
    def _pending_name_from_key(k: ClaimPolicy._ShardKey) -> str:
        return format_pending_filename(
            pathhash=k.file_key, start=k.start, end=k.end, shard_id=k.shard_id
        )

    @staticmethod
    def _leased_name_from_key(k: ClaimPolicy._ShardKey, worker_id: int) -> str:
        return format_leased_filename(
            pathhash=k.file_key,
            start=k.start,
            end=k.end,
            shard_id=k.shard_id,
            worker_id=worker_id,
        )

    def claim(self, previous: Shard | None = None) -> Shard | None:
        self._require_worker_id()
        now = int(time.time())
        lease_seconds = int(self.config.lease_seconds)

        done_names = {p.name for p in self._done_dir.iterdir() if p.is_file()}
        failed_names = {
            p.name
            for p in self._failed_dir.iterdir()
            if p.is_file() and not p.name.endswith(".error")
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
            if age <= lease_seconds:
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

        # Build key-space view from filenames only; read JSON only after successful claim.
        pending_names = [p.name for p in self._pending_dir.iterdir() if p.is_file()]

        pending_ids: set[str] = set()
        all_keys: set[ClaimPolicy._ShardKey] = set()

        for name in pending_names:
            k = self._key_from_filename(name)
            if k is None:
                continue
            pending_ids.add(k.shard_id)
            all_keys.add(k)

        for name in done_names | failed_names:
            k = self._key_from_filename(name)
            if k is not None:
                all_keys.add(k)

        for name in leased_names_effective:
            try:
                base = strip_worker_suffix(name)
            except Exception:
                continue
            k = self._key_from_filename(base)
            if k is not None:
                all_keys.add(k)

        policy = ClaimPolicy(run_id=self.run_id, worker_id=self._require_worker_id())

        def _try_claim(k: ClaimPolicy._ShardKey) -> bool:
            base = self._pending_name_from_key(k)
            src = self._pending_dir / base
            dst = self._leased_dir / self._leased_name_from_key(
                k, self._require_worker_id()
            )
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

        # Read exactly one file (claimed) to get the full path.
        leased_name = self._leased_name_from_key(picked, self._require_worker_id())
        leased_path = self._leased_dir / leased_name
        payload = json.loads(leased_path.read_text())
        return Shard(
            path=payload["path"], start=int(payload["start"]), end=int(payload["end"])
        )

    def heartbeat(self, shard: Shard) -> None:
        self._require_worker_id()
        now = int(time.time())
        lease_path = self._leased_dir / shard.leased_filename(self._require_worker_id())
        try:
            os.utime(lease_path, (now, now))
        except Exception:
            pass

    def complete(self, shard: Shard) -> None:
        self._require_worker_id()
        leased_path = self._leased_dir / shard.leased_filename(
            self._require_worker_id()
        )
        done_path = self._done_dir / shard.pending_filename()
        try:
            os.replace(leased_path, done_path)
        except Exception:
            pass

    def fail(self, shard: Shard, error: str | None = None) -> None:
        self._require_worker_id()
        leased_path = self._leased_dir / shard.leased_filename(
            self._require_worker_id()
        )
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


__all__ = ["FsLedger"]
