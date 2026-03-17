from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

from refiner.pipeline.data.shard import Shard
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.context import RunHandle
from refiner.worker.lifecycle.local.claim import ClaimPolicy
from refiner.worker.lifecycle.local.files import leased_filename
from refiner.worker.lifecycle.local.files import parse_shard_filename
from refiner.worker.lifecycle.local.files import pending_filename
from refiner.worker.lifecycle.local.files import safe_replace
from refiner.worker.lifecycle.local.files import safe_unlink
from refiner.worker.workdir import resolve_workdir


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


def _runtime_max_attempts() -> int:
    raw = os.environ.get("SHARD_MAX_ATTEMPTS", "")
    if not raw:
        return 3
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid SHARD_MAX_ATTEMPTS={raw!r}; expected integer"
        ) from exc
    if value <= 0:
        raise ValueError("SHARD_MAX_ATTEMPTS must be > 0")
    return value


class LocalRuntimeLifecycle:
    """Filesystem-backed runtime shard lifecycle.

    Layout (under `<workdir>/runs/<job_id>/lifecycle/`):
      - pending/<shardid>.json
      - leased/<shardid>__w<workerid>.json
      - done/<shardid>.json
      - failed/<shardid>.json (+ optional `<same>.error`)
    """

    def __init__(
        self,
        *,
        run: RunHandle,
        workdir: str | None = None,
        lease_seconds: int | None = None,
    ):
        self.job_id = str(run.job_id)
        self.stage_index = int(run.stage_index)
        self.worker_id = str(run.worker_id) if run.worker_id is not None else None
        self.workdir = resolve_workdir(workdir)
        self.lease_seconds = lease_seconds or _runtime_lease_seconds()
        self.max_attempts = _runtime_max_attempts()

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

    def _require_worker_id(self) -> str:
        if self.worker_id is None:
            raise ValueError("worker_id is required for this operation")
        return self.worker_id

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        for directory in (
            self._pending_dir,
            self._leased_dir,
            self._done_dir,
            self._failed_dir,
        ):
            for path in directory.iterdir():
                if path.is_file():
                    try:
                        path.unlink()
                    except Exception:
                        pass

        for shard in shards:
            (self._pending_dir / pending_filename(shard.id)).write_text(
                json.dumps(shard.to_dict(), sort_keys=True)
            )

    def _reclaim_stale_leases(
        self, *, done_names: set[str], failed_names: set[str], now: int
    ) -> set[str]:
        leased_names: set[str] = set()
        for entry in os.scandir(self._leased_dir):
            if not entry.is_file():
                continue
            name = entry.name
            try:
                st = entry.stat()
            except FileNotFoundError:
                continue
            except Exception:
                leased_names.add(name)
                continue

            if now - int(st.st_mtime) <= self.lease_seconds:
                leased_names.add(name)
                continue

            try:
                base = pending_filename(parse_shard_filename(name)[0])
            except Exception:
                safe_unlink(entry.path)
                continue

            if base in done_names or base in failed_names:
                safe_unlink(entry.path)
                continue

            dst = self._pending_dir / base
            if dst.exists():
                safe_unlink(entry.path)
                continue
            if not safe_replace(entry.path, str(dst)):
                safe_unlink(entry.path)
        return leased_names

    @staticmethod
    def _load_shard(path: Path) -> Shard:
        return Shard.from_dict(json.loads(path.read_text()))

    @staticmethod
    def _attempt_count(path: Path) -> int:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        raw = payload.get("_attempt_count", 0)
        if isinstance(raw, int) and raw >= 0:
            return raw
        return 0

    def claim(self, previous: Shard | None = None) -> Shard | None:
        worker_id = self._require_worker_id()
        now = int(time.time())

        done_names = {path.name for path in self._done_dir.iterdir() if path.is_file()}
        failed_names = {
            path.name
            for path in self._failed_dir.iterdir()
            if path.is_file() and not path.name.endswith(".error")
        }
        leased_names = self._reclaim_stale_leases(
            done_names=done_names, failed_names=failed_names, now=now
        )

        pending_ids: set[str] = set()
        all_shards: dict[str, Shard] = {}

        for path in self._pending_dir.iterdir():
            if not path.is_file():
                continue
            shard = self._load_shard(path)
            pending_ids.add(shard.id)
            all_shards[shard.id] = shard

        for name in done_names | failed_names:
            directory = self._done_dir if name in done_names else self._failed_dir
            shard = self._load_shard(directory / name)
            all_shards[shard.id] = shard

        for name in leased_names:
            try:
                parse_shard_filename(name)
            except Exception:
                continue
            shard = self._load_shard(self._leased_dir / name)
            all_shards[shard.id] = shard

        def try_claim(shard: Shard) -> bool:
            src = self._pending_dir / pending_filename(shard.id)
            dst = self._leased_dir / leased_filename(shard.id, worker_id)
            if dst.exists() or not safe_replace(str(src), str(dst)):
                return False
            try:
                payload = json.loads(dst.read_text())
                if not isinstance(payload, dict):
                    raise ValueError("shard payload must be an object")
                payload["_attempt_count"] = self._attempt_count(dst) + 1
                dst.write_text(json.dumps(payload, sort_keys=True))
            except Exception:
                safe_replace(str(dst), str(src))
                return False
            return True

        picked = ClaimPolicy(job_id=self.job_id, worker_id=worker_id).claim(
            previous=previous,
            all_shards=all_shards.values(),
            pending_ids=pending_ids,
            try_claim=try_claim,
        )
        if picked is None:
            return None
        return self._load_shard(
            self._leased_dir / leased_filename(picked.id, worker_id)
        )

    def heartbeat(self, shards: Iterable[Shard]) -> None:
        worker_id = self._require_worker_id()
        now = int(time.time())
        for shard in shards:
            lease_path = self._leased_dir / leased_filename(shard.id, worker_id)
            try:
                os.utime(lease_path, (now, now))
            except Exception:
                pass

    def complete(self, shard: Shard) -> None:
        worker_id = self._require_worker_id()
        leased_path = self._leased_dir / leased_filename(shard.id, worker_id)
        done_path = self._done_dir / pending_filename(shard.id)
        try:
            payload = json.loads(leased_path.read_text())
            payload["_finalized_worker_id"] = str(worker_id)
            done_path.write_text(json.dumps(payload, sort_keys=True))
            leased_path.unlink()
        except Exception:
            pass

    def fail(self, shard: Shard, error: str | None = None) -> None:
        worker_id = self._require_worker_id()
        leased_path = self._leased_dir / leased_filename(shard.id, worker_id)
        if self._attempt_count(leased_path) < self.max_attempts:
            pending_path = self._pending_dir / pending_filename(shard.id)
            try:
                os.replace(leased_path, pending_path)
            except Exception:
                pass
            return
        failed_base = pending_filename(shard.id)
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

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        done_dir = (
            self._done_dir
            if stage_index is None or stage_index == self.stage_index
            else Path(self.workdir)
            / "runs"
            / self.job_id
            / "lifecycle"
            / f"stage-{stage_index}"
            / "done"
        )
        out: list[FinalizedShardWorker] = []
        if not done_dir.exists():
            return out
        for path in done_dir.iterdir():
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text())
                shard_id = str(payload["shard_id"])
                worker_id = str(payload["_finalized_worker_id"])
            except Exception:
                continue
            out.append(FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id))
        out.sort(key=lambda row: row.shard_id)
        return out
