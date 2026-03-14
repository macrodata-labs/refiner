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
    ShardPart,
    format_leased_filename,
    format_pending_filename,
    parse_shard_filename,
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
        start_key: str | None
        end_key: str | None
        global_ordinal: int | None
        shard_id: str

    def claim_key(
        self,
        *,
        previous: Shard | None,
        all_keys: Iterable[_ShardKey],
        pending_ids: set[str],
        try_claim: Callable[[_ShardKey], bool],
    ) -> _ShardKey | None:
        by_key_all: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)
        by_key_pending: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)

        for key in all_keys:
            if key.start_key is None:
                continue
            by_key_all[key.start_key].append(key)
        for start_key in by_key_all:
            by_key_all[start_key].sort(
                key=lambda key: (
                    key.global_ordinal is None,
                    key.global_ordinal,
                    key.shard_id,
                )
            )

        for start_key, all_list in by_key_all.items():
            pending_for_key = [key for key in all_list if key.shard_id in pending_ids]
            if not pending_for_key:
                continue
            by_key_pending[start_key] = pending_for_key

        if not by_key_pending:
            return None

        # Claim order:
        # 1. exact next global ordinal after the previous shard
        # 2. block-based spreading in the previous shard's end locality
        # 3. block-based spreading in other localities
        # 4. greedy claim in the previous shard's end locality
        # 5. greedy claim anywhere
        #
        # This preserves sequential reads when possible, but still spreads
        # workers across blocks so they do not alternate every shard.
        def _try_exact_next(previous_ordinal: int) -> ClaimPolicy._ShardKey | None:
            for pending_list in by_key_pending.values():
                for candidate in pending_list:
                    if (
                        candidate.global_ordinal is not None
                        and candidate.global_ordinal == previous_ordinal + 1
                        and try_claim(candidate)
                    ):
                        return candidate
            return None

        def _try_blocks(start_keys: Iterable[str]) -> ClaimPolicy._ShardKey | None:
            for start_key in start_keys:
                pending_list = by_key_pending.get(start_key, [])
                if not pending_list:
                    continue
                total = len(by_key_all[start_key])
                if total < ClaimPolicy.BLOCK_SIZE * 2:
                    continue
                num_blocks = (
                    total + ClaimPolicy.BLOCK_SIZE - 1
                ) // ClaimPolicy.BLOCK_SIZE
                offset = _h_int(self.job_id, str(self.worker_id), start_key) % max(
                    1, num_blocks
                )
                for block_index in range(num_blocks):
                    target_ordinal = (
                        (offset + block_index) % num_blocks
                    ) * ClaimPolicy.BLOCK_SIZE
                    for candidate in pending_list:
                        if candidate.global_ordinal == target_ordinal and try_claim(
                            candidate
                        ):
                            return candidate
            return None

        def _try_greedy(start_keys: Iterable[str]) -> ClaimPolicy._ShardKey | None:
            for start_key in start_keys:
                for candidate in by_key_pending.get(start_key, []):
                    if try_claim(candidate):
                        return candidate
            return None

        tried: set[str] = set()
        if previous is not None:
            previous_key = previous.end_key or previous.start_key
            if previous.global_ordinal is not None:
                picked = _try_exact_next(previous.global_ordinal)
                if picked is not None:
                    return picked
            if previous_key and previous_key in by_key_pending:
                tried.add(previous_key)
                picked = _try_blocks((previous_key,))
                if picked is not None:
                    return picked

        start_keys = sorted(
            (start_key for start_key in by_key_pending if start_key not in tried),
            key=lambda start_key: _h_int(self.job_id, str(self.worker_id), start_key),
        )

        picked = _try_blocks(start_keys)
        if picked is not None:
            return picked

        if tried:
            picked = _try_greedy(tried)
            if picked is not None:
                return picked

        return _try_greedy(start_keys)


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
      - pending/<shardid>.json
      - leased/<shardid>__w<workerid>.json
      - done/<shardid>.json
      - failed/<shardid>.json (+ optional `<same>.error`)
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
            payload = shard.to_dict()
            (self._pending_dir / shard.pending_filename()).write_text(
                json.dumps(payload, sort_keys=True)
            )

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _safe_replace(src: str, dst: str) -> bool:
        try:
            os.replace(src, dst)
            return True
        except OSError:
            return False

    @staticmethod
    def _key_from_payload(payload: dict[str, object]) -> ClaimPolicy._ShardKey | None:
        try:
            shard_id = str(payload["shard_id"])
        except Exception:
            return None
        start_key = payload.get("start_key")
        end_key = payload.get("end_key")
        global_ordinal = payload.get("global_ordinal")
        return ClaimPolicy._ShardKey(
            start_key=start_key if isinstance(start_key, str) else None,
            end_key=end_key if isinstance(end_key, str) else None,
            global_ordinal=global_ordinal if isinstance(global_ordinal, int) else None,
            shard_id=shard_id,
        )

    @staticmethod
    def _pending_name_from_key(key: ClaimPolicy._ShardKey) -> str:
        return format_pending_filename(shard_id=key.shard_id)

    @staticmethod
    def _leased_name_from_key(key: ClaimPolicy._ShardKey, worker_id: int) -> str:
        return format_leased_filename(shard_id=key.shard_id, worker_id=worker_id)

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
                base = format_pending_filename(shard_id=parse_shard_filename(name)[0])
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
            payload = json.loads((self._pending_dir / name).read_text())
            key = self._key_from_payload(payload)
            if key is None:
                continue
            pending_ids.add(key.shard_id)
            all_keys.add(key)

        for name in done_names | failed_names:
            directory = self._done_dir if name in done_names else self._failed_dir
            key = self._key_from_payload(json.loads((directory / name).read_text()))
            if key is not None:
                all_keys.add(key)

        for name in leased_names_effective:
            try:
                parse_shard_filename(name)
            except Exception:
                continue
            key = self._key_from_payload(
                json.loads((self._leased_dir / name).read_text())
            )
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
        descriptor = payload.get("descriptor") or {
            "parts": [
                {
                    "path": payload["path"],
                    "start": payload["start"],
                    "end": payload["end"],
                    "source_index": payload.get("source_index", 0),
                    "unit": payload.get("unit", "bytes"),
                }
            ]
        }
        return Shard(
            path=payload["path"],
            start=int(payload["start"]),
            end=int(payload["end"]),
            parts=tuple(
                ShardPart(
                    path=part["path"],
                    start=int(part["start"]),
                    end=int(part["end"]),
                    source_index=int(part.get("source_index", 0)),
                    unit=str(part.get("unit", "bytes")),
                )
                for part in descriptor["parts"]
            ),
            source_index=int(payload.get("source_index", 0)),
            unit=str(payload.get("unit", "bytes")),
            global_ordinal=payload.get("global_ordinal")
            if isinstance(payload.get("global_ordinal"), int)
            else None,
            start_key=payload.get("start_key")
            if isinstance(payload.get("start_key"), str)
            else None,
            end_key=payload.get("end_key")
            if isinstance(payload.get("end_key"), str)
            else None,
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
