from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .shard import Shard


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
        file_key: str  # pathhash
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
        """Pick and claim a shard following the agreed heuristic.

        - strict consecutive in same file: start == previous.end
        - else prefer block starts (block_size=8) in pseudo-random order
        - else uniform pseudo-random pending shard in file
        - else switch files in pseudo-random order
        """
        by_file_all: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)
        by_file_pending: dict[str, list[ClaimPolicy._ShardKey]] = defaultdict(list)
        by_file_pending_by_start: dict[tuple[str, int], ClaimPolicy._ShardKey] = {}

        for k in all_keys:
            by_file_all[k.file_key].append(k)
        for fk in by_file_all:
            by_file_all[fk].sort(key=lambda r: (r.start, r.end, r.shard_id))

        for fk, all_list in by_file_all.items():
            # pending subset for this file
            p = [k for k in all_list if k.shard_id in pending_ids]
            if not p:
                continue
            by_file_pending[fk] = p
            for k in p:
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

            # 1) strict consecutive
            if prev is not None and prev.file_key == file_key:
                cand = by_file_pending_by_start.get((file_key, int(prev.end)))
                if cand is not None:
                    if try_claim(cand):
                        return cand

            # 2) block starts in pseudo-random block order, using *all* shard order to define blocks
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
                s = all_list[idx]
                # block start is only claimable if currently pending
                if s.shard_id not in pending_ids:
                    continue
                if try_claim(s):
                    return s

            # 3) uniform pseudo-random pending shard in this file (deterministic permutation)
            ordered = sorted(
                pending_list,
                key=lambda s: _h_int(self.job_id, str(self.worker_id), s.shard_id),
            )
            for s in ordered:
                if try_claim(s):
                    return s
            return None

        # File order:
        # - if previous exists, try its file first
        # - then pseudo-random order over remaining files with pending shards
        file_keys = sorted(by_file_pending.keys())

        tried: set[str] = set()
        if previous is not None:
            fk_prev = previous.file_key
            if fk_prev in by_file_pending:
                tried.add(fk_prev)
                out = _try_file(fk_prev, previous)
                if out is not None:
                    return out

        # pseudo-random file order (deterministic)
        file_keys_sorted = sorted(
            (fk for fk in file_keys if fk not in tried),
            key=lambda fk: _h_int(self.job_id, str(self.worker_id), fk),
        )
        for fk in file_keys_sorted:
            out = _try_file(fk, None)
            if out is not None:
                return out

        return None


__all__ = ["ClaimPolicy"]
