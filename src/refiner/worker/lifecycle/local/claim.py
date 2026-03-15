from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable

from refiner.pipeline.data.shard import Shard


def _h_int(*parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


class ClaimPolicy:
    BLOCK_SIZE = 8

    def __init__(self, *, job_id: str, worker_id: int):
        self.job_id = str(job_id)
        self.worker_id = int(worker_id)

    def claim(
        self,
        *,
        previous: Shard | None,
        all_shards: Iterable[Shard],
        pending_ids: set[str],
        try_claim: Callable[[Shard], bool],
    ) -> Shard | None:
        pending_all: list[Shard] = []
        by_key_all: dict[str, list[Shard]] = defaultdict(list)
        by_key_pending: dict[str, list[Shard]] = defaultdict(list)

        for shard in all_shards:
            if shard.id in pending_ids:
                pending_all.append(shard)
            if shard.start_key is not None:
                by_key_all[shard.start_key].append(shard)

        pending_all.sort(
            key=lambda shard: (
                shard.global_ordinal is None,
                shard.global_ordinal,
                shard.id,
            )
        )

        for start_key, all_for_key in by_key_all.items():
            all_for_key.sort(
                key=lambda shard: (
                    shard.global_ordinal is None,
                    shard.global_ordinal,
                    shard.id,
                )
            )
            pending_for_key = [
                shard for shard in all_for_key if shard.id in pending_ids
            ]
            if pending_for_key:
                by_key_pending[start_key] = pending_for_key

        if not pending_all:
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
        def try_exact_next(previous_ordinal: int) -> Shard | None:
            for shard in pending_all:
                if (
                    shard.global_ordinal is not None
                    and shard.global_ordinal == previous_ordinal + 1
                    and try_claim(shard)
                ):
                    return shard
            return None

        def try_blocks(start_keys: Iterable[str]) -> Shard | None:
            for start_key in start_keys:
                pending_for_key = by_key_pending.get(start_key, [])
                if not pending_for_key:
                    continue
                total = len(by_key_all[start_key])
                if total < self.BLOCK_SIZE * 2:
                    continue
                num_blocks = (total + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
                offset = _h_int(self.job_id, str(self.worker_id), start_key) % max(
                    1, num_blocks
                )
                for block_index in range(num_blocks):
                    target_ordinal = (
                        (offset + block_index) % num_blocks
                    ) * self.BLOCK_SIZE
                    for shard in pending_for_key:
                        if shard.global_ordinal == target_ordinal and try_claim(shard):
                            return shard
            return None

        def try_greedy(start_keys: Iterable[str]) -> Shard | None:
            for start_key in start_keys:
                for shard in by_key_pending.get(start_key, []):
                    if try_claim(shard):
                        return shard
            return None

        tried: set[str] = set()
        if previous is not None:
            previous_key = previous.end_key or previous.start_key
            if previous.global_ordinal is not None:
                picked = try_exact_next(previous.global_ordinal)
                if picked is not None:
                    return picked
            if previous_key and previous_key in by_key_pending:
                tried.add(previous_key)
                picked = try_blocks((previous_key,))
                if picked is not None:
                    return picked

        start_keys = sorted(
            (start_key for start_key in by_key_pending if start_key not in tried),
            key=lambda start_key: _h_int(self.job_id, str(self.worker_id), start_key),
        )

        picked = try_blocks(start_keys)
        if picked is not None:
            return picked
        if tried:
            picked = try_greedy(tried)
            if picked is not None:
                return picked
        picked = try_greedy(start_keys)
        if picked is not None:
            return picked
        for shard in pending_all:
            if try_claim(shard):
                return shard
        return None
