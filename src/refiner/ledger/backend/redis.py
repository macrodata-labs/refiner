from __future__ import annotations

from collections.abc import Iterable

from ..config import load_ledger_config_from_env, redis_url_from_env
from ..policy import ClaimPolicy
from ..shard import Shard, format_pending_filename, parse_shard_filename, path_hash
from .base import BaseLedger, LedgerConfig


class RedisLedger(BaseLedger):
    """Redis-backed shard ledger (optional extra)."""

    def __init__(
        self,
        *,
        run_id: str,
        worker_id: int | None = None,
        url: str | None = None,
        config: LedgerConfig | None = None,
    ):
        cfg = config or load_ledger_config_from_env()
        super().__init__(run_id=run_id, worker_id=worker_id, config=cfg)
        url = url or redis_url_from_env()
        if not url:
            raise ValueError(
                "RedisLedger requires a redis URL (url= or REFINER_LEDGER_REDIS_URL)"
            )

        import redis  # type: ignore[import-not-found]

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._pfx = f"refiner:{self.run_id}"

        self._k_files = f"{self._pfx}:files"
        self._k_id2path = f"{self._pfx}:id2path"
        self._k_id2member = f"{self._pfx}:id2member"
        self._k_done = f"{self._pfx}:done"
        self._k_failed = f"{self._pfx}:failed"
        self._k_errors = f"{self._pfx}:errors"

        self._try_claim_script = self._r.register_script(
            r"""
local pending_zset = KEYS[1]
local done_set = KEYS[2]
local failed_set = KEYS[3]
local id2member = KEYS[4]

local member = ARGV[1]
local shardid = ARGV[2]
local lease_key = ARGV[3]
local worker_id = ARGV[4]
local lease_seconds = tonumber(ARGV[5])

if redis.call('ZSCORE', pending_zset, member) == false then
  return 0
end
if redis.call('SISMEMBER', done_set, shardid) == 1 then
  return 0
end
if redis.call('SISMEMBER', failed_set, shardid) == 1 then
  return 0
end
local ok = redis.call('SET', lease_key, worker_id, 'NX', 'EX', lease_seconds)
if not ok then
  return 0
end
redis.call('ZREM', pending_zset, member)
redis.call('HSET', id2member, shardid, member)
return 1
"""
        )
        self._heartbeat_script = self._r.register_script(
            r"""
local lease_key = KEYS[1]
local worker_id = ARGV[1]
local lease_seconds = tonumber(ARGV[2])
local v = redis.call('GET', lease_key)
if not v then
  return 0
end
if v ~= worker_id then
  return 0
end
redis.call('EXPIRE', lease_key, lease_seconds)
return 1
"""
        )

    def _lease_key(self, shard_id: str) -> str:
        return f"{self._pfx}:lease:{shard_id}"

    def _k_file_all(self, file_key: str) -> str:
        return f"{self._pfx}:file:{file_key}:all"

    def _k_file_pending(self, file_key: str) -> str:
        return f"{self._pfx}:file:{file_key}:pending"

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        shards_l = list(shards)
        for k in self._r.scan_iter(match=f"{self._pfx}:*"):
            self._r.delete(k)

        pipe = self._r.pipeline(transaction=True)
        for shard in shards_l:
            file_key = path_hash(shard.path)
            member = format_pending_filename(
                pathhash=file_key, start=shard.start, end=shard.end, shard_id=shard.id
            )
            pipe.sadd(self._k_files, file_key)
            pipe.hset(self._k_id2path, shard.id, shard.path)
            pipe.hset(self._k_id2member, shard.id, member)
            pipe.zadd(self._k_file_all(file_key), {member: int(shard.start)})
            pipe.zadd(self._k_file_pending(file_key), {member: int(shard.start)})
        pipe.execute()

    def claim(self, previous: Shard | None = None) -> Shard | None:
        worker_id_str = self._require_worker_id_str()
        lease_seconds = int(self.config.lease_seconds)

        file_keys = sorted(self._r.smembers(self._k_files))
        if not file_keys:
            return None

        p = self._r.pipeline(transaction=False)
        for fk in file_keys:
            p.zrange(self._k_file_all(fk), 0, -1)
        for fk in file_keys:
            p.zrange(self._k_file_pending(fk), 0, -1)
        res = p.execute()

        all_by_file: dict[str, list[str]] = {}
        pending_by_file: dict[str, list[str]] = {}
        n = len(file_keys)
        for i, fk in enumerate(file_keys):
            all_by_file[fk] = list(res[i] or [])
        for i, fk in enumerate(file_keys):
            pending_by_file[fk] = list(res[n + i] or [])

        pending_ids: set[str] = set()
        all_keys: set[ClaimPolicy._ShardKey] = set()

        for members in all_by_file.values():
            for m in members:
                try:
                    ph, start, end, shard_id, w = parse_shard_filename(m)
                except Exception:
                    continue
                if w is not None:
                    continue
                all_keys.add(
                    ClaimPolicy._ShardKey(
                        file_key=ph, start=int(start), end=int(end), shard_id=shard_id
                    )
                )
        for members in pending_by_file.values():
            for m in members:
                try:
                    ph, start, end, shard_id, w = parse_shard_filename(m)
                except Exception:
                    continue
                if w is not None:
                    continue
                pending_ids.add(shard_id)

        policy = ClaimPolicy(run_id=self.run_id, worker_id=self._require_worker_id())

        def _try_claim(k: ClaimPolicy._ShardKey) -> bool:
            member = format_pending_filename(
                pathhash=k.file_key, start=k.start, end=k.end, shard_id=k.shard_id
            )
            ok = self._try_claim_script(
                keys=[
                    self._k_file_pending(k.file_key),
                    self._k_done,
                    self._k_failed,
                    self._k_id2member,
                ],
                args=[
                    member,
                    k.shard_id,
                    self._lease_key(k.shard_id),
                    worker_id_str,
                    str(lease_seconds),
                ],
            )
            return bool(ok)

        picked = policy.claim_key(
            previous=previous,
            all_keys=all_keys,
            pending_ids=pending_ids,
            try_claim=_try_claim,
        )
        if picked is None:
            return None
        path = self._r.hget(self._k_id2path, picked.shard_id)
        if not path:
            self._r.sadd(self._k_failed, picked.shard_id)
            return None
        return Shard(path=path, start=picked.start, end=picked.end)

    def heartbeat(self, shard: Shard) -> None:
        worker_id = self._require_worker_id_str()
        lease_seconds = int(self.config.lease_seconds)
        self._heartbeat_script(
            keys=[self._lease_key(shard.id)],
            args=[worker_id, str(lease_seconds)],
        )

    def complete(self, shard: Shard) -> None:
        self._require_worker_id()
        shard_id = shard.id
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(self._lease_key(shard_id))
        pipe.sadd(self._k_done, shard_id)
        pipe.execute()

    def fail(self, shard: Shard, error: str | None = None) -> None:
        self._require_worker_id()
        shard_id = shard.id
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(self._lease_key(shard_id))
        pipe.sadd(self._k_failed, shard_id)
        if error:
            pipe.hset(self._k_errors, shard_id, error)
        pipe.execute()


__all__ = ["RedisLedger"]
