from __future__ import annotations

import json
from collections.abc import Iterable

from refiner.readers.base import Shard

from .base import BaseLedger
from .config import load_ledger_config_from_env, redis_url_from_env

# Import LedgerConfig for type annotations
from .backend.base import LedgerConfig


class RedisLedger(BaseLedger):
    """Redis-backed shard ledger (optional extra).

    Keys are namespaced by run_id:
      - refiner:{run_id}:shards      (hash: id -> JSON payload)
      - refiner:{run_id}:enqueued    (set: ids that were enqueued at least once)
      - refiner:{run_id}:pending     (list: ids to claim)
      - refiner:{run_id}:done        (set)
      - refiner:{run_id}:failed      (set)
      - refiner:{run_id}:lease:{id}  (string with TTL; value = worker_id)
    """

    def __init__(
        self,
        *,
        worker_id: str,
        run_id: str,
        url: str | None = None,
        config: LedgerConfig | None = None,
    ):
        cfg = config or load_ledger_config_from_env()
        super().__init__(worker_id=worker_id, run_id=run_id, config=cfg)
        url = url or redis_url_from_env()
        if not url:
            raise ValueError(
                "RedisLedger requires a redis URL (url= or REFINER_LEDGER_REDIS_URL)"
            )

        # Optional dependency.
        import redis  # type: ignore[import-not-found]

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._pfx = f"refiner:{self.run_id}"

        self._k_shards = f"{self._pfx}:shards"
        self._k_enqueued = f"{self._pfx}:enqueued"
        self._k_pending = f"{self._pfx}:pending"
        self._k_done = f"{self._pfx}:done"
        self._k_failed = f"{self._pfx}:failed"

        # Lua scripts for atomic operations.
        self._claim_script = self._r.register_script(
            r"""
local pending = KEYS[1]
local done = KEYS[2]
local failed = KEYS[3]
local lease_prefix = ARGV[1]
local worker_id = ARGV[2]
local lease_seconds = tonumber(ARGV[3])

-- try a few times to skip duplicates/stale ids
for i=1,50 do
  local id = redis.call('LPOP', pending)
  if not id then
    return nil
  end
  if redis.call('SISMEMBER', done, id) == 1 then
    -- already done; skip
  elseif redis.call('SISMEMBER', failed, id) == 1 then
    -- already failed; skip
  else
    local lk = lease_prefix .. id
    -- set lease iff absent
    local ok = redis.call('SET', lk, worker_id, 'NX', 'EX', lease_seconds)
    if ok then
      return id
    else
      -- someone else holds lease; put back to pending tail
      redis.call('RPUSH', pending, id)
      return nil
    end
  end
end
return nil
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

    def enqueue(self, shards: Iterable[Shard]) -> None:
        # SADD per shard and only RPUSH when SADD==1 (first time seen).
        shards_l = list(shards)
        pipe = self._r.pipeline(transaction=True)
        for shard in shards_l:
            payload = json.dumps(
                {
                    "id": shard.id,
                    "path": shard.path,
                    "start": int(shard.start),
                    "end": int(shard.end),
                },
                sort_keys=True,
            )
            pipe.hset(self._k_shards, shard.id, payload)
            pipe.sadd(self._k_enqueued, shard.id)
        res = pipe.execute()
        # res is [hset, sadd, hset, sadd, ...]
        pipe = self._r.pipeline(transaction=True)
        i = 0
        for shard in shards_l:
            sadd_res = res[i + 1]
            if sadd_res == 1:
                pipe.rpush(self._k_pending, shard.id)
            i += 2
        pipe.execute()

    def claim(self) -> Shard | None:
        lease_seconds = int(self.config.lease_seconds)
        shard_id = self._claim_script(
            keys=[self._k_pending, self._k_done, self._k_failed],
            args=[f"{self._pfx}:lease:", self.worker_id, str(lease_seconds)],
        )
        if not shard_id:
            return None
        payload = self._r.hget(self._k_shards, shard_id)
        if not payload:
            # best-effort: mark failed and drop
            self._r.sadd(self._k_failed, shard_id)
            return None
        s = json.loads(payload)
        return Shard(path=s["path"], start=int(s["start"]), end=int(s["end"]))

    def heartbeat(self, shard: Shard) -> None:
        lease_seconds = int(self.config.lease_seconds)
        self._heartbeat_script(
            keys=[self._lease_key(shard.id)],
            args=[self.worker_id, str(lease_seconds)],
        )

    def complete(self, shard: Shard) -> None:
        shard_id = shard.id
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(self._lease_key(shard_id))
        pipe.sadd(self._k_done, shard_id)
        pipe.execute()

    def fail(self, shard: Shard, error: str | None = None) -> None:
        shard_id = shard.id
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(self._lease_key(shard_id))
        pipe.sadd(self._k_failed, shard_id)
        if error:
            pipe.hset(f"{self._pfx}:errors", shard_id, error)
        pipe.execute()


__all__ = ["RedisLedger"]
