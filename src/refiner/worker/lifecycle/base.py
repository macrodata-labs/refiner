from __future__ import annotations

from typing import Protocol

from refiner.pipeline.data.shard import Shard


class RuntimeLifecycle(Protocol):
    def claim(self, previous: Shard | None = None) -> Shard | None: ...

    def heartbeat(self, shards: list[Shard]) -> None: ...

    def complete(self, shard: Shard) -> None: ...

    def fail(self, shard: Shard, error: str | None = None) -> None: ...


__all__ = ["RuntimeLifecycle"]
