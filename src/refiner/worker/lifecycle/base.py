from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from refiner.pipeline.data.shard import Shard

if TYPE_CHECKING:
    from refiner.platform.client.models import FinalizedShardWorker


class RuntimeLifecycle(Protocol):
    def claim(self, previous: Shard | None = None) -> Shard | None: ...

    def heartbeat(self, shards: list[Shard]) -> None: ...

    def complete(self, shard: Shard) -> None: ...

    def fail(self, shard: Shard, error: str | None = None) -> None: ...

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]: ...


__all__ = ["RuntimeLifecycle"]
