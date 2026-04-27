from __future__ import annotations

from abc import ABC
from typing import Any

from refiner.execution.tracking.shards import count_block_by_shard
from refiner.pipeline.data.block import Block, split_block_by_shard
from refiner.worker.metrics.api import log_throughput


class BaseSink(ABC):
    """Base sink interface with shard-local writes as the default behavior.

    Most sinks should implement `write_shard_block(...)` and inherit the default
    `write_block(...)` behavior. Override `write_block(...)` only for sinks that
    do not operate shard-by-shard, such as pure counting or global reduction
    sinks.
    """

    def write_block(self, block: Block) -> dict[str, int]:
        """Split a mixed block by shard and dispatch each shard-local block.

        This is the normal entry point and should usually be inherited as-is.
        """
        blocks_by_shard, counts = split_block_by_shard(block)
        for shard_id, shard_block in blocks_by_shard.items():
            self.write_shard_block(shard_id, shard_block)
            log_throughput(
                "rows_written",
                counts[shard_id],
                shard_id=shard_id,
                unit="rows",
            )
        return counts

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        """Write one already shard-local block.

        This is the main method concrete shard-writing sinks should implement.
        """
        del shard_id, block
        raise NotImplementedError

    def describe(self) -> tuple[str, str, dict[str, Any] | None] | None:
        """Return an optional sink description for planning and tracing output.

        Override this when the sink should expose structured configuration in
        pipeline descriptions.
        """
        return None

    def build_reducer(self) -> "BaseSink | None":
        """Return an optional 1-worker reducer sink for launched execution.

        Reducers run as a follow-up stage after the main writer stage. Use this
        when a sink needs a final cleanup or reduction pass once all shard-local
        writer outputs are known.
        """
        return None

    @property
    def counts_output_rows(self) -> bool:
        """Whether blocks written into this sink should count toward output_rows."""
        return True

    @property
    def requires_tabular_input(self) -> bool:
        """Whether this sink expects blocks to be converted to `Tabular` before writing."""
        return False

    def on_shard_complete(self, shard_id: str) -> None:
        """Flush or finalize state for one shard after upstream completion.

        Override this only when the sink keeps shard-local buffered state that
        should be finalized before `close()`.
        """
        del shard_id

    def close(self) -> None:
        """Finalize sink resources after all shard work is complete.

        Override this when the sink keeps process-wide resources or deferred
        state that should be flushed at the end of execution.
        """
        return None


class NullSink(BaseSink):
    """Sink that discards data and only reports shard counts."""

    def write_block(self, block: Block) -> dict[str, int]:
        return count_block_by_shard(block)


__all__ = ["BaseSink", "NullSink"]
