from __future__ import annotations

import pytest

from refiner.execution.tracking.shards import ShardDeltaTracker


def test_shard_delta_tracker_does_not_emit_on_exception() -> None:
    emitted: list[dict[str, int]] = []

    with pytest.raises(RuntimeError):
        with ShardDeltaTracker(emitted.append) as delta:
            delta.add("s1", -1)
            raise RuntimeError("boom")

    assert emitted == []
