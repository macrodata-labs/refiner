from __future__ import annotations

MAX_SHARDS = 10_000


def validate_shard_count(count: int, *, source: str) -> None:
    if count > MAX_SHARDS:
        raise ValueError(
            f"{source} shard plan exceeds the {MAX_SHARDS:,}-shard limit; "
            f"planned {count:,} shards"
        )


def validate_num_shards(num_shards: int | None) -> None:
    if num_shards is not None:
        validate_shard_count(num_shards, source="Explicit")
