from __future__ import annotations

import time
from collections.abc import Iterator

import refiner as mdr

SAMPLE_PARQUET = (
    "hf://datasets/OpenResearcher/OpenResearcher-Dataset/"
    "seed_51/train-00000-of-00003.parquet"
)
SLEEP_SECONDS_PER_SHARD = 120


class SlowPerShardReader(mdr.BaseReader):
    """Reader wrapper that sleeps once before each shard is read.
    simulate longer processing
    """

    def __init__(self, inner: mdr.BaseReader, *, sleep_seconds: int):
        self.inner = inner
        self.sleep_seconds = sleep_seconds

    def list_shards(self) -> list[mdr.Shard]:
        return self.inner.list_shards()

    def read_shard(self, shard: mdr.Shard) -> Iterator[mdr.Row]:
        print(
            f"[demo] sleeping {self.sleep_seconds}s before shard "
            f"path={shard.path} start={shard.start} end={shard.end}",
            flush=True,
        )
        time.sleep(self.sleep_seconds)
        yield from self.inner.read_shard(shard)


def add_text_len(row):
    text = row.get("text")
    if text is None:
        return {"text_len": 0}
    return {"text_len": len(str(text))}


def duplicate_short_rows(row):
    # Example row expansion: duplicate rows under a small threshold.
    text_len = int(row.get("text_len", 0))
    if text_len == 0:
        return []
    if text_len < 50:
        return [row, {"text_len": text_len, "augmented": True}]
    return [row]


def main() -> None:
    base_pipeline = mdr.read_parquet(
        SAMPLE_PARQUET,
        # Keep shards small so the demo exercises multi-shard scheduling/observer updates.
        target_shard_bytes=1 * 1024 * 1024,
    )
    slow_reader = SlowPerShardReader(
        base_pipeline.source, sleep_seconds=SLEEP_SECONDS_PER_SHARD
    )
    pipeline = (
        mdr.RefinerPipeline(slow_reader)
        .map(add_text_len)
        .filter(lambda row: int(row.get("text_len", 0)) > 0)
        .flat_map(duplicate_short_rows)
    )

    print("Launching local Refiner job...")
    print(f"Input: {SAMPLE_PARQUET}")
    print("Workers: 3")
    print(f"Sleep per shard: {SLEEP_SECONDS_PER_SHARD}s")

    t0 = time.time()
    stats = pipeline.launch_local(
        name="notebook-local-demo",
        num_workers=3,
        # Optional: pin each worker to a CPU subset if desired.
        # cpus_per_worker=2,
    )
    dt = time.time() - t0

    print("\nLaunch complete")
    print(f"workers     : {stats.workers}")
    print(f"claimed     : {stats.claimed}")
    print(f"completed   : {stats.completed}")
    print(f"failed      : {stats.failed}")
    print(f"output_rows : {stats.output_rows}")
    print(f"elapsed_s   : {dt:.2f}")


if __name__ == "__main__":
    main()
