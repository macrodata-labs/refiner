from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator

import refiner as mdr
from refiner.pipeline import Row, Shard
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sources.base import BaseSource

DEFAULT_DELAYS_SECONDS = (5, 10, 15, 20, 25, 30)


class DelayedShardSource(BaseSource):
    """Synthetic source for exercising the local launcher CLI."""

    name = "delayed_shard_demo"

    def __init__(
        self,
        *,
        delays_seconds: tuple[float, ...],
        rows_per_shard: int,
    ) -> None:
        self._delays_seconds = delays_seconds
        self._rows_per_shard = rows_per_shard

    def describe(self) -> dict[str, object]:
        return {
            "kind": "synthetic-delayed-shards",
            "num_shards": len(self._delays_seconds),
            "rows_per_shard": self._rows_per_shard,
            "delays_seconds": list(self._delays_seconds),
        }

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_row_range(start=index, end=index + 1, global_ordinal=index)
            for index in range(len(self._delays_seconds))
        ]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("DelayedShardSource requires row-range shards")

        shard_index = int(descriptor.start)
        delay_seconds = float(self._delays_seconds[shard_index])
        sleep_budget = delay_seconds

        mdr.logger.info(
            "demo shard start shard={} ordinal={} delay_s={:.1f}",
            shard_index + 1,
            shard.global_ordinal,
            delay_seconds,
        )
        halfway_logged = False
        while sleep_budget > 0:
            step = min(1.0, sleep_budget)
            time.sleep(step)
            sleep_budget -= step
            elapsed = delay_seconds - sleep_budget
            if (
                not halfway_logged
                and delay_seconds >= 4.0
                and elapsed >= delay_seconds / 2
            ):
                halfway_logged = True
                mdr.logger.info(
                    "demo shard progress shard={} elapsed_s={:.1f}/{:.1f}",
                    shard_index + 1,
                    elapsed,
                    delay_seconds,
                )

        for row_index in range(self._rows_per_shard):
            yield DictRow(
                data={
                    "shard_index": shard_index + 1,
                    "row_index": row_index + 1,
                    "delay_seconds": delay_seconds,
                    "worker_note": f"shard-{shard_index + 1}",
                }
            )

        mdr.logger.info(
            "demo shard complete shard={} rows={} delay_s={:.1f}",
            shard_index + 1,
            self._rows_per_shard,
            delay_seconds,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise local launcher UI with staggered shard completion.",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rows-per-shard", type=int, default=2)
    parser.add_argument(
        "--delay-scale",
        type=float,
        default=float(os.environ.get("MDR_DEMO_DELAY_SCALE", "1.0")),
        help="Multiply the default 5s..30s delays by this factor.",
    )
    parser.add_argument(
        "--rundir",
        default=None,
        help="Optional local rundir for resume testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be > 0")
    if args.delay_scale <= 0:
        raise ValueError("--delay-scale must be > 0")

    delays_seconds = tuple(delay * args.delay_scale for delay in DEFAULT_DELAYS_SECONDS)
    pipeline = mdr.from_source(
        DelayedShardSource(
            delays_seconds=delays_seconds,
            rows_per_shard=args.rows_per_shard,
        )
    ).map(
        lambda row: {
            **row,
            "summary": (
                f"shard={row['shard_index']} "
                f"row={row['row_index']} "
                f"delay_s={row['delay_seconds']:.1f}"
            ),
        }
    )

    print("Launching local Refiner job...")
    print(f"Workers: {args.workers}")
    print(f"Shard delays (s): {', '.join(f'{delay:.1f}' for delay in delays_seconds)}")

    started_at = time.time()
    stats = pipeline.launch_local(
        name="local-cli-demo",
        num_workers=args.workers,
        rundir=args.rundir,
    )
    elapsed_seconds = time.time() - started_at

    print("stats:", stats)
    print(f"elapsed_s: {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
