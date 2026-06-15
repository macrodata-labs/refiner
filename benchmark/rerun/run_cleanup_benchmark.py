from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

from refiner.pipeline.sinks.reducer.file import _cleanup_default_root_entries

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_REGEX_PATTERN = re.compile(
    r"^(?P<shard_id>[0-9a-f]{12})__w(?P<worker_id>[0-9a-f]{12})$"
)


@dataclass(slots=True)
class CaseResult:
    mode: str
    wall_time_ns: int
    entries: int
    deleted_entries: int

    @property
    def wall_time_s(self) -> float:
        return self.wall_time_ns / 1_000_000_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the default-root RRD cleanup matcher."
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--entries", type=int, default=10000)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-token")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_entries(entries: int) -> tuple[list[str], set[tuple[str, str]]]:
    root_entries = [f"{index:012x}__w{(index + 1):012x}" for index in range(entries)]
    keep_pairs = {
        (f"{index:012x}", f"{(index + 1):012x}") for index in range(0, entries, 2)
    }
    return root_entries, keep_pairs


def _regex_cleanup(
    root_entries: list[str],
    keep_pairs: set[tuple[str, str]],
) -> set[str]:
    paths_to_delete: set[str] = set()
    for rel_path in root_entries:
        match = _REGEX_PATTERN.fullmatch(rel_path)
        if match is None:
            continue
        if (match.group("shard_id"), match.group("worker_id")) not in keep_pairs:
            paths_to_delete.add(rel_path)
    return paths_to_delete


def _benchmark(
    *,
    mode: str,
    fn: Callable[[list[str], set[tuple[str, str]]], set[str]],
    root_entries: list[str],
    keep_pairs: set[tuple[str, str]],
    iterations: int,
) -> CaseResult:
    start = perf_counter_ns()
    deleted_entries = 0
    for _ in range(iterations):
        deleted_entries = len(fn(root_entries, keep_pairs))
    return CaseResult(
        mode=mode,
        wall_time_ns=perf_counter_ns() - start,
        entries=len(root_entries),
        deleted_entries=deleted_entries,
    )


def main() -> int:
    args = _parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.entries < 1:
        raise ValueError("--entries must be >= 1")

    run_token = (
        args.run_token
        or f"cleanup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    artifacts_dir = args.artifacts_dir / run_token
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    root_entries, keep_pairs = _generate_entries(args.entries)
    results = [
        _benchmark(
            mode="regex",
            fn=_regex_cleanup,
            root_entries=root_entries,
            keep_pairs=keep_pairs,
            iterations=args.iterations,
        ),
        _benchmark(
            mode="fixed-slice",
            fn=_cleanup_default_root_entries,
            root_entries=root_entries,
            keep_pairs=keep_pairs,
            iterations=args.iterations,
        ),
    ]

    summary = {
        "run_token": run_token,
        "started_at_utc": _utc_now(),
        "iterations": args.iterations,
        "entries": args.entries,
        "results": [asdict(result) for result in results],
    }
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Summary written to {summary_path}")
    for result in results:
        print(
            f"{result.mode}: {result.wall_time_s:.6f}s deleted={result.deleted_entries}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
