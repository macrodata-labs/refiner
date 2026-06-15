from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator, cast

import numpy as np

import refiner as mdr
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks import rerun as rerun_sink

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(slots=True)
class CaseResult:
    mode: str
    wall_time_s: float
    output_size_bytes: int
    output_file_count: int
    output_matches_input: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local Rerun copy benchmark comparing direct byte copies "
            "with the chunk-selection fallback."
        )
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--rows-per-chunk", type=int, default=1000)
    parser.add_argument("--writes-per-iteration", type=int, default=1)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-token")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_ref() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    return {
        "macrodata-refiner": _package_version("macrodata-refiner"),
        "rerun-sdk": _package_version("rerun-sdk"),
        "pyarrow": _package_version("pyarrow"),
        "numpy": _package_version("numpy"),
    }


def _generate_input(path: Path, *, chunks: int, rows_per_chunk: int) -> None:
    import rerun as rr

    rec = rr.RecordingStream("refiner-rerun-local-benchmark", recording_id="episode-a")
    rec.save(path)
    for chunk_index in range(chunks):
        start = chunk_index * rows_per_chunk
        frames = np.arange(start, start + rows_per_chunk, dtype=np.int64)
        values = np.asarray(frames, dtype=np.float64)
        rec.send_columns(
            "/action/x",
            indexes=[rr.TimeColumn("frame", sequence=frames)],
            columns=rr.Scalars.columns(scalars=values),
        )
    rec.flush()
    rec.disconnect()


@contextmanager
def _force_chunk_fallback() -> Iterator[None]:
    original = rerun_sink._can_copy_source_rrd
    # This is a deliberate benchmark switch: compare the optimized path to the
    # existing chunk-selection fallback on the same source file.
    rerun_sink._can_copy_source_rrd = lambda recording: False  # type: ignore[assignment]
    try:
        yield
    finally:
        rerun_sink._can_copy_source_rrd = original


def _run_copy_case(
    source: Path,
    output: Path,
    *,
    force_fallback: bool,
    writes_per_iteration: int,
) -> CaseResult:
    source_row = next(
        mdr.read_rerun(str(source), materialize_tables=False).source.read()
    )
    block = cast(list[Row], source_row)
    sink = rerun_sink.RerunSink(str(output))
    start = perf_counter()
    if force_fallback:
        with _force_chunk_fallback():
            for _ in range(writes_per_iteration):
                sink.write_shard_block("shard-a", block)
    else:
        for _ in range(writes_per_iteration):
            sink.write_shard_block("shard-a", block)
    sink.on_shard_complete("shard-a")
    wall_time_s = perf_counter() - start
    written = sorted(output.glob("**/*.rrd"))
    if len(written) != writes_per_iteration:
        raise RuntimeError(
            f"expected {writes_per_iteration} output RRDs, got {len(written)}"
        )
    return CaseResult(
        mode="chunk-fallback" if force_fallback else "direct-copy",
        wall_time_s=wall_time_s,
        output_size_bytes=sum(path.stat().st_size for path in written),
        output_file_count=len(written),
        output_matches_input=all(
            path.read_bytes() == source.read_bytes() for path in written
        ),
    )


def main() -> int:
    args = _parse_args()
    run_token = (
        args.run_token
        or f"local-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    artifacts_dir = args.artifacts_dir / run_token
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifacts_dir / "input.rrd"
    _generate_input(input_path, chunks=args.chunks, rows_per_chunk=args.rows_per_chunk)

    results: list[CaseResult] = []
    for iteration in range(args.iterations):
        for mode_name, force_fallback in (
            ("direct-copy", False),
            ("chunk-fallback", True),
        ):
            output_dir = artifacts_dir / f"{mode_name}-{iteration:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            result = _run_copy_case(
                input_path,
                output_dir,
                force_fallback=force_fallback,
                writes_per_iteration=args.writes_per_iteration,
            )
            results.append(result)
            print(
                f"{mode_name} iteration {iteration}: "
                f"{result.wall_time_s:.3f}s output={result.output_size_bytes}"
            )

    summary = {
        "run_token": run_token,
        "git_ref": _git_ref(),
        "started_at_utc": _utc_now(),
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "chunks": args.chunks,
            "rows_per_chunk": args.rows_per_chunk,
        },
        "iterations": args.iterations,
        "results": [asdict(result) for result in results],
        "package_versions": _package_versions(),
    }
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Summary written to {summary_path}")

    direct = [result.wall_time_s for result in results if result.mode == "direct-copy"]
    fallback = [
        result.wall_time_s for result in results if result.mode == "chunk-fallback"
    ]
    if direct and fallback:
        print(
            "direct-copy avg="
            f"{sum(direct) / len(direct):.3f}s "
            "chunk-fallback avg="
            f"{sum(fallback) / len(fallback):.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
