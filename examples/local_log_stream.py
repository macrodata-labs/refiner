from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import refiner as mdr
from refiner.pipeline.data.row import Row

_ROWS_PER_WORKER = 5
_WORKER_SPACING_SECONDS = 7.5


def emit_logs(row: Row) -> dict[str, Any]:
    row_x = row["x"]
    assert isinstance(row_x, int)
    row_id = row_x
    worker_slot = row_id // _ROWS_PER_WORKER
    mdr.logger.info("loguru row={} starting", row_id)
    time.sleep(_WORKER_SPACING_SECONDS if row_id % _ROWS_PER_WORKER == 0 else 0.2)
    mdr.logger.info("loguru row={} finished", row_id)
    return {"x": row_id, "double_x": row_id * 2, "worker_slot": worker_slot}


def build_input(path: Path, *, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(rows):
            handle.write(json.dumps({"x": idx}) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise local launcher live worker log streaming."
    )
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--logs",
        choices=("all", "none", "one", "errors"),
        default="all",
    )
    parser.add_argument(
        "--rundir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_rundir = Path("tmp") / (
        "local-log-stream-run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    rundir = (args.rundir or default_rundir).expanduser().resolve()
    input_path = rundir / "demo-input.jsonl"
    output_path = rundir / "out"
    build_input(input_path, rows=args.rows)

    pipeline = (
        mdr.read_jsonl(str(input_path), num_shards=args.workers)
        .map(emit_logs)
        .write_jsonl(output_path)
    )
    os.environ["REFINER_LOCAL_LOGS"] = args.logs
    pipeline.launch_local(
        name="local-log-stream-demo",
        num_workers=args.workers,
        rundir=str(rundir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
