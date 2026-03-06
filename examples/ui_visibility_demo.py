from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import refiner as mdr
from refiner.sources.row import Row


def build_input(path: Path) -> None:
    rows = [
        {"id": 1, "country": "us", "score": 10},
        {"id": 2, "country": "fr", "score": 25},
        {"id": 3, "country": "de", "score": 40},
        {"id": 4, "country": "us", "score": 55},
        {"id": 5, "country": "ca", "score": 70},
        {"id": 6, "country": "us", "score": 85},
        {"id": 7, "country": "mx", "score": 95},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _to_int(value: Any) -> int:
    return int(value)


def _to_str(value: Any) -> str:
    return str(value)


def map_row(row: Row) -> dict[str, Any]:
    shard_id = str(row.get("__shard_id", ""))
    mdr.log_throughput("rows_seen", 1, shard_id=shard_id, unit="rows")
    mdr.log_histogram(
        "score_hist", float(row["score"]), shard_id=shard_id, unit="points"
    )
    return {
        "id": _to_int(row["id"]),
        "country": _to_str(row["country"]),
        "score": _to_int(row["score"]),
        "bonus": _to_int(row["id"]) % 7,
    }


def normalize_batch(rows: list[Row]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["score"] = _to_int(updated["score"]) + 5
        out.append(updated)
    return out


def duplicate_selected(row: Row) -> list[dict[str, Any]]:
    base = dict(row)
    base["dup"] = 0
    if _to_int(base["id"]) % 2 == 0:
        extra = dict(base)
        extra["dup"] = 1
        return [base, extra]
    return [base]


def main() -> None:
    with TemporaryDirectory(prefix="mdr-ui-visibility-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        build_input(input_path)

        pipeline = (
            mdr.read_jsonl(str(input_path), target_shard_bytes=90)
            .map(map_row)
            .filter(lambda row: int(row["score"]) >= 15)
            .batch_map(normalize_batch, batch_size=3)
            .flat_map(duplicate_selected)
            .with_columns(
                total=mdr.col("score") + mdr.col("bonus"),
                country_upper=mdr.col("country").str.upper().str.strip(),
                cohort=mdr.if_else(mdr.col("score") >= 80, "high", "baseline"),
                keep=mdr.col("country").str.contains("u") | (mdr.col("total") >= 30),
            )
            .filter(mdr.col("keep"))
            .rename(country_upper="country_code")
            .cast(total="int64", score="int64")
            .drop("bonus", "keep")
            .select("id", "country_code", "score", "total", "cohort", "dup")
        )

        stats = pipeline.launch_local(
            name="Example: UI Visibility Mixed Steps",
            num_workers=2,
            heartbeat_every_rows=2,
            workdir=str(tmp_path / "workdir"),
        )
        print("Launch stats:", stats)


if __name__ == "__main__":
    main()
