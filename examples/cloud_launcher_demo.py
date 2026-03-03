from __future__ import annotations

import os
import time

import refiner as mdr


def main() -> None:
    if not os.getenv("MACRODATA_API_KEY"):
        raise SystemExit(
            "MACRODATA_API_KEY is required. "
            "Run `macrodata login` or export MACRODATA_API_KEY."
        )

    items = [
        {"id": "r1", "text": "hello world"},
        {"id": "r2", "text": "cloud demo"},
    ]

    pipeline = mdr.from_items(items, shard_size_rows=1).map(
        lambda row: {"id": row["id"], "text": row["text"], "text_len": len(row["text"])}
    )

    run_name = f"cloud-demo-{int(time.time())}"
    print(f"Submitting cloud run: {run_name}")

    result = pipeline.launch_cloud(name=run_name, num_workers=1)

    print("Cloud submission accepted")
    print(f"job_id  : {result.job_id}")
    print(f"stage_id: {result.stage_id}")
    print(f"status  : {result.status}")


if __name__ == "__main__":
    main()
