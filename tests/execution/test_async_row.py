from __future__ import annotations

import asyncio

from refiner.pipeline import from_items


async def _delayed_value(value: int, delay: float) -> dict[str, int]:
    await asyncio.sleep(delay)
    return {"x": value}


def test_map_async_preserves_order_by_default() -> None:
    pipeline = from_items([1, 2, 3]).map_async(
        lambda row: _delayed_value(
            int(row["item"]), 0.03 if int(row["item"]) == 1 else 0.0
        )
    )

    out = list(pipeline.iter_rows())
    assert [int(row["x"]) for row in out] == [1, 2, 3]


def test_map_async_without_order_preservation_still_emits_all_rows() -> None:
    pipeline = from_items([1, 2, 3]).map_async(
        lambda row: _delayed_value(
            int(row["item"]), 0.03 if int(row["item"]) == 1 else 0.0
        ),
        preserve_order=False,
    )

    out = list(pipeline.iter_rows())
    assert sorted(int(row["x"]) for row in out) == [1, 2, 3]
