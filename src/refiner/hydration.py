from __future__ import annotations
import asyncio
import heapq
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from typing import Any, Literal
from refiner.io import DataFile
from refiner.runtime.execution import submit
from refiner.sources.row import Row
from refiner.video import Video, VideoFile

def hydrate_file(
    columns: str | Sequence[str] | None = None,
    *,
    on_error: Literal["raise", "null"] = "raise",
    video_hydration: Literal["file", "bytes"] = "file",
    timeout_s: float = 60.0,
    max_in_flight: int = 16,
) -> Callable[[Row], Iterable[Row]]:
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    if video_hydration not in {"file", "bytes"}:
        raise ValueError("video_hydration must be 'file' or 'bytes'")
    if timeout_s <= 0 or max_in_flight <= 0:
        raise ValueError("timeout_s and max_in_flight must be > 0")


    if columns is None:
        wanted = None

    elif isinstance(columns, str):
        if not columns:
            raise ValueError("columns cannot be empty")
        wanted = (columns,)
    else:
        wanted = tuple(columns)
        if not wanted or any((not isinstance(c, str) or not c) for c in wanted):
            raise ValueError("columns must contain non-empty strings")
    futures: set[Future[tuple[int, Row]]] = set()
    ready: list[tuple[int, Row]] = []
    next_submit = 0
    next_yield = 0

    async def _read(uri: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: (d := DataFile.resolve(uri)).fs.cat(d.path)
        )

    async def _hydrate(row: Row, idx: int) -> tuple[int, Row]:
        updates: dict[str, Any] = {}
        for key in (wanted or tuple(row.keys())):
            if key not in row:
                continue
            value = row[key]
            try:
                if isinstance(value, Video):
                    if video_hydration == "file" and value.file is None:
                        fh = VideoFile(value.uri)
                        await asyncio.get_running_loop().run_in_executor(None, fh.to_local_path)
                        updates[key] = value.with_file(fh)
                    elif video_hydration == "bytes" and value.bytes is None:
                        updates[key] = value.with_bytes(await _read(value.uri))
                elif isinstance(value, str):
                    updates[key] = await _read(value)
            except Exception:
                if on_error == "raise":
                    raise
                if isinstance(value, Video):
                    updates[key] = value.with_file(None) if video_hydration == "file" else value.with_bytes(None)
                else:
                    updates[key] = None
        return idx, (row if not updates else row.update(updates))

    def _drain(flush: bool) -> Iterable[Row]:
        nonlocal futures, next_yield
        while futures and (flush or len(futures) >= max_in_flight):
            done, pending = wait(futures, return_when=ALL_COMPLETED if flush else FIRST_COMPLETED)
            futures = set(pending)
            for f in done:
                heapq.heappush(ready, f.result(timeout=timeout_s))
            while ready and ready[0][0] == next_yield:
                _, row = heapq.heappop(ready)
                next_yield += 1
                yield row

    def _fn(row: Row) -> Iterable[Row]:
        nonlocal next_submit
        futures.add(submit(_hydrate(row, next_submit)))
        next_submit += 1
        yield from _drain(False)

    def _flush() -> Iterable[Row]:
        yield from _drain(True)

    _fn.flush = _flush  # type: ignore[attr-defined]
    return _fn
__all__ = ["hydrate_file"]
