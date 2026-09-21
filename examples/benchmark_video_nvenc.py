"""Compare NVENC presets and concurrency on a local video: python this.py input.mp4."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import time

import refiner as mdr


async def benchmark(source: Path, *, copies: int = 8) -> list[dict]:
    if copies <= 0:
        raise ValueError("copies must be positive")
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(source),
            ]
        )
    )["streams"][0]
    frames = int(stream["nb_frames"])
    results = []
    for preset in ("p1", "p6"):
        for lanes in (1, 2, 3, 4):
            semaphore = asyncio.Semaphore(lanes)
            config = mdr.video.NVENCConfig(preset=preset)
            with tempfile.TemporaryDirectory(prefix="refiner-benchmark-") as temp:
                # Exclude driver initialization from the measured batch.
                await mdr.video.transcode_video(
                    source, Path(temp) / "warmup.mp4", config=config
                )

                async def encode(index: int) -> None:
                    async with semaphore:
                        await mdr.video.transcode_video(
                            source, Path(temp) / f"{index}.mp4", config=config
                        )

                started = time.perf_counter()
                await asyncio.gather(*(encode(index) for index in range(copies)))
                elapsed = time.perf_counter() - started
            result = {
                "preset": preset,
                "lanes": lanes,
                "copies": copies,
                "frames": frames * copies,
                "wall_s": round(elapsed, 3),
                "fps": round(frames * copies / elapsed, 1),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--copies", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(benchmark(args.source.resolve(), copies=args.copies))
