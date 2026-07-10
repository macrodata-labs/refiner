"""Opt-in live-provider smoke test.

Run with:
    REFINER_RUN_LIVE_GEMINI=1 GOOGLE_GENERATIVE_AI_API_KEY=... \
      pytest tests/robotics/test_subtask_annotation_live.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
from refiner.pipeline.data.row import DictRow
from refiner.robotics.row import RoboticsRow, _robot_row_converter


pytestmark = pytest.mark.skipif(
    os.environ.get("REFINER_RUN_LIVE_GEMINI") != "1",
    reason="set REFINER_RUN_LIVE_GEMINI=1 to run billable live-provider smoke",
)


def _write_smoke_video(path) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            image = np.full((64, 96, 3), 235, dtype=np.uint8)
            x = min(68, 4 + index * 3)
            image[24:42, x : x + 18] = (210, 30, 30)
            if index >= 12:
                image[24:42, 72:90] = (30, 170, 60)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_live_gemini_subtask_annotation_contract(tmp_path) -> None:
    if not os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY"):
        pytest.skip("GOOGLE_GENERATIVE_AI_API_KEY is required")
    model = os.environ.get("REFINER_GEMINI_SMOKE_MODEL", "gemini-3.5-flash")
    thinking_budget = int(os.environ.get("REFINER_GEMINI_THINKING_BUDGET", "16384"))
    video_path = tmp_path / "smoke.mp4"
    _write_smoke_video(video_path)
    converter = _robot_row_converter(
        episode_id_key="episode_id",
        task_key="task",
        fps=8,
        video_keys={"observation.images.main": "video"},
    )
    row = cast(
        RoboticsRow,
        converter(
            DictRow(
                {
                    "episode_id": "live-smoke",
                    "task": "pick the red block and place it at the green target",
                    "video": str(video_path),
                    "partitioner_segment_count": 2,
                }
            )
        ),
    )
    block = mdr.robotics.subtask_annotation(
        profile=mdr.robotics.WALDEN_V1,
        provider=mdr.inference.GoogleEndpointProvider(model=model),
        video_key="observation.images.main",
        count_prior_column="partitioner_segment_count",
        thinking_budget=thinking_budget,
        max_concurrent_requests=1,
    )

    async def _run() -> Any:
        try:
            return await block(row)
        finally:
            await block.aclose()

    result_row = asyncio.run(_run())
    result = result_row["subtask_annotation_result"]
    assert result["status"] in {"ok", "empty", "partial"}
    assert result["provenance"]["model"] == model
    assert result["provenance"]["count_prior"] == 2
    assert len(result["provenance"]["config_hash"]) == 64
    assert result_row["predicted_subtasks"] is not None
