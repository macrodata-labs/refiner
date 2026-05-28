from __future__ import annotations

import asyncio
import importlib
import io
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
import refiner.inference as inference_module
from refiner.io import DataFile
from refiner.io import DataFolder
from refiner.inference import InferenceResponse
from refiner.pipeline.data.row import DictRow
from refiner.robotics.lerobot_format import LeRobotMetadata, LeRobotRow

subtask_annotation_module = importlib.import_module(
    "refiner.robotics.subtask_annotation"
)


def _write_video(path, *, num_frames: int = 6, fps: int = 5) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"

        for value in range(num_frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((12, 16, 3), 64 + value * 8, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


async def _build_sheets(video: mdr.video.VideoFile):
    return await mdr.robotics.timestamped_contact_sheets(
        video,
        sample_sec=0.4,
        frame_width=64,
        frames_per_sheet=2,
        columns=2,
        quality=95,
    )


def _lerobot_row(tmp_path, *, tasks: list[str] | None = None) -> LeRobotRow:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=3, fps=5)
    return LeRobotRow(
        DictRow(
            {
                "episode_index": 3,
                "length": 3,
                "tasks": tasks or [],
                "videos/observation.images.main/uri": "video.mp4",
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 0.6,
            }
        ),
        metadata=cast(LeRobotMetadata, None),
        frames=[],
        root=DataFolder.resolve(tmp_path),
    )


def test_timestamped_contact_sheets_sample_and_tile_video(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=6, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheets = asyncio.run(_build_sheets(video))

    assert len(sheets) == 2
    assert sheets[0].media_type == "image/jpeg"
    assert sheets[0].index == 1
    assert sheets[0].timestamps == (0.0, 0.4)
    assert sheets[0].start_sec == 0.0
    assert sheets[0].end_sec == 0.4
    assert sheets[0].frame_count == 2
    assert sheets[1].timestamps == (0.8,)
    assert sheets[0].width == 128
    assert sheets[0].height == 48
    assert sheets[0].rows == 1
    assert sheets[0].columns == 2
    assert sheets[0].data.startswith(b"\xff\xd8")


def test_timestamped_contact_sheets_engravings_are_visible(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=2, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheet = asyncio.run(_build_sheets(video))[0]
    image = Image.open(io.BytesIO(sheet.data)).convert("RGB")
    pixels = np.asarray(image)

    badge = pixels[:14, :48]
    plain_frame_area = pixels[18:34, :48]

    assert badge.min() < 16
    assert badge.max() > 220
    assert plain_frame_area.mean() > badge.mean()


def test_timestamped_contact_sheets_reject_invalid_options(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    with pytest.raises(ValueError, match="sample_sec must be > 0"):
        asyncio.run(mdr.robotics.timestamped_contact_sheets(video, sample_sec=0))


def test_contact_sheet_prompt_manifest_describes_continuity(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=6, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheets = asyncio.run(_build_sheets(video))
    manifest = mdr.robotics.contact_sheet_prompt_manifest(sheets)

    assert "ordered chronologically" in manifest
    assert "Actions may continue across contact sheet boundaries" in manifest
    assert "Sheet 1: 2 frames, 1x2 grid, 0.00s through 0.40s." in manifest
    assert "Sheet 2: 1 frames, 1x2 grid, 0.80s through 0.80s." in manifest


def test_contact_sheet_prompt_manifest_rejects_empty_sheets() -> None:
    with pytest.raises(ValueError, match="sheets must be non-empty"):
        mdr.robotics.contact_sheet_prompt_manifest([])


def test_subtask_annotation_builds_generate_text_block(monkeypatch) -> None:
    seen = {}

    def _fake_generate_text(**kwargs):
        seen.update(kwargs)
        return "annotation-block"

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)
    provider = mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest")

    block = mdr.robotics.subtask_annotation(
        provider=provider,
        max_concurrent_requests=17,
    )

    assert block == "annotation-block"
    assert seen["provider"] is provider
    assert seen["max_concurrent_requests"] == 17
    assert callable(seen["fn"])


def test_subtask_annotation_block_updates_row(tmp_path, monkeypatch) -> None:
    seen = {}

    def _fake_generate_text(**kwargs):
        seen.update(kwargs)
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(tmp_path, tasks=["open the drawer"])
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
    )
    request = {}

    async def _fake_request(**kwargs):
        request.update(kwargs)
        return InferenceResponse(
            text=(
                '{"segments":['
                '{"start_sec":0.4,"end_sec":0.2,"subtask":"ignored"},'
                '{"start_sec":0.0,"end_sec":4.0,"subtask":"open drawer"}'
                "]}"
            ),
            finish_reason="stop",
            usage={},
            response={},
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert seen["provider"].model == "gemini-flash-latest"
    assert request["temperature"] == 0.1
    assert request["schema"] is subtask_annotation_module._SubtaskAnnotationResult
    message = request["messages"][0]
    assert message["role"] == "user"
    assert "Episode instruction: open the drawer" in message["content"][0]["text"]
    assert (
        "Each image is a contact sheet with 5 columns and 4 rows."
        in (message["content"][0]["text"])
    )
    assert (
        "Use the visible timestamp printed inside the tile"
        in (message["content"][0]["text"])
    )
    assert (
        "Actions may continue across contact sheet boundaries"
        not in message["content"][0]["text"]
    )
    assert message["content"][1]["mediaType"] == "image/jpeg"
    assert result["predicted_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 4.0, "subtask": "open drawer"}
    ]
    assert "predicted_subtasks_json" not in result
    assert "annotation_model" not in result
    assert "raw_annotation_output" not in result


def test_subtask_annotation_can_include_contact_sheet_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(tmp_path, tasks=["open the drawer"])
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
        include_contact_sheet_manifest=True,
    )
    request = {}

    async def _fake_request(**kwargs):
        request.update(kwargs)
        return InferenceResponse(
            text='{"segments":[]}',
            finish_reason="stop",
            usage={},
            response={},
            object=subtask_annotation_module._SubtaskAnnotationResult(segments=[]),
        )

    asyncio.run(cast(Any, block)(row, _fake_request))

    assert (
        "Actions may continue across contact sheet boundaries"
        in request["messages"][0]["content"][0]["text"]
    )


def test_subtask_annotation_keeps_short_segments_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(tmp_path)
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
    )

    async def _fake_request(**kwargs):
        return InferenceResponse(
            text=(
                "```json\n"
                '{"segments":['
                '{"start_sec":0.0,"end_sec":3.49,"subtask":"short action"},'
                '{"start_sec":3.5,"end_sec":7.0,"subtask":"long action"}'
                "]}\n"
                "```"
            ),
            finish_reason="stop",
            usage={},
            response={},
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert result["predicted_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 3.49, "subtask": "short action"},
        {"start_sec": 3.5, "end_sec": 7.0, "subtask": "long action"},
    ]
