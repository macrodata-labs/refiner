from __future__ import annotations

import asyncio
import importlib
import io
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
import refiner.robotics
import refiner.inference as inference_module
from refiner.io import DataFile
from refiner.io import DataFolder
from refiner.inference import InferenceResponse
from refiner.pipeline.data.row import DictRow
from refiner.robotics.lerobot_format import LeRobotMetadata, LeRobotRow
from refiner.robotics.row import RoboticsRow, _robot_row_converter

subtask_labeling_module = importlib.import_module(
    "refiner.robotics.subtask_annotation.labeling"
)
subtask_segmentation_module = importlib.import_module(
    "refiner.robotics.subtask_annotation.segmentation"
)
subtask_utils_module = importlib.import_module(
    "refiner.robotics.subtask_annotation.utils"
)


def _subtask_annotation_result(segments: list[dict[str, Any]]) -> Any:
    return subtask_segmentation_module._SubtaskAnnotationResult.model_validate(
        {"segments": segments}
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
    return [
        sheet
        async for sheet in subtask_utils_module.timestamped_contact_sheets(
            video,
            sample_sec=0.4,
            frame_width=64,
            frames_per_sheet=2,
            columns=2,
            quality=95,
        )
    ]


async def _collect_sheets(video: Any, **kwargs: Any):
    return [
        sheet
        async for sheet in subtask_utils_module.timestamped_contact_sheets(
            video,
            **kwargs,
        )
    ]


async def _first_contact_sheet(video: Any):
    sheets = subtask_utils_module.timestamped_contact_sheets(
        video,
        sample_sec=0.5,
        frame_width=16,
        frames_per_sheet=2,
        columns=2,
    )
    try:
        return await anext(sheets)
    finally:
        await sheets.aclose()


def _lerobot_row(
    tmp_path,
    *,
    tasks: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> LeRobotRow:
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
                **(extra or {}),
            }
        ),
        metadata=cast(LeRobotMetadata, None),
        frames=[],
        root=DataFolder.resolve(tmp_path),
    )


def _robotics_row(
    tmp_path,
    *,
    tasks: str | list[str] | None = None,
) -> RoboticsRow:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=3, fps=5)
    row = DictRow(
        {
            "episode_id": "episode-1",
            "tasks": [] if tasks is None else tasks,
            "video": str(path),
        }
    )
    converter = _robot_row_converter(
        episode_id_key="episode_id",
        task_key="tasks",
        fps=5,
        video_keys={"observation.images.main": "video"},
    )
    return cast(RoboticsRow, converter(row))


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


def test_timestamped_contact_sheets_streams_sheet_batches() -> None:
    from PIL import Image

    class _FakeImageFrame:
        def __init__(self, value: int) -> None:
            self._value = value

        def to_image(self) -> Image.Image:
            return Image.new("RGB", (16, 12), color=(self._value, 0, 0))

    class _FakeDecodedFrame:
        def __init__(self, timestamp_s: float, value: int) -> None:
            self.timestamp_s = timestamp_s
            self.frame = _FakeImageFrame(value)

    class _FakeVideo:
        def __init__(self) -> None:
            self.frames_consumed = 0

        async def iter_frames(self):
            for index in range(5):
                self.frames_consumed += 1
                yield _FakeDecodedFrame(timestamp_s=index * 0.5, value=index)

    video = _FakeVideo()
    sheet = asyncio.run(_first_contact_sheet(video))

    assert sheet.timestamps == (0.0, 0.5)
    assert video.frames_consumed == 2


def test_timestamped_contact_sheets_engravings_are_visible(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=2, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheet = asyncio.run(_build_sheets(video))[0]
    image = Image.open(io.BytesIO(sheet.data)).convert("RGB")
    pixels = np.asarray(image)

    top_right_badge = pixels[:16, 64 - 48 : 64]
    top_left_badge = pixels[:26, :72]
    lower_left_frame_area = pixels[32:48, :16]

    assert top_left_badge.mean() < 80
    assert top_left_badge[:, :, :].max() > 180
    assert top_right_badge.mean() > 50
    assert lower_left_frame_area.mean() < 120


def test_timestamped_contact_sheets_reject_invalid_options(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    with pytest.raises(ValueError, match="sample_sec must be > 0"):
        asyncio.run(_collect_sheets(video, sample_sec=0))


def test_subtask_annotation_builds_generate_text_block(monkeypatch) -> None:
    seen = {}

    def _fake_generate_text(**kwargs):
        seen.update(kwargs)
        return "annotation-block"

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)
    provider = mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest")

    block = mdr.robotics.subtask_annotation(
        provider=provider,
        video_key="observation.images.main",
        max_concurrent_requests=17,
    )

    assert block == "annotation-block"
    assert seen["provider"] is provider
    assert seen["max_concurrent_requests"] == 17
    assert callable(seen["fn"])


def test_subtask_annotation_requires_video_key() -> None:
    with pytest.raises(ValueError, match="video_key must be non-empty"):
        mdr.robotics.subtask_annotation(video_key="")


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
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=_subtask_annotation_result(
                [
                    {"start_sec": 0.4, "end_sec": 0.2, "subtask": "ignored"},
                    {"start_sec": 0.0, "end_sec": 4.0, "subtask": "open drawer"},
                ],
            ),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert seen["provider"].model == "gemini-flash-latest"
    assert request["temperature"] == 0.1
    assert request["maxRetries"] == 4
    assert request["schema"] is subtask_segmentation_module._SubtaskAnnotationResult
    assert request["provider_options"] == {
        "google": {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }
    }
    message = request["messages"][0]
    assert message["role"] == "user"
    assert "Episode instruction: open the drawer" in message["content"][0]["text"]
    assert (
        "Segment only completed robot manipulation events"
        in (message["content"][0]["text"])
    )
    assert "Do not split approach, grasp adjustment" in (message["content"][0]["text"])
    assert "Most segments should be 2-10 seconds" in message["content"][0]["text"]
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


def test_subtask_annotation_accepts_robotics_row(tmp_path, monkeypatch) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _robotics_row(tmp_path, tasks=["pick cable", "route cable"])
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
    )
    request = {}

    async def _fake_request(**kwargs):
        request.update(kwargs)
        return InferenceResponse(
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=_subtask_annotation_result(
                [{"start_sec": 0.0, "end_sec": 0.4, "subtask": "pick"}],
            ),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert (
        "Episode instruction: pick cable; route cable"
        in request["messages"][0]["content"][0]["text"]
    )
    assert result["predicted_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 0.4, "subtask": "pick"}
    ]


def test_subtask_annotation_writes_empty_segments_for_blocked_prompt(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _robotics_row(tmp_path, tasks=["inspect workspace"])
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
    )

    async def _blocked_request(**kwargs):
        raise RuntimeError(
            "google generation response is missing candidates[0]: "
            "promptFeedback.blockReason=PROHIBITED_CONTENT"
        )

    result = asyncio.run(cast(Any, block)(row, _blocked_request))

    assert result["predicted_subtasks"] == []


def test_subtask_annotation_can_raise_for_blocked_prompt(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _robotics_row(tmp_path, tasks=["inspect workspace"])
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
        on_blocked_prompt="raise",
    )

    async def _blocked_request(**kwargs):
        raise RuntimeError(
            "google generation response is missing candidates[0]: "
            "promptFeedback.blockReason=PROHIBITED_CONTENT"
        )

    with pytest.raises(RuntimeError, match="PROHIBITED_CONTENT"):
        asyncio.run(cast(Any, block)(row, _blocked_request))


def test_subtask_annotation_prompt_uses_completed_events_count_guard(
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
        frames_per_sheet=6,
        columns=4,
    )
    request = {}

    async def _fake_request(**kwargs):
        request.update(kwargs)
        return InferenceResponse(
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=_subtask_annotation_result([]),
        )

    asyncio.run(cast(Any, block)(row, _fake_request))

    prompt = request["messages"][0]["content"][0]["text"]
    assert "Segment only completed robot manipulation events" in prompt
    assert "Do not merge separate pick/place/open/close/pour/wipe events" in prompt
    assert "Ignore label wording quality" in prompt
    assert "columns and" not in prompt


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
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=_subtask_annotation_result(
                [
                    {"start_sec": 0.0, "end_sec": 3.49, "subtask": "short action"},
                    {"start_sec": 3.5, "end_sec": 7.0, "subtask": "long action"},
                ],
            ),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert result["predicted_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 3.49, "subtask": "short action"},
        {"start_sec": 3.5, "end_sec": 7.0, "subtask": "long action"},
    ]


def test_subtask_annotation_logs_on_overlapping_segments(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)
    logged_warnings = []
    monkeypatch.setattr(
        subtask_utils_module.logger,
        "warning",
        lambda *args: logged_warnings.append(args),
    )

    row = _lerobot_row(tmp_path)
    block = mdr.robotics.subtask_annotation(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
    )

    async def _fake_request(**kwargs):
        return InferenceResponse(
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=_subtask_annotation_result(
                [
                    {"start_sec": 0.0, "end_sec": 2.0, "subtask": "reach"},
                    {"start_sec": 1.5, "end_sec": 3.0, "subtask": "grasp"},
                ],
            ),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert result["predicted_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 2.0, "subtask": "reach"},
        {"start_sec": 1.5, "end_sec": 3.0, "subtask": "grasp"},
    ]
    assert len(logged_warnings) == 1
    assert "overlapping segments" in logged_warnings[0][0]


def test_subtask_labeling_builds_generate_text_block(monkeypatch) -> None:
    seen = {}

    def _fake_generate_text(**kwargs):
        seen.update(kwargs)
        return "labeling-block"

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)
    provider = mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest")

    block = mdr.robotics.subtask_labeling(
        provider=provider,
        video_key="observation.images.main",
        max_concurrent_requests=11,
    )

    assert block == "labeling-block"
    assert seen["provider"] is provider
    assert seen["max_concurrent_requests"] == 11
    assert callable(seen["fn"])


def test_subtask_labeling_labels_fixed_segments_with_seed_labels(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(
        tmp_path,
        tasks=["open the drawer"],
        extra={
            "predicted_subtasks": [
                {"start_sec": 0.0, "end_sec": 0.2, "label": "reach drawer"},
                {"start_sec": 0.2, "end_sec": 0.4, "label": "pull drawer"},
            ],
        },
    )
    block = mdr.robotics.subtask_labeling(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
        max_frames_per_segment=2,
    )
    requests = []

    async def _fake_request(**kwargs):
        requests.append(kwargs)
        label = (
            " grasp the drawer handle " if len(requests) == 1 else "pull open drawer"
        )
        return InferenceResponse(
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=subtask_labeling_module._SubtaskLabelingResult(label=label),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert len(requests) == 2
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["schema"] is subtask_labeling_module._SubtaskLabelingResult
    assert requests[0]["provider_options"] == {
        "google": {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }
    }
    first_prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert "Original predicted label for this exact segment" in first_prompt
    assert "reach drawer" in first_prompt
    assert "Use previous/next images only" in first_prompt
    assert len(requests[0]["messages"][0]["content"]) == 4
    assert requests[0]["messages"][0]["content"][1]["mediaType"] == "image/jpeg"
    assert result["labeled_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 0.2, "label": "grasp the drawer handle"},
        {"start_sec": 0.2, "end_sec": 0.4, "label": "pull open drawer"},
    ]


def test_subtask_labeling_uses_plain_prompt_without_seed_labels(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(
        tmp_path,
        tasks=["open the drawer"],
        extra={"predicted_subtasks": [{"start_sec": 0.0, "end_sec": 0.2}]},
    )
    block = mdr.robotics.subtask_labeling(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
        max_frames_per_segment=2,
    )
    request = {}

    async def _fake_request(**kwargs):
        request.update(kwargs)
        return InferenceResponse(
            text="",
            finish_reason="stop",
            usage={},
            response={},
            object=subtask_labeling_module._SubtaskLabelingResult(
                label="pull open drawer"
            ),
        )

    result = asyncio.run(cast(Any, block)(row, _fake_request))

    assert request["provider_options"] == {
        "google": {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }
    }
    prompt = request["messages"][0]["content"][0]["text"]
    assert "Original predicted label for this exact segment" not in prompt
    assert "Treat the original predicted label as a strong prior" not in prompt
    assert "The segment boundaries are fixed; do not split or merge" in prompt
    assert "Compare the beginning and end of the current segment" in prompt
    assert result["labeled_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 0.2, "label": "pull open drawer"}
    ]


def test_subtask_labeling_falls_back_to_seed_label(
    tmp_path,
    monkeypatch,
) -> None:
    def _fake_generate_text(**kwargs):
        return kwargs["fn"]

    monkeypatch.setattr(inference_module, "generate_text", _fake_generate_text)

    row = _lerobot_row(
        tmp_path,
        extra={
            "segments": [
                {"start_sec": 0.0, "end_sec": 0.2, "label": " Pick Up Object "}
            ],
        },
    )
    block = mdr.robotics.subtask_labeling(
        provider=mdr.inference.GoogleEndpointProvider(model="gemini-flash-latest"),
        video_key="observation.images.main",
        segments_column="segments",
    )

    async def _blocked_request(**kwargs):
        raise RuntimeError(
            "google generation response is missing candidates[0]: "
            "promptFeedback.blockReason=PROHIBITED_CONTENT"
        )

    result = asyncio.run(cast(Any, block)(row, _blocked_request))

    assert result["labeled_subtasks"] == [
        {"start_sec": 0.0, "end_sec": 0.2, "label": "pick up object"}
    ]
