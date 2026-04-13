from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

import refiner as mdr
from refiner.io import DataFolder

INPUT_DATASET = "hf://datasets/your-username/your-robot-dataset"
OUTPUT_DATASET = "s3://your-bucket/your-sarm-annotated-dataset"
VIDEO_KEY = "observation.images.main"
STATE_KEY = "observation.state"
FRAME_STRIDE = 30
MAX_IN_FLIGHT = 8

SUBTASKS = [
    "reach object",
    "grasp object",
    "move object",
    "place object",
]

PROVIDER = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    model="gpt-4.1-mini",
)


def _annotation_prompt(subtasks: Sequence[str]) -> str:
    bullet_list = "\n".join(f"- {name}" for name in subtasks)
    return (
        "Segment this robot demonstration into the fixed subtask list below.\n"
        "Use every listed subtask exactly once and keep the entire video covered.\n"
        "Return JSON only with this shape:\n"
        '{"subtasks":[{"name":"...", "timestamps":{"start":"MM:SS","end":"MM:SS"}}]}\n'
        "Allowed subtask names:\n"
        f"{bullet_list}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("annotation payload must be a JSON object")
    return payload


def _parse_timestamp(raw: str) -> float:
    value = raw.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(int(parts[0]) * 60 + int(parts[1]))
    raise ValueError(f"unsupported timestamp format: {raw!r}")


def _seconds_to_frame_index(seconds: float, *, fps: int, max_frame_index: int) -> int:
    frame_index = int(round(seconds * float(fps)))
    return max(0, min(max_frame_index, frame_index))


def _image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


async def _sample_episode_images(row) -> list[str]:
    video = row.videos[VIDEO_KEY].video
    images: list[str] = []
    async for window in mdr.video.iter_frame_windows(
        video,
        offsets=[0],
        stride=FRAME_STRIDE,
        drop_incomplete=False,
    ):
        frame = window.anchor.frame
        image = frame.to_image()
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()
        images.append(_image_data_url(jpeg_bytes))
    if not images:
        raise ValueError(f"episode {row.episode_index} yielded no sampled frames")
    return images


def _annotation_patch(
    annotation: Mapping[str, Any],
    *,
    prefix: str,
    fps: int,
    frame_count: int,
) -> dict[str, Any]:
    subtasks = annotation.get("subtasks")
    if not isinstance(subtasks, Sequence):
        raise ValueError("annotation payload must contain a subtasks list")

    names: list[str] = []
    start_times: list[float] = []
    end_times: list[float] = []

    for item in subtasks:
        if not isinstance(item, Mapping):
            raise ValueError("subtask entries must be objects")
        timestamps = item.get("timestamps")
        if not isinstance(timestamps, Mapping):
            raise ValueError("subtask entries must contain timestamps")
        name = item.get("name")
        start = timestamps.get("start")
        end = timestamps.get("end")
        if (
            not isinstance(name, str)
            or not isinstance(start, str)
            or not isinstance(end, str)
        ):
            raise ValueError("subtask annotation fields must be strings")
        names.append(name)
        start_times.append(_parse_timestamp(start))
        end_times.append(_parse_timestamp(end))

    max_frame_index = max(0, frame_count - 1)
    start_frames = [
        _seconds_to_frame_index(value, fps=fps, max_frame_index=max_frame_index)
        for value in start_times
    ]
    end_frames = [
        _seconds_to_frame_index(value, fps=fps, max_frame_index=max_frame_index)
        for value in end_times
    ]

    patch = {
        f"{prefix}_subtask_names": names,
        f"{prefix}_subtask_start_times": start_times,
        f"{prefix}_subtask_end_times": end_times,
        f"{prefix}_subtask_start_frames": start_frames,
        f"{prefix}_subtask_end_frames": end_frames,
    }
    if prefix == "sparse":
        patch.update(
            {
                "subtask_names": names,
                "subtask_start_times": start_times,
                "subtask_end_times": end_times,
                "subtask_start_frames": start_frames,
                "subtask_end_frames": end_frames,
            }
        )
    return patch


def _single_stage_sparse_patch(row) -> dict[str, Any]:
    fps = int(row.metadata.info.fps)
    frame_count = int(row.length)
    if frame_count <= 0:
        raise ValueError(f"episode {row.episode_index} has no frames")
    end_time = max(0.0, float(frame_count - 1) / float(fps))
    return _annotation_patch(
        {
            "subtasks": [
                {
                    "name": "task",
                    "timestamps": {
                        "start": "00:00",
                        "end": f"{int(end_time // 60):02d}:{int(end_time % 60):02d}",
                    },
                }
            ]
        },
        prefix="sparse",
        fps=fps,
        frame_count=frame_count,
    )


async def annotate_dense_subtasks(row, generate):
    if VIDEO_KEY not in row.videos:
        raise ValueError(
            f"episode {row.episode_index} is missing required video key {VIDEO_KEY!r}"
        )
    if STATE_KEY not in row.frames.table.column_names:
        raise ValueError(
            f"episode {row.episode_index} is missing required state key {STATE_KEY!r}"
        )

    image_urls = await _sample_episode_images(row)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _annotation_prompt(SUBTASKS)}
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    )

    response = await generate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "response_format": {"type": "json_object"},
        }
    )

    dense_annotation = _extract_json_object(response.text)
    patch = {}
    patch.update(
        _annotation_patch(
            dense_annotation,
            prefix="dense",
            fps=int(row.metadata.info.fps),
            frame_count=int(row.length),
        )
    )
    patch.update(_single_stage_sparse_patch(row))
    return row.update(patch)


def compute_temporal_proportions(
    dataset_root: str,
    *,
    prefix: str,
) -> dict[str, float]:
    rows = mdr.read_lerobot(dataset_root).materialize()
    totals: dict[str, list[float]] = {}

    for row in rows:
        names = row.get(f"{prefix}_subtask_names")
        starts = row.get(f"{prefix}_subtask_start_times")
        ends = row.get(f"{prefix}_subtask_end_times")
        if (
            not isinstance(names, Sequence)
            or not isinstance(starts, Sequence)
            or not isinstance(ends, Sequence)
        ):
            continue
        durations: dict[str, float] = {}
        total_duration = 0.0
        for name, start, end in zip(names, starts, ends, strict=False):
            if not isinstance(name, str):
                continue
            duration = max(0.0, float(end) - float(start))
            durations[name] = duration
            total_duration += duration
        if total_duration <= 0:
            continue
        for name, duration in durations.items():
            totals.setdefault(name, []).append(duration / total_duration)

    averaged = {
        name: sum(values) / len(values) for name, values in totals.items() if values
    }
    total = sum(averaged.values())
    if total > 0:
        averaged = {name: value / total for name, value in averaged.items()}
    return averaged


def write_temporal_proportions(dataset_root: str, *, prefix: str) -> None:
    payload = compute_temporal_proportions(dataset_root, prefix=prefix)
    output = DataFolder.resolve(dataset_root)
    with output.open(
        f"meta/temporal_proportions_{prefix}.json",
        mode="wt",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, sort_keys=True)


if __name__ == "__main__":
    pipeline = (
        mdr.read_lerobot(INPUT_DATASET)
        .map_async(
            mdr.inference.generate(
                fn=annotate_dense_subtasks,
                provider=PROVIDER,
                default_generation_params={"temperature": 0.1},
                max_concurrent_requests=MAX_IN_FLIGHT,
            ),
            max_in_flight=MAX_IN_FLIGHT,
        )
        .write_lerobot(OUTPUT_DATASET)
    )

    pipeline.launch_local(name="lerobot-sarm-annotation", num_workers=1)
    write_temporal_proportions(OUTPUT_DATASET, prefix="sparse")
    write_temporal_proportions(OUTPUT_DATASET, prefix="dense")
