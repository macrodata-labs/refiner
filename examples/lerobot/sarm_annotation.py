from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

import av

import refiner as mdr
from refiner.io import DataFolder

INPUT_DATASET = "hf://datasets/lerobot/aloha_sim_transfer_cube_human"
OUTPUT_DATASET = (
    "hf://buckets/macrodata/test_bucket/aloha_sim_insertion_human_sarm_annotated"
)
VIDEO_KEY = "observation.images.top"
STATE_KEY = "observation.state"
MAX_IN_FLIGHT = 8
MIN_DECODED_CLIP_FRAMES = 4
QWEN_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

SUBTASKS = [
    "reach object",
    "grasp object",
    "move object",
    "place object",
]

PROVIDER = mdr.inference.VLLMProvider(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    model_max_context=32768,
    extra_kwargs={"limit-mm-per-prompt": "video=1"},
)


def _annotation_prompt(subtasks: Sequence[str]) -> str:
    bullet_list = "\n".join(f" - {name}" for name in subtasks)
    return (
        "# Role\n"
        "You are a robotics vision system for temporal action localization.\n\n"
        "# Task\n"
        "Segment one successful demonstration video into distinct, non-overlapping"
        " atomic actions from the fixed subtask vocabulary below.\n\n"
        "# Subtask Label Set\n"
        "Use only these labels exactly as written:\n"
        f"[\n{bullet_list}\n]\n\n"
        "# Hard Constraints\n"
        "1. Cover the full video from 00:00 to the final timestamp.\n"
        "2. No gaps between subtasks.\n"
        "3. The end of one subtask must equal the start of the next.\n"
        "4. Use each subtask exactly once in logical order.\n"
        "5. Do not split the video into equal chunks unless the visuals truly support it.\n"
        "6. Timestamps must be MM:SS.\n\n"
        "# Output\n"
        "First write a detailed timeline as bullet points.\n"
        "Then output only valid JSON in this shape:\n"
        '{"subtasks":[{"name":"EXACT_NAME_FROM_LIST","timestamps":{"start":"MM:SS","end":"MM:SS"}}]}\n'
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


def _video_data_url(video_bytes: bytes) -> str:
    encoded = base64.b64encode(video_bytes).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _prompt_text_for_provider(provider: object, prompt: str) -> str:
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model.startswith("Qwen/"):
        return f"{QWEN_VIDEO_PLACEHOLDER}\n{prompt}"
    return prompt


def _inspect_exported_clip(video_bytes: bytes) -> dict[str, float | int]:
    decoded_frames = 0
    width = 0
    height = 0
    first_timestamp_s: float | None = None
    last_timestamp_s: float | None = None

    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "video")
        for frame in container.decode(stream):
            if not isinstance(frame, av.VideoFrame):
                continue
            decoded_frames += 1
            width = frame.width
            height = frame.height
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp_s = float(frame.pts * frame.time_base)
            if first_timestamp_s is None:
                first_timestamp_s = timestamp_s
            last_timestamp_s = timestamp_s

    duration_s = 0.0
    if first_timestamp_s is not None and last_timestamp_s is not None:
        duration_s = max(0.0, last_timestamp_s - first_timestamp_s)

    return {
        "bytes": len(video_bytes),
        "decoded_frames": decoded_frames,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


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

    video = row.videos[VIDEO_KEY].video
    clip_bytes = await mdr.video.export_clip_bytes(video, stream_key=VIDEO_KEY)
    clip_stats = _inspect_exported_clip(clip_bytes)
    mdr.logger.info(
        "episode {} exported clip bytes={} decoded_frames={} size={}x{} duration_s={}",
        row.episode_index,
        clip_stats["bytes"],
        clip_stats["decoded_frames"],
        clip_stats["width"],
        clip_stats["height"],
        clip_stats["duration_s"],
    )
    if int(clip_stats["decoded_frames"]) < MIN_DECODED_CLIP_FRAMES:
        raise ValueError(
            "exported clip decodes to too few frames for VLM video inference: "
            f"{clip_stats['decoded_frames']} < {MIN_DECODED_CLIP_FRAMES}"
        )
    content: list[dict[str, Any]] = [
        {
            "type": "video_url",
            "video_url": {
                "url": _video_data_url(clip_bytes),
            },
        },
        {
            "type": "text",
            "text": _prompt_text_for_provider(
                PROVIDER,
                _annotation_prompt(SUBTASKS),
            ),
        },
    ]

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

    pipeline.launch_cloud(
        name="lerobot-sarm-annotation",
        num_workers=1,
        secrets={"HF_TOKEN": None},
    )
