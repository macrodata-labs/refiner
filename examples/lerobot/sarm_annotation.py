from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any
import os

import refiner as mdr

INPUT_DATASET = "hf://datasets/lerobot/aloha_sim_transfer_cube_human"
OUTPUT_DATASET = (
    "hf://buckets/macrodata/test_bucket/aloha_sim_insertion_human_sarm_annotated"
)
VIDEO_KEY = "observation.images.top"
MAX_IN_FLIGHT = 200

SUBTASKS = [
    "reach object",
    "grasp object",
    "move object",
    "place object",
]

PROVIDER = mdr.inference.VLLMProvider(
    model="Qwen/Qwen3-VL-8B-Instruct",
)

VLM_TRANSCODE_CONFIG = mdr.video.VideoTranscodeConfig(
    codec="libx264",
    pix_fmt="yuv420p",
    encoder_options={"preset": "veryfast"},
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


async def annotate_dense_subtasks(row, generate):
    if VIDEO_KEY not in row.videos:
        raise ValueError(
            f"episode {row.episode_index} is missing required video key {VIDEO_KEY!r}"
        )

    video = row.videos[VIDEO_KEY].video
    clip_bytes = await video.export_clip(
        force_transcode=True,
        transcode_config=VLM_TRANSCODE_CONFIG,
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
            "text": _annotation_prompt(SUBTASKS),
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
            "media_io_kwargs": {
                "video": {
                    "fps": 1,
                }
            },
            "mm_processor_kwargs": {
                "do_sample_frames": False,
            },
            "response_format": {"type": "json_object"},
        }
    )

    dense_annotation = json.loads(response.text)
    if not isinstance(dense_annotation, Mapping):
        raise ValueError("annotation payload must be a JSON object")
    subtasks = dense_annotation.get("subtasks")
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

    max_frame_index = max(0, int(row.length) - 1)
    fps = int(row.metadata.info.fps)
    return row.update(
        {
            "dense_subtask_names": names,
            "dense_subtask_start_times": start_times,
            "dense_subtask_end_times": end_times,
            "dense_subtask_start_frames": [
                _seconds_to_frame_index(
                    value,
                    fps=fps,
                    max_frame_index=max_frame_index,
                )
                for value in start_times
            ],
            "dense_subtask_end_frames": [
                _seconds_to_frame_index(
                    value,
                    fps=fps,
                    max_frame_index=max_frame_index,
                )
                for value in end_times
            ],
        }
    )


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
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    )
