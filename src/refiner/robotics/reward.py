from __future__ import annotations

import base64
import io
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.lerobot_format import LeRobotRow
from refiner.video.decode import DecodedVideoFrame

if TYPE_CHECKING:
    from refiner.inference.generate_pooling import GeneratePoolingFn
else:
    GeneratePoolingFn = Callable[[Mapping[str, Any]], Any]

_DEFAULT_ROBOMETER_MODEL = "robometer/Robometer-4B"
_PROGRESS_TOKEN = "<|prog_token|>"
_MAX_DEBUG_TEXT_CHARS = 1000

logger = logging.getLogger(__name__)


TaskSource = str | Callable[[LeRobotRow], str]


def reward_score(
    *,
    model: str = _DEFAULT_ROBOMETER_MODEL,
    video_key: str | None = None,
    task: TaskSource | None = None,
    max_frames: int = 8,
    output_column: str = "reward_score",
    success_column: str = "robometer_success",
    error_column: str = "robometer_error",
    debug_column: str = "robometer_debug",
    max_concurrent_requests: int = 256,
    fail_soft: bool = True,
) -> Callable[[Row], Any]:
    """Return an async map function that scores LeRobot episodes with Robometer."""

    if not model.strip():
        raise ValueError("model must be non-empty")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if not success_column.strip():
        raise ValueError("success_column must be non-empty")
    if fail_soft and not error_column.strip():
        raise ValueError("error_column must be non-empty when fail_soft=True")
    if fail_soft and not debug_column.strip():
        raise ValueError("debug_column must be non-empty when fail_soft=True")

    from refiner.inference import (
        InferenceAPICallError,
        InferenceRetryError,
        generate_pooling,
    )
    from refiner.inference.providers import VLLMProvider

    provider = VLLMProvider(model=model, config="throughput")

    async def _score_episode(
        row: Row,
        generate_pooling_request: GeneratePoolingFn,
    ) -> MapResult:
        if not isinstance(row, LeRobotRow):
            raise TypeError("reward_score expects rows from read_lerobot(...)")

        selected_video_key = _resolve_video_key(row, video_key)
        frames = await _sample_video_frames(
            row,
            video_key=selected_video_key,
            max_frames=max_frames,
        )
        task_text = _resolve_task_text(row, task)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _robometer_progress_prompt(task_text)}
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _frame_data_url(frame)},
                }
            )
            content.append({"type": "text", "text": _PROGRESS_TOKEN})

        payload = {
            "task": "token_classify",
            "use_activation": False,
            "chat_template_kwargs": {
                "add_vision_id": True,
                "enable_thinking": False,
                "fps": 1,
            },
            "mm_processor_kwargs": {"do_resize": False},
            "messages": [{"role": "user", "content": content}],
        }
        try:
            response = await generate_pooling_request(payload)
            token_logits = _extract_progress_token_logits(response, len(frames))
            progress = [expected_progress(row) for row in token_logits]
            success = [sigmoid(float(row[10])) for row in token_logits]
        except (InferenceAPICallError, InferenceRetryError) as exc:
            if not fail_soft:
                raise
            debug = _robometer_failure_debug(
                exc,
                row=row,
                video_key=selected_video_key,
                task_text=task_text,
                frames=frames,
                payload=payload,
            )
            logger.warning(
                "Robometer scoring failed for episode %s video_key=%s: %s",
                row.episode_index,
                selected_video_key,
                debug["error"],
            )
            logger.debug("Robometer failure debug: %s", debug)
            row.log_throughput("robometer_failed_rows", 1, unit="rows")
            return row.update(
                {
                    output_column: [],
                    success_column: [],
                    error_column: debug["error"],
                    debug_column: debug,
                }
            )
        return row.update(
            {
                output_column: progress,
                success_column: success,
            }
        )

    return generate_pooling(
        fn=_score_episode,
        provider=provider,
        max_concurrent_requests=max_concurrent_requests,
    )


def expected_progress(logits: Sequence[float]) -> float:
    if len(logits) < 10:
        raise ValueError("Robometer token row must contain at least 10 logits")
    probabilities = softmax([float(value) for value in logits[:10]])
    return sum(
        (index / 9.0) * probability for index, probability in enumerate(probabilities)
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("softmax values must be non-empty")
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0.0:
        raise ValueError("softmax values produced a non-positive denominator")
    return [value / total for value in exps]


def _extract_token_logits(response: Mapping[str, Any]) -> list[list[float]]:
    data = response.get("data")
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        data = data[0].get("data")
    if (
        not isinstance(data, Sequence)
        or isinstance(data, (str, bytes, bytearray))
        or not data
    ):
        raise ValueError("Robometer pooling response did not contain token logits")
    logits = [list(row) for row in data]
    if any(
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes, bytearray))
        or len(row) < 13
        for row in logits
    ):
        raise ValueError("Robometer token rows must contain at least 13 logits")
    return [[float(item) for item in row] for row in logits]


def _extract_progress_token_logits(
    response: Mapping[str, Any], expected_count: int
) -> list[list[float]]:
    logits = _extract_token_logits(response)
    if len(logits) < expected_count:
        raise ValueError(
            "Robometer pooling response returned fewer progress token rows than "
            "sampled frames"
        )
    return logits[-expected_count:]


def _robometer_failure_debug(
    exc: BaseException,
    *,
    row: LeRobotRow,
    video_key: str,
    task_text: str,
    frames: Sequence[DecodedVideoFrame],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    frame_indexes = [int(frame.index) for frame in frames]
    return {
        "error": _exception_summary(exc),
        "exception_type": type(exc).__name__,
        "exception_chain": _exception_chain(exc),
        "episode_index": row.episode_index,
        "row_length": row.length,
        "video_key": video_key,
        "video_keys": list(row.videos),
        "task_text": task_text,
        "sampled_frame_count": len(frames),
        "sampled_frame_indexes": frame_indexes,
        "payload": _pooling_payload_debug(payload),
    }


def _exception_summary(exc: BaseException) -> str:
    return _truncate_debug_text(f"{type(exc).__name__}: {exc}")


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": _truncate_debug_text(str(current)),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _pooling_payload_debug(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    content: Sequence[Any] = ()
    if (
        isinstance(messages, Sequence)
        and not isinstance(messages, str | bytes | bytearray)
        and messages
        and isinstance(messages[0], Mapping)
    ):
        raw_content = messages[0].get("content")
        if isinstance(raw_content, Sequence) and not isinstance(
            raw_content, str | bytes | bytearray
        ):
            content = raw_content

    image_url_lengths: list[int] = []
    text_lengths: list[int] = []
    content_types: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            content_types.append(type(part).__name__)
            continue
        part_type = part.get("type")
        content_types.append(str(part_type))
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, Mapping):
                url = image_url.get("url")
                if isinstance(url, str):
                    image_url_lengths.append(len(url))
        elif part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_lengths.append(len(text))

    return {
        "task": payload.get("task"),
        "use_activation": payload.get("use_activation"),
        "chat_template_kwargs": payload.get("chat_template_kwargs"),
        "mm_processor_kwargs": payload.get("mm_processor_kwargs"),
        "message_count": len(messages) if isinstance(messages, Sequence) else None,
        "content_count": len(content),
        "content_types": content_types,
        "image_count": len(image_url_lengths),
        "image_url_length_min": min(image_url_lengths) if image_url_lengths else None,
        "image_url_length_max": max(image_url_lengths) if image_url_lengths else None,
        "image_url_length_total": sum(image_url_lengths),
        "text_count": len(text_lengths),
        "text_length_total": sum(text_lengths),
    }


def _truncate_debug_text(text: str, *, limit: int = _MAX_DEBUG_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... <truncated {len(text) - limit} chars>"


def _resolve_video_key(row: LeRobotRow, video_key: str | None) -> str:
    video_keys = list(row.videos)
    if video_key is not None:
        if video_key not in video_keys:
            raise ValueError(
                f"episode {row.episode_index} is missing video key {video_key!r}"
            )
        return video_key
    if not video_keys:
        raise ValueError(f"episode {row.episode_index} has no videos")
    return video_keys[0]


def _resolve_task_text(row: LeRobotRow, task: TaskSource | None) -> str:
    if isinstance(task, str):
        value = task
    elif task is None:
        value = "; ".join(row.tasks)
    else:
        value = cast(Callable[[LeRobotRow], str], task)(row)
    if not value.strip():
        raise ValueError(f"episode {row.episode_index} has no task text")
    return value.strip().removesuffix(".")


def _robometer_progress_prompt(task_text: str) -> str:
    return (
        f"The task for the robot is '{task_text}'. Given the trajectory video, "
        "predict the task progress at each frame, how far along the robot is "
        "towards completing the task, a float between 0 and 1, where 0 is the "
        "starting state and 1 is when the task is completed. If the robot is "
        "not performing the same task, predict 0 progress."
    )


async def _sample_video_frames(
    row: LeRobotRow,
    *,
    video_key: str,
    max_frames: int,
) -> list[DecodedVideoFrame]:
    target_indexes = set(_sample_indexes(row.length, max_frames=max_frames))
    frames: list[DecodedVideoFrame] = []
    async for frame in row.videos[video_key].iter_frames():
        if frame.index in target_indexes:
            frames.append(frame)
        if len(frames) == len(target_indexes):
            break
    if not frames:
        raise ValueError(
            f"episode {row.episode_index} video {video_key!r} has no frames"
        )
    return frames


def _sample_indexes(length: int, *, max_frames: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if length <= max_frames:
        return tuple(range(length))
    if max_frames == 1:
        return (length - 1,)
    return tuple(
        sorted(
            {
                round(index * float(length - 1) / float(max_frames - 1))
                for index in range(max_frames)
            }
        )
    )


def _frame_data_url(frame: DecodedVideoFrame) -> str:
    image_bytes = _encode_png(frame)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _encode_png(frame: DecodedVideoFrame) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    image = (
        frame.frame.to_image()
        .convert("RGB")
        .resize(
            (256, 256),
            Image.Resampling.BICUBIC,
        )
    )
    image.save(output, format="PNG")
    return output.getvalue()


__all__ = [
    "reward_score",
    "expected_progress",
    "sigmoid",
    "softmax",
]
