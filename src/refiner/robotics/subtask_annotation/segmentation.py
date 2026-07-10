from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.robotics.subtask_annotation.profile import DomainProfile
from refiner.robotics.subtask_annotation.result import (
    SegmentationProvenance,
    SegmentationResult,
    TimelineValidation,
    validate_subtask_segments,
)
from refiner.robotics.subtask_annotation.utils import (
    GEMINI_BLOCK_NONE_SAFETY_SETTINGS,
    _blocked_prompt_reason,
    _log_on_overlapping_segments,
    _resolve_video,
    timestamped_contact_sheets,
)
from refiner.worker.context import logger

if TYPE_CHECKING:
    from refiner.inference.generate_text import GenerateTextFn
    from refiner.inference.internal.response import InferenceResponse
    from refiner.video import VideoSource


_SEGMENTATION_CONTACT_SHEET_QUALITY = 95
_SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY = 70
_SEGMENTATION_SAMPLE_SEC = 0.5
_SEGMENTATION_FRAME_WIDTH = 224
_SEGMENTATION_FRAMES_PER_SHEET = 20
_SEGMENTATION_COLUMNS = 5
_SEGMENTATION_BACKEND = "vlm-contact-sheets"
_SEGMENTATION_PROMPT_VERSION = "2"

_SUBTASK_ANNOTATION_PROMPT_TEMPLATE = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Global rules:
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- Apply the supplied segmentation policy consistently; do not invent a different level of granularity.
"""


class _SubtaskSegment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str


class _SubtaskAnnotationResult(BaseModel):
    segments: list[_SubtaskSegment]


def subtask_annotation(
    *,
    profile: DomainProfile,
    provider: InferenceProvider = GoogleEndpointProvider(model="gemini-3.5-flash"),
    video_key: str,
    output_column: str = "predicted_subtasks",
    result_column: str = "subtask_annotation_result",
    count_prior_column: str | None = None,
    temperature: float = 0.1,
    thinking_budget: int | None = None,
    on_blocked_prompt: Literal["mark", "raise"] = "mark",
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that annotates robotics episode subtasks.

    Args:
        profile: Explicit domain and segmentation-policy definition. The complete
            profile is serialized into the pipeline plan and hashed into results.
        provider: Text-generation provider used to run the vision-language model.
        video_key: Video key to annotate.
        output_column: Row column receiving validated segments. A blocked or wholly
            invalid result writes ``None`` instead of looking like a valid empty video.
        result_column: Row column receiving status, raw/validated segments, issues,
            and reproducibility provenance.
        count_prior_column: Optional row column containing a positive integer segment
            count from an upstream, domain-matched partitioner.
        temperature: Generation temperature passed to the provider request.
        thinking_budget: Optional Google thinking-token budget. When supplied, this
            is pinned in both the request and reproducibility hash.
        on_blocked_prompt: ``"mark"`` writes an explicit blocked result; ``"raise"``
            propagates the provider error.
        max_concurrent_requests: Maximum provider requests allowed at once per worker.
    """

    from refiner.inference import generate_text

    if not isinstance(profile, DomainProfile):
        raise TypeError("profile must be a DomainProfile")
    if not video_key.strip():
        raise ValueError("video_key must be non-empty")
    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if not result_column.strip():
        raise ValueError("result_column must be non-empty")
    if result_column == output_column:
        raise ValueError("result_column must differ from output_column")
    if count_prior_column is not None and not count_prior_column.strip():
        raise ValueError("count_prior_column must be non-empty when provided")
    if on_blocked_prompt not in {"mark", "raise"}:
        raise ValueError("on_blocked_prompt must be 'mark' or 'raise'")
    if thinking_budget is not None:
        if isinstance(thinking_budget, bool) or thinking_budget <= 0:
            raise ValueError("thinking_budget must be a positive integer")

    render_config = _render_config(_SEGMENTATION_CONTACT_SHEET_QUALITY)
    config_hash = _config_hash(
        profile=profile,
        provider=provider,
        temperature=temperature,
        thinking_budget=thinking_budget,
        render_config=render_config,
    )

    @describe_builtin(
        "robotics:subtask_annotation",
        profile=profile.to_dict(),
        profile_hash=profile.profile_hash,
        provider=provider.to_builtin_args(),
        video_key=video_key,
        output_column=output_column,
        result_column=result_column,
        count_prior_column=count_prior_column,
        temperature=temperature,
        thinking_budget=thinking_budget,
        on_blocked_prompt=on_blocked_prompt,
        max_concurrent_requests=max_concurrent_requests,
        config_hash=config_hash,
    )
    async def _annotate_subtasks(
        row: Row,
        generate_text: "GenerateTextFn",
    ) -> MapResult:
        started = time.perf_counter()
        if not isinstance(row, RoboticsRow):
            raise TypeError("subtask_annotation expects RoboticsRow inputs")

        video = _resolve_video(row, video_key)
        duration_s = _video_duration_s(row, video)
        count_prior = _count_prior(row, count_prior_column)
        content = await _subtask_annotation_content(
            video=video,
            tasks=row.tasks,
            profile=profile,
            count_prior=count_prior,
            quality=_SEGMENTATION_CONTACT_SHEET_QUALITY,
        )
        fallback_used = False
        try:
            response = await _request_subtask_annotation(
                generate_text=generate_text,
                content=content,
                temperature=temperature,
                thinking_budget=thinking_budget,
            )
        except RuntimeError as exc:
            block_reason = _blocked_prompt_reason(exc)
            if block_reason is None:
                raise
            fallback_used = True
            try:
                fallback_content = await _subtask_annotation_content(
                    video=video,
                    tasks=row.tasks,
                    profile=profile,
                    count_prior=count_prior,
                    quality=_SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY,
                )
                response = await _request_subtask_annotation(
                    generate_text=generate_text,
                    content=fallback_content,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                )
            except RuntimeError as fallback_exc:
                fallback_block_reason = _blocked_prompt_reason(fallback_exc)
                if fallback_block_reason is None or on_blocked_prompt == "raise":
                    raise
                logger.warning(
                    "subtask annotation provider blocked episode {}: {}",
                    row.episode_id,
                    fallback_block_reason,
                )
                result = SegmentationResult(
                    status="blocked",
                    segments=[],
                    raw_segments=[],
                    issues=["provider blocked both primary and fallback requests"],
                    block_reason=fallback_block_reason,
                    provenance=_provenance(
                        profile=profile,
                        provider=provider,
                        config_hash=config_hash,
                        count_prior=count_prior,
                        fallback_used=True,
                        quality=_SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY,
                        latency_ms=_elapsed_ms(started),
                        usage={},
                    ),
                )
                return row.update(
                    {
                        output_column: None,
                        result_column: result.model_dump(mode="json"),
                    }
                )

        parsed = response.object
        if not isinstance(parsed, _SubtaskAnnotationResult):
            raise TypeError("subtask_annotation expected a structured response object")
        validation = validate_subtask_segments(
            [segment.model_dump() for segment in parsed.segments],
            video_duration_s=duration_s,
        )
        _log_on_overlapping_segments(validation.segments)
        result = _segmentation_result(
            validation=validation,
            profile=profile,
            provider=provider,
            config_hash=config_hash,
            count_prior=count_prior,
            fallback_used=fallback_used,
            latency_ms=_elapsed_ms(started),
            response=response,
        )
        output_segments: list[dict[str, Any]] | None = result.segments
        if result.status == "invalid":
            output_segments = None
        return row.update(
            {
                output_column: output_segments,
                result_column: result.model_dump(mode="json"),
            }
        )

    return generate_text(
        fn=_annotate_subtasks,
        provider=provider,
        max_concurrent_requests=max_concurrent_requests,
    )


async def _request_subtask_annotation(
    *,
    generate_text: "GenerateTextFn",
    content: list[dict[str, Any]],
    temperature: float,
    thinking_budget: int | None,
) -> Any:
    messages = cast(list[Message], [{"role": "user", "content": content}])
    google_options: dict[str, Any] = {
        "safetySettings": GEMINI_BLOCK_NONE_SAFETY_SETTINGS,
    }
    if thinking_budget is not None:
        google_options["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return await generate_text(
        messages=messages,
        schema=_SubtaskAnnotationResult,
        provider_options={
            "google": google_options,
        },
        maxRetries=4,
        temperature=temperature,
    )


async def _subtask_annotation_content(
    *,
    video: VideoSource,
    tasks: list[str],
    profile: DomainProfile,
    count_prior: int | None,
    quality: int,
) -> list[dict[str, Any]]:
    file_parts: list[dict[str, Any]] = []

    async for sheet in timestamped_contact_sheets(
        video,
        sample_sec=_SEGMENTATION_SAMPLE_SEC,
        frame_width=_SEGMENTATION_FRAME_WIDTH,
        frames_per_sheet=_SEGMENTATION_FRAMES_PER_SHEET,
        columns=_SEGMENTATION_COLUMNS,
        quality=quality,
    ):
        file_parts.append(
            {"type": "file", "mediaType": sheet.media_type, "data": sheet.data}
        )

    if not file_parts:
        raise ValueError("video produced no frames")

    text = (
        f"{_SUBTASK_ANNOTATION_PROMPT_TEMPLATE}\n"
        f"Domain: {profile.domain_id}@{profile.version}\n"
        f"{profile.policy.prompt_section()}\n"
    )
    if count_prior is not None:
        text += (
            "\nCount prior from the domain-matched partitioner:\n"
            f"- Return {count_prior} segments. Use this only to select policy depth; "
            "place boundaries at the semantic events visible in the video.\n"
        )
    instruction = "; ".join(task for task in tasks if task.strip())
    if instruction:
        text += f"\nEpisode instruction: {instruction}\n"
    return [{"type": "text", "text": text}, *file_parts]


def _segmentation_result(
    *,
    validation: TimelineValidation,
    profile: DomainProfile,
    provider: InferenceProvider,
    config_hash: str,
    count_prior: int | None,
    fallback_used: bool,
    latency_ms: float,
    response: InferenceResponse,
) -> SegmentationResult:
    return SegmentationResult(
        status=validation.status,
        segments=validation.segments,
        raw_segments=validation.raw_segments,
        issues=validation.issues,
        provenance=_provenance(
            profile=profile,
            provider=provider,
            config_hash=config_hash,
            count_prior=count_prior,
            fallback_used=fallback_used,
            quality=(
                _SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY
                if fallback_used
                else _SEGMENTATION_CONTACT_SHEET_QUALITY
            ),
            latency_ms=latency_ms,
            usage=dict(response.usage),
        ),
    )


def _provenance(
    *,
    profile: DomainProfile,
    provider: InferenceProvider,
    config_hash: str,
    count_prior: int | None,
    fallback_used: bool,
    quality: int,
    latency_ms: float,
    usage: Mapping[str, Any],
) -> SegmentationProvenance:
    return SegmentationProvenance(
        domain_id=profile.domain_id,
        profile_version=profile.version,
        policy_id=profile.policy.policy_id,
        policy_version=profile.policy.version,
        profile_hash=profile.profile_hash,
        config_hash=config_hash,
        backend=_SEGMENTATION_BACKEND,
        model=provider.model,
        count_prior=count_prior,
        fallback_used=fallback_used,
        render_config=_render_config(quality),
        latency_ms=latency_ms,
        usage=dict(usage),
    )


def _count_prior(row: Row, column: str | None) -> int | None:
    if column is None:
        return None
    try:
        raw = row[column]
    except KeyError as exc:
        raise ValueError(f"count prior column {column!r} is missing") from exc
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("count prior must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("count prior must be a positive integer") from exc
    if value <= 0 or value != raw:
        raise ValueError("count prior must be a positive integer")
    return value


def _video_duration_s(row: RoboticsRow, video: VideoSource) -> float | None:
    duration = getattr(video, "duration_s", None)
    if duration is not None:
        duration = float(duration)
        if math.isfinite(duration) and duration > 0:
            return duration
    from_sec = getattr(video, "from_timestamp_s", None)
    to_sec = getattr(video, "to_timestamp_s", None)
    if to_sec is not None:
        duration = float(to_sec) - float(from_sec or 0.0)
        if math.isfinite(duration) and duration > 0:
            return duration
    if row.fps is not None and row.num_frames > 0:
        duration = row.num_frames / float(row.fps)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def _render_config(quality: int) -> dict[str, int | float]:
    return {
        "sample_sec": _SEGMENTATION_SAMPLE_SEC,
        "frame_width": _SEGMENTATION_FRAME_WIDTH,
        "frames_per_sheet": _SEGMENTATION_FRAMES_PER_SHEET,
        "columns": _SEGMENTATION_COLUMNS,
        "quality": quality,
    }


def _config_hash(
    *,
    profile: DomainProfile,
    provider: InferenceProvider,
    temperature: float,
    thinking_budget: int | None,
    render_config: Mapping[str, int | float],
) -> str:
    payload = {
        "backend": _SEGMENTATION_BACKEND,
        "prompt_version": _SEGMENTATION_PROMPT_VERSION,
        "profile": profile.to_dict(),
        "provider": provider.to_builtin_args(),
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "render_config": dict(render_config),
        "fallback_render_config": _render_config(
            _SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


__all__ = ["subtask_annotation"]
