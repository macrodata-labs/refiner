from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from refiner.inference.internal.transport import (
    AiohttpAPIClient,
    post_json_to_api,
)
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.subtask_annotation.profile import DomainProfile


class CountPriorResult(BaseModel):
    """Auditable result from a learned partitioner or precomputed segments."""

    status: Literal["ok", "empty", "unavailable", "invalid"]
    count: int | None = Field(default=None, ge=1)
    segments: list[dict[str, Any]] = Field(default_factory=list)
    domain_id: str
    profile_version: str
    profile_hash: str
    model_artifact: str
    backend: str
    latency_ms: float = Field(ge=0.0)
    issue: str | None = None


def count_prior_from_segments(
    *,
    profile: DomainProfile,
    segments_column: str,
    output_column: str = "partitioner_segment_count",
    result_column: str = "partitioner_count_result",
) -> Callable[[Row], Row]:
    """Build a positive count prior from persisted domain-partitioner segments."""

    _validate_profile(profile)
    _validate_columns(
        segments_column=segments_column,
        output_column=output_column,
        result_column=result_column,
    )

    @describe_builtin(
        "robotics:count_prior_from_segments",
        profile=profile.to_dict(),
        profile_hash=profile.profile_hash,
        segments_column=segments_column,
        output_column=output_column,
        result_column=result_column,
    )
    def _count(row: Row) -> Row:
        started = time.perf_counter()
        try:
            raw_segments = row[segments_column]
        except KeyError as exc:
            raise ValueError(
                f"partitioner segments column {segments_column!r} is missing"
            ) from exc
        try:
            segments = _normalize_partitioner_segments(raw_segments)
        except ValueError as exc:
            result = _result(
                profile=profile,
                status="invalid",
                segments=[],
                backend="precomputed-partitioner",
                started=started,
                issue=str(exc),
            )
            return row.update(
                {
                    output_column: None,
                    result_column: result.model_dump(mode="json"),
                }
            )
        status: Literal["ok", "empty"] = "ok" if segments else "empty"
        result = _result(
            profile=profile,
            status=status,
            segments=segments,
            backend="precomputed-partitioner",
            started=started,
        )
        return row.update(
            {
                output_column: len(segments) or None,
                result_column: result.model_dump(mode="json"),
            }
        )

    return _count


def partitioner_count_prior(
    *,
    profile: DomainProfile,
    endpoint: str,
    video_url_column: str,
    data_hash_column: str = "data_hash",
    token_env: str = "REFINER_PARTITIONER_TOKEN",
    output_column: str = "partitioner_segment_count",
    segments_column: str = "partitioner_segments",
    result_column: str = "partitioner_count_result",
    timeout_s: float = 120.0,
    max_concurrent_requests: int = 8,
    on_unavailable: Literal["mark", "raise"] = "mark",
) -> Callable[[Row], Any]:
    """Call a learned domain partitioner and write its segment count.

    The service contract is ``POST {endpoint}`` with ``video_url`` and
    ``data_hash``. It must return ``status`` and scored ``start_s``/``end_s``
    segments. Authentication is resolved lazily from ``token_env`` and never
    serialized into the pipeline plan.
    """

    _validate_profile(profile)
    _validate_columns(
        segments_column=segments_column,
        output_column=output_column,
        result_column=result_column,
    )
    for value, name in (
        (video_url_column, "video_url_column"),
        (data_hash_column, "data_hash_column"),
        (token_env, "token_env"),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and > 0")
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    if on_unavailable not in {"mark", "raise"}:
        raise ValueError("on_unavailable must be 'mark' or 'raise'")

    base_url, endpoint_path = _split_endpoint(endpoint)
    client: AiohttpAPIClient | None = None

    def _ensure_client() -> AiohttpAPIClient:
        nonlocal client
        if client is None:
            token = os.environ.get(token_env)
            if not token:
                raise RuntimeError(
                    f"partitioner bearer token is missing from {token_env}"
                )
            client = AiohttpAPIClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout_s=timeout_s,
                max_connections=max_concurrent_requests,
            )
        return client

    @describe_builtin(
        "robotics:partitioner_count_prior",
        profile=profile.to_dict(),
        profile_hash=profile.profile_hash,
        endpoint=endpoint,
        video_url_column=video_url_column,
        data_hash_column=data_hash_column,
        token_env=token_env,
        output_column=output_column,
        segments_column=segments_column,
        result_column=result_column,
        timeout_s=timeout_s,
        max_concurrent_requests=max_concurrent_requests,
        on_unavailable=on_unavailable,
    )
    async def _partition(row: Row) -> Row:
        started = time.perf_counter()
        try:
            video_url = str(row[video_url_column]).strip()
            data_hash = str(row[data_hash_column]).strip()
        except KeyError as exc:
            raise ValueError(
                f"partitioner input column {exc.args[0]!r} is missing"
            ) from exc
        if not video_url or not data_hash:
            raise ValueError("partitioner video_url and data_hash must be non-empty")
        if urlsplit(video_url).scheme not in {"http", "https"}:
            raise ValueError("partitioner video_url must be an http(s) URL")

        try:
            response = await post_json_to_api(
                _ensure_client(),
                endpoint_path,
                {"video_url": video_url, "data_hash": data_hash},
                operation="domain partitioning",
                max_retries=2,
            )
            value = response.value
            if not isinstance(value, Mapping):
                raise ValueError("partitioner response must be a JSON object")
            if value.get("status") != "ok":
                message = str(
                    value.get("error") or "partitioner returned non-ok status"
                )
                raise RuntimeError(message)
            segments = _normalize_partitioner_segments(value.get("segments"))
        except Exception as exc:
            if on_unavailable == "raise":
                raise
            status: Literal["unavailable", "invalid"] = (
                "invalid" if isinstance(exc, ValueError) else "unavailable"
            )
            result = _result(
                profile=profile,
                status=status,
                segments=[],
                backend="learned-partitioner-service",
                started=started,
                issue=f"{type(exc).__name__}: partitioner request failed",
            )
            return row.update(
                {
                    output_column: None,
                    segments_column: None,
                    result_column: result.model_dump(mode="json"),
                }
            )

        status: Literal["ok", "empty"] = "ok" if segments else "empty"
        result = _result(
            profile=profile,
            status=status,
            segments=segments,
            backend="learned-partitioner-service",
            started=started,
        )
        return row.update(
            {
                output_column: len(segments) or None,
                segments_column: segments,
                result_column: result.model_dump(mode="json"),
            }
        )

    async def _close() -> None:
        nonlocal client
        if client is not None:
            current = client
            client = None
            await current.close()

    setattr(_partition, "aclose", _close)
    return _partition


def _validate_profile(profile: DomainProfile) -> None:
    if not isinstance(profile, DomainProfile):
        raise TypeError("profile must be a DomainProfile")
    if profile.model_artifact is None:
        raise ValueError("count priors require a profile with model_artifact")


def _validate_columns(
    *,
    segments_column: str,
    output_column: str,
    result_column: str,
) -> None:
    values = {
        "segments_column": segments_column,
        "output_column": output_column,
        "result_column": result_column,
    }
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if len(set(values.values())) != len(values):
        raise ValueError("count-prior output columns must be distinct")


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute http(s) URL")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("endpoint must not contain credentials, query, or fragment")
    path = parsed.path.strip("/")
    if not path:
        raise ValueError("endpoint must include a request path")
    return f"{parsed.scheme}://{parsed.netloc}", path


def _normalize_partitioner_segments(value: Any) -> list[dict[str, Any]]:
    if value is None:
        raise ValueError("partitioner segments are missing")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("partitioner segments must be a list")
    segments = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"partitioner segment[{index}] must be an object")
        try:
            start_s = _number(item.get("start_s", item.get("start_sec")))
            end_s = _number(item.get("end_s", item.get("end_sec")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"partitioner segment[{index}] timestamps must be numeric"
            ) from exc
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise ValueError(
                f"partitioner segment[{index}] must have finite increasing timestamps"
            )
        segment: dict[str, Any] = {
            "start_sec": round(start_s, 3),
            "end_sec": round(end_s, 3),
        }
        score = item.get("score")
        if score is not None:
            try:
                score_value = _number(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"partitioner segment[{index}] score must be numeric"
                ) from exc
            if not math.isfinite(score_value):
                raise ValueError(f"partitioner segment[{index}] score must be finite")
            segment["score"] = score_value
        segments.append(segment)
    segments.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return segments


def _number(value: object) -> float:
    if not isinstance(value, (str, bytes, int, float)):
        raise TypeError("value must be numeric")
    return float(value)


def _result(
    *,
    profile: DomainProfile,
    status: Literal["ok", "empty", "unavailable", "invalid"],
    segments: list[dict[str, Any]],
    backend: str,
    started: float,
    issue: str | None = None,
) -> CountPriorResult:
    assert profile.model_artifact is not None
    return CountPriorResult(
        status=status,
        count=len(segments) or None,
        segments=segments,
        domain_id=profile.domain_id,
        profile_version=profile.version,
        profile_hash=profile.profile_hash,
        model_artifact=profile.model_artifact,
        backend=backend,
        latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
        issue=issue,
    )


__all__ = [
    "CountPriorResult",
    "count_prior_from_segments",
    "partitioner_count_prior",
]
