from __future__ import annotations

import asyncio
import email.utils
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import aiohttp

T = TypeVar("T")

_DEFAULT_MAX_RETRIES = 2
_INITIAL_RETRY_DELAY_SECONDS = 2.0
_BACKOFF_FACTOR = 2.0
_MAX_REASONABLE_RETRY_DELAY_SECONDS = 60.0
_MAX_ERROR_BODY_CHARS = 4096
_MAX_ERROR_STRING_CHARS = 256
_MAX_ERROR_SEQUENCE_ITEMS = 8


@dataclass(frozen=True, slots=True)
class APIResponse:
    value: Any
    response_headers: Mapping[str, str]
    raw_value: Any | None = None


class InferenceAPICallError(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        url: str,
        request_body: Any,
        status_code: int | None = None,
        response_headers: Mapping[str, str] | None = None,
        response_body: str | None = None,
        data: Any | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.request_body = _summarize_error_value(request_body)
        self.status_code = status_code
        self.response_headers = dict(response_headers or {})
        self.response_body = _truncate_error_text(response_body)
        self.data = _summarize_error_value(data)
        self.is_retryable = (
            _is_retryable_status(status_code) if is_retryable is None else is_retryable
        )


class InferenceRetryError(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        reason: str,
        errors: list[BaseException],
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.errors = errors


@dataclass(slots=True)
class AiohttpAPIClient:
    base_url: str
    headers: Mapping[str, str]
    timeout_s: float = 600.0
    max_connections: int | None = None
    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)
    _session_loop: asyncio.AbstractEventLoop | None = field(
        default=None, init=False, repr=False
    )

    def _ensure_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        session = self._session
        if (
            session is not None
            and not session.closed
            and self._session_loop is not loop
        ):
            raise RuntimeError(
                "AiohttpAPIClient cannot be reused across event loops while its "
                "session is open; call close() before reusing it on another loop."
            )
        if session is None or session.closed:
            connector_kwargs: dict[str, Any] = {}
            if self.max_connections is not None:
                connector_kwargs["limit"] = self.max_connections
            connector = aiohttp.TCPConnector(**connector_kwargs)
            session = aiohttp.ClientSession(
                connector=connector,
                headers=dict(self.headers),
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                trust_env=True,
            )
            self._session = session
            self._session_loop = loop
        return session

    async def post(self, endpoint_path: str, **kwargs: Any) -> aiohttp.ClientResponse:
        return await self._ensure_session().post(
            _request_url(self, endpoint_path),
            **kwargs,
        )

    async def close(self) -> None:
        session = self._session
        if session is None:
            return
        self._session = None
        self._session_loop = None
        await session.close()


def provider_request_options(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None, dict[str, str] | None]:
    request = dict(payload)
    raw_max_retries = request.pop("__refiner_max_retries", None)
    raw_headers = request.pop("__refiner_headers", None)
    headers = None
    if isinstance(raw_headers, Mapping):
        headers = {
            str(key): str(value)
            for key, value in raw_headers.items()
            if value is not None
        }
    if raw_max_retries is None:
        return request, None, headers
    if not isinstance(raw_max_retries, int):
        raise ValueError("maxRetries must be an integer")
    return request, raw_max_retries, headers


async def post_json_to_api(
    client: AiohttpAPIClient,
    endpoint_path: str,
    payload: Mapping[str, Any],
    *,
    operation: str,
    max_retries: int | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> APIResponse:
    retry = _prepare_retries(max_retries)

    async def _post() -> APIResponse:
        try:
            kwargs: dict[str, Any] = {"json": dict(payload)}
            if extra_headers:
                kwargs["headers"] = dict(extra_headers)
            response = await client.post(endpoint_path, **kwargs)

            return await _handle_json_response(
                response,
                url=_request_url(client, endpoint_path),
                request_body=dict(payload),
                operation=operation,
            )
        except (
            aiohttp.ClientError,
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
        ) as err:
            raise InferenceAPICallError(
                message=f"Cannot connect to API: {type(err).__name__}: {err}",
                url=_request_url(client, endpoint_path),
                request_body=dict(payload),
                is_retryable=True,
            ) from err

    return await retry(_post)


async def _handle_json_response(
    response: aiohttp.ClientResponse,
    *,
    url: str,
    request_body: Mapping[str, Any],
    operation: str,
) -> APIResponse:
    response_headers = _response_headers(response)
    status_code = response.status
    if status_code >= 400:
        response_body = await response.text()
        try:
            data = await response.json(content_type=None)
        except (ValueError, aiohttp.ContentTypeError):
            data = None
        message = _error_message(
            operation=operation,
            status_code=status_code,
            status_text=response.reason or "",
            data=data,
            response_body=response_body,
        )
        raise InferenceAPICallError(
            message=message,
            url=url,
            request_body=request_body,
            status_code=status_code,
            response_headers=response_headers,
            response_body=response_body,
            data=data,
        )

    try:
        value = await response.json(content_type=None)
    except (ValueError, aiohttp.ContentTypeError) as err:
        raise InferenceAPICallError(
            message="Invalid JSON response",
            url=url,
            request_body=request_body,
            status_code=status_code,
            response_headers=response_headers,
            response_body=await response.text(),
            is_retryable=False,
        ) from err
    return APIResponse(
        value=value,
        response_headers=response_headers,
        raw_value=value,
    )


def _prepare_retries(
    max_retries: int | None,
) -> Callable[[Callable[[], Awaitable[T]]], Awaitable[T]]:
    if max_retries is not None:
        if not isinstance(max_retries, int):
            raise ValueError("maxRetries must be an integer")
        if max_retries < 0:
            raise ValueError("maxRetries must be >= 0")
    resolved_max_retries = _DEFAULT_MAX_RETRIES if max_retries is None else max_retries

    async def _retry(fn: Callable[[], Awaitable[T]]) -> T:
        delay_seconds = _INITIAL_RETRY_DELAY_SECONDS
        errors: list[BaseException] = []
        while True:
            try:
                return await fn()
            except Exception as err:
                if resolved_max_retries == 0:
                    raise
                errors.append(err)
                try_number = len(errors)
                if try_number > resolved_max_retries:
                    raise InferenceRetryError(
                        message=(
                            f"Failed after {try_number} attempts. Last error: {err}"
                        ),
                        reason="maxRetriesExceeded",
                        errors=errors,
                    ) from err
                if not _is_retryable_error(err):
                    if try_number == 1:
                        raise
                    raise InferenceRetryError(
                        message=(
                            "Failed after "
                            f"{try_number} attempts with non-retryable error: {err}"
                        ),
                        reason="errorNotRetryable",
                        errors=errors,
                    ) from err
                await asyncio.sleep(
                    _retry_delay_seconds(
                        err,
                        exponential_backoff_delay=delay_seconds,
                    )
                )
                delay_seconds *= _BACKOFF_FACTOR

    return _retry


def _is_retryable_error(err: BaseException) -> bool:
    return isinstance(err, InferenceAPICallError) and err.is_retryable


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code is not None and (
        status_code in {408, 409, 429} or status_code >= 500
    )


def _retry_delay_seconds(
    err: BaseException,
    *,
    exponential_backoff_delay: float,
) -> float:
    if not isinstance(err, InferenceAPICallError):
        return exponential_backoff_delay
    retry_after_ms = err.response_headers.get("retry-after-ms")
    if retry_after_ms is not None:
        delay = _parse_float_seconds(retry_after_ms, scale=0.001)
        if delay is not None and _is_reasonable_retry_delay(
            delay, exponential_backoff_delay
        ):
            return delay

    retry_after = err.response_headers.get("retry-after")
    if retry_after is not None:
        delay = _parse_float_seconds(retry_after, scale=1.0)
        if delay is None:
            try:
                parsed_date = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                parsed_date = None
            delay = parsed_date.timestamp() - time.time() if parsed_date else None
        if delay is not None and _is_reasonable_retry_delay(
            delay, exponential_backoff_delay
        ):
            return delay

    return exponential_backoff_delay


def _parse_float_seconds(value: str, *, scale: float) -> float | None:
    try:
        return float(value) * scale
    except ValueError:
        return None


def _is_reasonable_retry_delay(
    delay: float | None,
    exponential_backoff_delay: float,
) -> bool:
    return (
        delay is not None
        and not delay != delay
        and 0 <= delay
        and (
            delay < _MAX_REASONABLE_RETRY_DELAY_SECONDS
            or delay < exponential_backoff_delay
        )
    )


def _response_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
    headers = getattr(response, "headers", {})
    return {str(key).lower(): str(value) for key, value in dict(headers).items()}


def _request_url(client: AiohttpAPIClient, endpoint_path: str) -> str:
    return f"{client.base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _error_message(
    *,
    operation: str,
    status_code: int,
    status_text: str,
    data: Any | None,
    response_body: str,
) -> str:
    detail = _extract_error_message(data) or response_body.strip() or status_text
    detail = _truncate_error_text(detail)
    message = f"{operation} request failed with HTTP {status_code}"
    return f"{message}: {detail}" if detail else message


def _extract_error_message(data: Any | None) -> str | None:
    if not isinstance(data, Mapping):
        return None
    error = data.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message
    message = data.get("message")
    if isinstance(message, str):
        return message
    return None


def _summarize_error_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _summarize_error_value(item)
            for key, item in list(value.items())[:_MAX_ERROR_SEQUENCE_ITEMS]
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = [
            _summarize_error_value(item) for item in value[:_MAX_ERROR_SEQUENCE_ITEMS]
        ]
        if len(value) > _MAX_ERROR_SEQUENCE_ITEMS:
            items.append(f"<{len(value) - _MAX_ERROR_SEQUENCE_ITEMS} more items>")
        return items
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__} {len(value)} bytes>"
    if isinstance(value, str):
        return _truncate_error_text(value, limit=_MAX_ERROR_STRING_CHARS)
    return value


def _truncate_error_text(
    text: str | None,
    *,
    limit: int = _MAX_ERROR_BODY_CHARS,
) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return f"{text[:limit]}... <truncated {len(text) - limit} chars>"


__all__ = [
    "APIResponse",
    "AiohttpAPIClient",
    "InferenceAPICallError",
    "InferenceRetryError",
    "provider_request_options",
    "post_json_to_api",
]
