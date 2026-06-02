from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

import aiohttp
import pytest

import refiner as mdr
from refiner.inference import (
    OpenAIEndpointProvider,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    openai_provider,
    transport_module,
)


class _FakeHTTPResponse:
    def __init__(
        self,
        status_code: int = 200,
        *,
        json: object | None = None,
        headers: Mapping[str, str] | None = None,
        text: str | None = None,
        reason_phrase: str = "",
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.text = text if text is not None else ""
        self.reason_phrase = reason_phrase
        self._json = json

    def json(self) -> object:
        if self._json is None:
            raise ValueError("no JSON")
        return self._json


class _FakeAiohttpSession:
    def __init__(self, *, connector, headers, timeout, trust_env):
        del connector, headers, timeout
        assert trust_env is True
        self.closed = False

    async def post(self, url, **kwargs):
        del url, kwargs
        return _FakeHTTPResponse(
            json={
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        )

    async def close(self) -> None:
        self.closed = True


def _install_fake_aiohttp_session(monkeypatch) -> list[_FakeAiohttpSession]:
    sessions: list[_FakeAiohttpSession] = []

    class _RecordingSession(_FakeAiohttpSession):
        def __init__(self, *, connector, headers, timeout, trust_env):
            super().__init__(
                connector=connector,
                headers=headers,
                timeout=timeout,
                trust_env=trust_env,
            )
            sessions.append(self)

    monkeypatch.setattr(transport_module.aiohttp, "ClientSession", _RecordingSession)
    monkeypatch.setattr(transport_module.aiohttp, "TCPConnector", lambda **_: object())
    monkeypatch.setattr(
        transport_module.aiohttp,
        "ClientTimeout",
        lambda *, total: ("timeout", total),
    )
    return sessions


def test_openai_endpoint_includes_api_key_in_requests(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            seen["base_url"] = str(base_url)
            seen["headers"] = dict(headers)
            seen["max_connections"] = max_connections

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)

    async def _inference_fn(row, generate_text):
        response = await generate_text(raw_payload={"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert seen["max_connections"] == 256


def test_openai_endpoint_preserves_base_url_path_prefix(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            del timeout_s, max_connections
            self.base_url = str(base_url)
            seen["base_url"] = str(base_url)
            seen["headers"] = dict(headers)

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://openrouter.ai/api/v1",
        ).generate(
            {
                "model": "openai/gpt-5.2",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen["base_url"] == "https://openrouter.ai/api"
    assert seen["path"] == "v1/chat/completions"
    assert seen["payload"] == {
        "model": "openai/gpt-5.2",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openai_endpoint_applies_configured_connection_limits(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers
            seen["max_connections"] = max_connections

        async def post(self, path, *, json):
            del path, json
            return _FakeResponse()

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
            max_connections=512,
        ).generate(
            {
                "model": "gpt-test",
                "prompt": "hello",
            }
        )
    )

    assert response.text == "ok"
    assert seen["max_connections"] == 512


def test_inference_map_exposes_client_close_hook(monkeypatch) -> None:
    seen: dict[str, int] = {"closed": 0}

    class _FakeResponse:
        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout_s, max_connections):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            return _FakeResponse()

        async def close(self) -> None:
            seen["closed"] += 1

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(raw_payload={"prompt": "hello"})
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )

    async def _invoke_and_close() -> object:
        result = await infer(DictRow({}))
        close = cast(Callable[[], Awaitable[None]], getattr(infer, "aclose"))
        await close()
        return result

    assert asyncio.run(_invoke_and_close()) == {"output": "ok"}
    assert seen["closed"] == 1


def test_aiohttp_client_reuses_session_on_same_loop(monkeypatch) -> None:
    sessions = _install_fake_aiohttp_session(monkeypatch)

    client = transport_module.AiohttpAPIClient(
        base_url="https://api.example.com",
        headers={},
    )

    async def _call_twice() -> None:
        await transport_module.post_json_to_api(
            client,
            "v1/completions",
            {"model": "gpt-test", "prompt": "hello"},
            operation="generation",
        )
        await transport_module.post_json_to_api(
            client,
            "v1/completions",
            {"model": "gpt-test", "prompt": "hello"},
            operation="generation",
        )
        assert len(sessions) == 1
        assert getattr(sessions[0], "closed") is False
        await client.close()

    asyncio.run(_call_twice())

    assert len(sessions) == 1
    assert getattr(sessions[0], "closed") is True


def test_aiohttp_client_rejects_open_session_on_different_loop(monkeypatch) -> None:
    sessions = _install_fake_aiohttp_session(monkeypatch)

    client = transport_module.AiohttpAPIClient(
        base_url="https://api.example.com",
        headers={},
    )

    async def _call_once() -> None:
        await transport_module.post_json_to_api(
            client,
            "v1/completions",
            {"model": "gpt-test", "prompt": "hello"},
            operation="generation",
        )

    asyncio.run(_call_once())
    with pytest.raises(RuntimeError, match="cannot be reused across event loops"):
        asyncio.run(_call_once())

    assert len(sessions) == 1
    assert getattr(sessions[0], "closed") is False


def test_aiohttp_client_can_be_closed_before_different_loop_reuse(monkeypatch) -> None:
    sessions = _install_fake_aiohttp_session(monkeypatch)

    client = transport_module.AiohttpAPIClient(
        base_url="https://api.example.com",
        headers={},
    )

    async def _call_once() -> None:
        await transport_module.post_json_to_api(
            client,
            "v1/completions",
            {"model": "gpt-test", "prompt": "hello"},
            operation="generation",
        )

    async def _close() -> None:
        await client.close()

    asyncio.run(_call_once())
    asyncio.run(_close())
    asyncio.run(_call_once())

    assert len(sessions) == 2
    assert getattr(sessions[0], "closed") is True
    assert getattr(sessions[1], "closed") is False


def test_openai_endpoint_retries_on_timeout(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] < 3:
                raise TimeoutError("timed out")
            return _FakeResponse()

    async def _fake_sleep(delay: float) -> None:
        seen["sleeps"] += 1
        assert delay in (2.0, 4.0)

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 3, "sleeps": 2}


def test_openai_endpoint_retries_on_connect_error(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] == 1:
                raise ConnectionError("connect failed")
            return _FakeResponse()

    async def _fake_sleep(delay: float) -> None:
        seen["sleeps"] += 1
        assert delay == 2.0

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "prompt": "hello",
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 2, "sleeps": 1}


def test_openai_endpoint_retries_on_remote_protocol_error(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] == 1:
                raise aiohttp.ServerDisconnectedError(
                    "Server disconnected without sending a response."
                )
            return _FakeResponse()

    async def _fake_sleep(delay: float) -> None:
        assert delay == 2.0
        seen["sleeps"] += 1

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "prompt": "hello",
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 2, "sleeps": 1}


def test_openai_endpoint_warns_on_null_chat_content(caplog) -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Thinking Process: still reasoning",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 64},
    }
    response = openai_provider.parse_chat_response(
        raw_response,
        use_chat=True,
    )

    assert response.text == ""
    assert response.finish_reason == "length"
    assert response.usage == {"completion_tokens": 64}
    assert response.raw == raw_response
    message = response.raw["choices"][0]["message"]
    assert message["reasoning"] == "Thinking Process: still reasoning"
    assert (
        "chat completion response had null message.content; returning empty text"
        in caplog.text
    )


def test_openai_endpoint_retries_on_http_503(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    error_response = _FakeHTTPResponse(
        503,
        json={"error": {"message": "Service unavailable"}},
    )
    success_response = _FakeHTTPResponse(
        200,
        json={
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
    )

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] == 1:
                return error_response
            return success_response

    async def _fake_sleep(delay: float) -> None:
        assert delay == 2.0
        seen["sleeps"] += 1

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 2, "sleeps": 1}


def test_openai_endpoint_respects_retry_after_ms(monkeypatch) -> None:
    seen: dict[str, object] = {"calls": 0, "sleeps": []}

    error_response = _FakeHTTPResponse(
        429,
        json={"error": {"message": "Rate limited"}},
        headers={"retry-after-ms": "125"},
    )
    success_response = _FakeHTTPResponse(
        200,
        json={
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
        headers={"x-request-id": "req_123"},
    )

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            seen["calls"] = cast(int, seen["calls"]) + 1
            if seen["calls"] == 1:
                return error_response
            return success_response

    async def _fake_sleep(delay: float) -> None:
        cast(list[float], seen["sleeps"]).append(delay)

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert response.headers["x-request-id"] == "req_123"
    assert seen == {"calls": 2, "sleeps": [0.125]}


def test_openai_endpoint_can_disable_retries(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    response = _FakeHTTPResponse(
        503,
        json={"error": {"message": "Service unavailable"}},
    )

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path
            assert "__refiner_max_retries" not in dict(json)
            seen["calls"] += 1
            return response

    async def _fake_sleep(delay: float) -> None:
        del delay
        seen["sleeps"] += 1

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(
        mdr.inference.InferenceAPICallError,
        match="generation request failed with HTTP 503: Service unavailable",
    ) as err:
        asyncio.run(
            openai_provider._OpenAIEndpointClient(
                base_url="https://api.example.com",
            ).generate(
                {
                    "model": "gpt-test",
                    "prompt": "hello",
                    "__refiner_max_retries": 0,
                }
            )
        )

    assert err.value.status_code == 503
    assert err.value.is_retryable is True
    assert seen == {"calls": 1, "sleeps": 0}


def test_inference_api_errors_store_bounded_payloads(monkeypatch) -> None:
    large_prompt = "x" * 10_000
    large_response = "y" * 10_000
    response = _FakeHTTPResponse(
        503,
        json={"error": {"message": large_response}, "raw": large_response},
        text=large_response,
    )

    class _FakeAsyncClient:
        def __init__(
            self,
            *,
            base_url,
            headers,
            timeout_s,
            max_connections,
        ):
            self.base_url = str(base_url)
            del headers, timeout_s, max_connections

        async def post(self, path, *, json):
            del path, json
            return response

    monkeypatch.setattr(openai_provider, "AiohttpAPIClient", _FakeAsyncClient)

    with pytest.raises(mdr.inference.InferenceAPICallError) as err:
        asyncio.run(
            openai_provider._OpenAIEndpointClient(
                base_url="https://api.example.com",
            ).generate(
                {
                    "model": "gpt-test",
                    "prompt": large_prompt,
                    "__refiner_max_retries": 0,
                }
            )
        )

    assert "<truncated " in err.value.request_body["prompt"]
    assert "<truncated " in str(err.value)
    assert len(str(err.value)) < len(large_response)
    assert len(err.value.request_body["prompt"]) < len(large_prompt)
    assert large_prompt not in err.value.request_body["prompt"]
    assert large_response not in str(err.value)
    assert err.value.response_body is not None
    assert "<truncated " in err.value.response_body
    assert len(err.value.response_body) < len(response.text)
    assert "<truncated " in err.value.data["raw"]
