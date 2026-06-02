from __future__ import annotations

import pytest

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client.api import _http_error_message, request_json


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        json_payload: object | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        reason: str = "HTTP error",
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text
        self.headers = headers or {}
        self.reason = reason

    def json(self) -> object:
        if self._json_payload is None:
            raise ValueError("no JSON")
        return self._json_payload


def test_http_error_message_uses_reason_phrase_for_html_body() -> None:
    resp = _FakeResponse(
        404,
        text="<!doctype html><html><body><h1>404</h1><p>Not Found</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
        reason="Not Found",
    )

    assert _http_error_message(resp) == "Not Found"


def test_http_error_message_strips_control_chars() -> None:
    resp = _FakeResponse(
        500,
        json_payload={"error": "\x1b[31mboom\x9b[0m"},
    )

    assert _http_error_message(resp) == "[31mboom[0m"


def test_request_json_raises_credentials_error_for_401(monkeypatch) -> None:
    def fake_request(*args, **kwargs) -> _FakeResponse:  # noqa: ANN002, ANN003
        return _FakeResponse(
            401,
            json_payload={"error": "Invalid API key"},
        )

    monkeypatch.setattr("refiner.platform.client.api.requests.request", fake_request)

    with pytest.raises(MacrodataCredentialsError, match="Invalid API key") as exc:
        request_json(
            method="GET",
            path="/api/me",
            api_key="md_bad",
            base_url="https://macrodata.co",
        )

    assert exc.value.missing is False
