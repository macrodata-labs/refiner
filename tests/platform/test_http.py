from __future__ import annotations

import httpx
import pytest

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client.api import _http_error_message, request_json


def test_http_error_message_uses_reason_phrase_for_html_body() -> None:
    resp = httpx.Response(
        404,
        text="<!doctype html><html><body><h1>404</h1><p>Not Found</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", "https://macrodata.co/api/me"),
    )

    assert _http_error_message(resp) == "Not Found"


def test_http_error_message_strips_control_chars() -> None:
    resp = httpx.Response(
        500,
        json={"error": "\x1b[31mboom\x1b[0m"},
        request=httpx.Request("GET", "https://macrodata.co/api/me"),
    )

    assert _http_error_message(resp) == "[31mboom[0m"


def test_request_json_raises_credentials_error_for_401(monkeypatch) -> None:
    def fake_request(*args, **kwargs) -> httpx.Response:  # noqa: ANN002, ANN003
        return httpx.Response(
            401,
            json={"error": "Invalid API key"},
            request=httpx.Request("GET", "https://macrodata.co/api/me"),
        )

    monkeypatch.setattr("refiner.platform.client.api.httpx.request", fake_request)

    with pytest.raises(MacrodataCredentialsError, match="Invalid API key") as exc:
        request_json(
            method="GET",
            path="/api/me",
            api_key="md_bad",
            base_url="https://macrodata.co",
        )

    assert exc.value.missing is False
