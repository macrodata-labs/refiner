from __future__ import annotations

import httpx

from refiner.platform.http import _http_error_message


def test_http_error_message_uses_reason_phrase_for_html_body() -> None:
    resp = httpx.Response(
        404,
        text="<!doctype html><html><body><h1>404</h1><p>Not Found</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", "https://macrodata.co/api/me"),
    )

    assert _http_error_message(resp) == "Not Found"
