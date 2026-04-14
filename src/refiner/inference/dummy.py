from __future__ import annotations

import atexit
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

_ACTIVE_DUMMY_SERVERS: set["_DummyRequestServer"] = set()
_ACTIVE_DUMMY_SERVERS_LOCK = threading.Lock()


@dataclass(eq=False, slots=True)
class _DummyRequestServer:
    _server: ThreadingHTTPServer
    _thread: threading.Thread

    @classmethod
    def start(
        cls,
        *,
        host: str,
        port: int,
        response_text: str,
    ) -> "_DummyRequestServer":
        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                    self._write_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": f"unsupported path: {self.path}"},
                    )
                    return
                try:
                    payload = self._read_payload()
                except ValueError as err:
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": str(err)},
                    )
                    return

                model = payload.get("model", "dummy-local")
                if self.path == "/v1/chat/completions":
                    body = {
                        "id": "dummy-chat-completion",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response_text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                else:
                    body = {
                        "id": "dummy-completion",
                        "object": "text_completion",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "text": response_text,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                self._write_json(HTTPStatus.OK, body)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def _read_payload(self) -> dict[str, Any]:
                raw_length = self.headers.get("Content-Length", "0").strip() or "0"
                try:
                    content_length = int(raw_length)
                except ValueError as err:
                    raise ValueError("content-length must be an integer") from err
                raw_body = (
                    self.rfile.read(content_length) if content_length > 0 else b"{}"
                )
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as err:
                    raise ValueError("request body must be valid JSON") from err
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                return payload

            def _write_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
                raw_body = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw_body)))
                self.end_headers()
                self.wfile.write(raw_body)

        server = ThreadingHTTPServer((host, port), _Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="refiner-dummy-request-server",
            daemon=True,
        )
        thread.start()
        instance = cls(_server=server, _thread=thread)
        with _ACTIVE_DUMMY_SERVERS_LOCK:
            _ACTIVE_DUMMY_SERVERS.add(instance)
        return instance

    @property
    def base_url(self) -> str:
        address = cast(tuple[str, int], self._server.server_address)
        host, port = address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)
        with _ACTIVE_DUMMY_SERVERS_LOCK:
            _ACTIVE_DUMMY_SERVERS.discard(self)


def _close_all_dummy_servers() -> None:
    with _ACTIVE_DUMMY_SERVERS_LOCK:
        servers = list(_ACTIVE_DUMMY_SERVERS)
    for server in servers:
        server.close()


atexit.register(_close_all_dummy_servers)
