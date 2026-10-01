"""Test doubles: scripted transports and a real local HTTP server.

Two levels are covered deliberately:

* :class:`ScriptedTransport` asserts on exact requests and injects failures, so
  retry and pagination logic is verified with zero waiting and no network.
* :class:`LocalServer` runs an actual ``http.server`` on a random port, so the
  wire behaviour (headers, query strings, status codes, connection reuse) is
  verified for real.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from apisync import Transport, TransportError


@dataclass
class RecordedCall:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None

    @property
    def path(self) -> str:
        return urllib.parse.urlsplit(self.url).path

    @property
    def query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.url).query)

    @property
    def json_body(self) -> Any:
        return json.loads(self.body.decode()) if self.body else None


@dataclass
class ScriptedTransport(Transport):
    """Replays a fixed list of responses and records what was requested.

    Each entry is either a ``(status, body)`` tuple or a raised exception.
    Running out of scripted responses raises loudly rather than hanging.
    """

    script: list[Any]
    calls: list[RecordedCall] = field(default_factory=list)
    _cursor: int = 0

    def request(self, method, url, headers, body, timeout):
        self.calls.append(
            RecordedCall(method.upper(), url, dict(headers), body)
        )
        if self._cursor >= len(self.script):
            raise AssertionError(
                f"unscripted request: {method} {url} "
                f"(script had {len(self.script)} response(s))"
            )
        item = self.script[self._cursor]
        self._cursor += 1
        if isinstance(item, Exception):
            raise item
        status, payload = item
        raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        headers_out = {"Content-Type": "application/json"}
        if status == 429:
            headers_out["Retry-After"] = "0"
        return status, headers_out, raw

    @property
    def call_count(self) -> int:
        return len(self.calls)


class LocalServer:
    """A real HTTP server for end-to-end tests.

    Routes are ``(method, path) -> callable(handler) -> tuple[status, body]``
    so a handler can inspect the live request. Lifecycle is handled by the
    context manager, and the socket is threaded so a blocking read does not
    stall the test.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Any] = {}
        self.requests: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def route(self, method: str, path: str):
        def decorator(fn):
            self.routes[(method.upper(), path)] = fn
            return fn

        return decorator

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "LocalServer":
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence test output
                return

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                record = {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                }
                server_self.requests.append(record)
                handler = server_self.routes.get((self.command, urllib.parse.urlsplit(self.path).path))
                if handler is None:
                    self._respond(404, {"error": "no route"})
                    return
                try:
                    status, payload = handler(record)
                except Exception as exc:  # noqa: BLE001 - surface as a 500
                    status, payload = 500, {"error": str(exc)}
                self._respond(status, payload)

            def _respond(self, status: int, payload: Any) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _handle
            do_POST = _handle
            do_PUT = _handle
            do_PATCH = _handle
            do_DELETE = _handle

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    @staticmethod
    def flaky(failures: int, then: tuple[int, Any]) -> Any:
        """Handler that returns 503 ``failures`` times, then returns ``then``.

        ``then`` is either a ``(status, payload)`` pair or a bare payload,
        which is served with 200.
        """
        state = {"n": 0}
        final = then if isinstance(then, tuple) else (200, then)

        def handler(_record: dict[str, Any]):
            state["n"] += 1
            if state["n"] <= failures:
                return 503, {"error": "temporarily unavailable"}
            return final

        return handler


def transport_error(message: str = "connection reset") -> TransportError:
    return TransportError(message)