"""api_sync - a REST client that behaves when the other side is unhealthy.

The hard part of talking to someone else's API is not the happy path, it is
what happens when they return 429 at 2am, when their cursor pagination
silently truncates your results, and when a webhook retry makes you process
the same order twice. Everything here exists to make those cases explicit.

Zero runtime dependencies: the standard library ships everything needed
(urllib, json, hashlib, time, email for HMAC).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__version__ = "1.0.0"

__all__ = [
    "ApiError",
    "AuthError",
    "Client",
    "HttpError",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "NullIdempotencyStore",
    "Page",
    "RateLimitError",
    "RetryPolicy",
    "Transport",
    "UrllibTransport",
    "sign_payload",
    "verify_signature",
]


class ApiError(Exception):
    """Base class for every failure this client raises."""


class TransportError(ApiError):
    """The request never produced a response (DNS, TCP, TLS, timeout)."""


class HttpError(ApiError):
    """The server answered with a non-2xx status."""

    def __init__(self, status: int, url: str, body: str = "", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")
        self.status = status
        self.url = url
        self.body = body
        self.headers = dict(headers or {})

    @property
    def is_retryable(self) -> bool:
        return self.status in (408, 425, 429, 500, 502, 503, 504)


class RateLimitError(HttpError):
    """HTTP 429. Carries ``retry_after`` when the server tells us how long."""


class AuthError(HttpError):
    """HTTP 401 or 403. Never retried: a bad token stays bad."""


@dataclass
class RetryPolicy:
    """Bounded exponential backoff with full jitter.

    Full jitter (``uniform(0, backoff)``) rather than fixed intervals because
    synchronised retries are how a recovering API gets knocked over again.
    """

    attempts: int = 4
    base: float = 0.5
    factor: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 425, 429, 500, 502, 503, 504})
    )
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number ``attempt`` (1-based)."""
        if retry_after is not None and self.respect_retry_after:
            return max(0.0, min(retry_after, self.max_delay))
        raw = self.base * (self.factor ** max(0, attempt - 1))
        raw = min(raw, self.max_delay)
        return random.uniform(0, raw) if self.jitter else raw

    def should_retry(self, attempt: int, status: int | None) -> bool:
        if attempt >= self.attempts:
            return False
        return status is None or status in self.retry_statuses


@dataclass
class Page:
    """One page of results plus whatever cursor metadata came with it."""

    items: list[Any]
    next_cursor: str | None = None
    next_page_token: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class IdempotencyStore:
    """Tracks keys already processed so a retry cannot double-apply a write."""

    def seen(self, key: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def remember(self, key: str, result: Any = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def result(self, key: str) -> Any:  # pragma: no cover
        return None


class NullIdempotencyStore(IdempotencyStore):
    """Records nothing. The default, so opt-in stays explicit."""

    def seen(self, key: str) -> bool:
        return False

    def remember(self, key: str, result: Any = None) -> None:
        return None

    def result(self, key: str) -> Any:
        return None


class InMemoryIdempotencyStore(IdempotencyStore):
    """Bounded in-memory store. Fine for a single process and a test suite."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: dict[str, Any] = {}
        self._max = max_entries

    def seen(self, key: str) -> bool:
        return key in self._entries

    def remember(self, key: str, result: Any = None) -> None:
        if len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)))  # FIFO eviction
        self._entries[key] = result

    def result(self, key: str) -> Any:
        return self._entries.get(key)


class Transport:
    """Performs one HTTP request. The seam that makes this client testable.

    Real code uses :class:`UrllibTransport`. Tests use a scripted double, so
    retry and pagination logic is verified without a network or a sleep.
    """

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, dict[str, str], bytes]:  # pragma: no cover - interface
        raise NotImplementedError


class UrllibTransport(Transport):
    """Standard-library transport, no dependencies."""

    def __init__(self, opener: Any | None = None) -> None:
        self._opener = opener or urllib.request.build_opener()

    def request(self, method, url, headers, body, timeout):
        req = urllib.request.Request(url, data=body, method=method.upper())
        for key, value in headers.items():
            req.add_header(key, value)
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as exc:
            payload = b""
            try:
                payload = exc.read()
            except Exception:  # noqa: BLE001 - body is best effort
                pass
            return exc.code, dict(exc.headers.items() if exc.headers else {}), payload
        except urllib.error.URLError as exc:
            raise TransportError(f"{type(exc.reason).__name__} for {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransportError(f"timeout after {timeout}s for {url}") from exc


class Client:
    """A REST client with retries, uniform pagination and idempotent writes.

    Usage::

        client = Client("https://api.example.com", token=os.environ["TOKEN"])
        for order in client.paginate("/orders", cursor_param="cursor"):
            sync(order)
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        auth_header: str = "Authorization",
        token_prefix: str = "Bearer ",
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        retry: RetryPolicy | None = None,
        transport: Transport | None = None,
        idempotency: IdempotencyStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = "api-sync/1.0",
        on_retry: Callable[[int, float, str], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.auth_header = auth_header
        self.token_prefix = token_prefix
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        self.transport = transport or UrllibTransport()
        self.idempotency = idempotency or NullIdempotencyStore()
        self._sleep = sleep
        self.on_retry = on_retry
        self.default_headers: dict[str, str] = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            **dict(headers or {}),
        }

    # -- public surface ----------------------------------------------------

    def get(self, path: str, params: Mapping[str, Any] | None = None, **kw: Any) -> Any:
        return self.request("GET", path, params=params, **kw)

    def post(self, path: str, payload: Any = None, *, idempotency_key: str | None = None, **kw: Any) -> Any:
        return self.request("POST", path, payload=payload, idempotency_key=idempotency_key, **kw)

    def put(self, path: str, payload: Any = None, **kw: Any) -> Any:
        return self.request("PUT", path, payload=payload, **kw)

    def patch(self, path: str, payload: Any = None, **kw: Any) -> Any:
        return self.request("PATCH", path, payload=payload, **kw)

    def delete(self, path: str, **kw: Any) -> Any:
        return self.request("DELETE", path, **kw)

    def paginate(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        mode: str = "cursor",
        cursor_param: str = "cursor",
        page_param: str = "page",
        items_path: str | None = None,
        cursor_path: str | None = None,
        page_size: int | None = None,
        max_pages: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[Any]:
        """Yield items across pages, transparently handling both styles.

        ``mode='cursor'`` follows a next-cursor until it is absent or null.
        ``mode='page'`` increments an integer page number until a page comes
        back short of ``page_size``. Either way the loop terminates on a
        bounded condition, because "until the API says stop" is how an
        integration ends up looping at 3am.
        """
        base_params = dict(params or {})
        if page_size is not None:
            base_params.setdefault("limit", page_size)
        cursor: str | None = None
        page_number = 1
        seen_cursors: set[str] = set()
        produced = 0
        pages_read = 0

        while True:
            call_params = dict(base_params)
            if mode == "cursor" and cursor is not None:
                call_params[cursor_param] = cursor
            elif mode == "page":
                call_params[page_param] = page_number

            raw = self.request("GET", path, params=call_params, _raw=True)
            pages_read += 1
            items = self._extract_items(raw, items_path)
            for item in items:
                yield item
                produced += 1
                if max_items is not None and produced >= max_items:
                    return

            if max_pages is not None and pages_read >= max_pages:
                return

            if mode == "cursor":
                cursor = self._extract_cursor(raw, cursor_path)
                if not cursor:
                    return
                if cursor in seen_cursors:
                    # A server echoing the same cursor forever is a real
                    # failure mode seen in the wild; stop instead of hanging.
                    return
                seen_cursors.add(cursor)
            else:
                if page_size is None or len(items) < page_size:
                    return
                page_number += 1

    # -- internals ---------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
        idempotency_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        _raw: bool = False,
    ) -> Any:
        url = self._build_url(path, params)
        body: bytes | None = None
        merged = dict(self.default_headers)
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            merged["Content-Type"] = "application/json"
        if self.token:
            merged[self.auth_header] = f"{self.token_prefix}{self.token}"
        merged.update(headers or {})

        cache_key = idempotency_key
        if cache_key and method.upper() in ("POST", "PUT", "PATCH") and self.idempotency.seen(cache_key):
            return self.idempotency.result(cache_key)
        if cache_key:
            merged["Idempotency-Key"] = cache_key

        status, response_headers, raw_body = self._send_with_retries(
            method, url, merged, body
        )

        try:
            decoded = json.loads(raw_body.decode("utf-8")) if raw_body.strip() else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            if status >= 400:
                raise HttpError(status, url, raw_body[:500].decode("utf-8", "replace"), response_headers)
            return raw_body

        if _raw:
            return decoded if isinstance(decoded, dict) else {"items": decoded}
        if status >= 400:
            raise self._error_for(status, url, raw_body, response_headers)
        if cache_key:
            self.idempotency.remember(cache_key, decoded)
        return decoded

    def _send_with_retries(self, method, url, headers, body):
        attempt = 0
        while True:
            attempt += 1
            try:
                status, resp_headers, raw = self.transport.request(
                    method, url, headers, body, self.timeout
                )
            except TransportError as exc:
                if not self.retry.should_retry(attempt, None):
                    raise
                delay = self.retry.delay_for(attempt)
                self._notify_retry(attempt, delay, f"transport error: {exc}")
                self._sleep(delay)
                continue

            if status in (401, 403):
                raise AuthError(status, url, raw[:300].decode("utf-8", "replace"), resp_headers)
            if status in self.retry.retry_statuses and self.retry.should_retry(attempt, status):
                retry_after = _parse_retry_after(resp_headers.get("Retry-After") or resp_headers.get("retry-after"))
                delay = self.retry.delay_for(attempt, retry_after)
                self._notify_retry(attempt, delay, f"HTTP {status}")
                self._sleep(delay)
                continue
            return status, resp_headers, raw

    def _error_for(self, status, url, raw, headers) -> HttpError:
        text = raw[:500].decode("utf-8", "replace")
        cls = RateLimitError if status == 429 else HttpError
        return cls(status, url, text, headers)

    def _notify_retry(self, attempt: int, delay: float, reason: str) -> None:
        if self.on_retry:
            self.on_retry(attempt, delay, reason)

    def _build_url(self, path: str, params: Mapping[str, Any] | None) -> str:
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}/{path.lstrip('/')}"
        if not params:
            return url
        clean = {k: v for k, v in params.items() if v is not None}
        if not clean:
            return url
        return f"{url}?{urllib.parse.urlencode(clean, doseq=True)}"

    @staticmethod
    def _extract_items(raw: Any, items_path: str | None) -> list[Any]:
        data = raw
        if items_path:
            for key in items_path.split("."):
                if not isinstance(data, Mapping):
                    return []
                data = data.get(key)
            return list(data or []) if isinstance(data, list) else []
        if isinstance(data, list):
            return data
        if isinstance(data, Mapping):
            for key in ("items", "data", "results", "records", "rows", "values"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        return []

    @staticmethod
    def _extract_cursor(raw: Any, cursor_path: str | None) -> str | None:
        """Read the next cursor, honouring an explicit dotted path first.

        With ``cursor_path="data.link"`` the navigated value *is* the cursor,
        so it is returned as-is. Without a path, the usual envelope keys are
        probed in order of how common they are in real APIs.
        """
        if not isinstance(raw, Mapping):
            return None
        if cursor_path:
            data: Any = raw
            for key in cursor_path.split("."):
                if not isinstance(data, Mapping):
                    return None
                data = data.get(key)
            return str(data) if data else None
        for key in ("next_cursor", "cursor", "nextCursor", "next", "after", "page_token"):
            value = raw.get(key)
            if value:
                return str(value)
        return None


def _parse_retry_after(value: str | None) -> float | None:
    """Read ``Retry-After`` in both its delay-seconds and HTTP-date forms."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        when = parsedate_to_datetime(text)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:  # noqa: BLE001 - unparseable header is simply ignored
        return None


def sign_payload(secret: str, payload: str | bytes, timestamp: str | None = None) -> str:
    """HMAC-SHA256 signature for webhooks, in the Stripe-style ``t=...,v1=...`` form."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    stamp = timestamp if timestamp is not None else str(int(time.time()))
    signed = f"{stamp}.".encode() + payload
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={stamp},v1={digest}"


def verify_signature(
    secret: str, payload: str | bytes, header: str, *, tolerance: float = 300.0
) -> bool:
    """Verify a webhook signature with a replay window, using constant-time compare."""
    parts = [piece.split("=", 1) for piece in header.split(",") if "=" in piece]
    stamps = [value for key, value in parts if key == "t"]
    provided = [value for key, value in parts if key == "v1"]
    if not stamps or not provided:
        return False
    for stamp in stamps:
        if tolerance:
            try:
                if abs(time.time() - float(stamp)) > tolerance:
                    continue
            except (TypeError, ValueError):
                continue
        expected = sign_payload(secret, payload, stamp).split("v1=", 1)[1]
        # Providers ship several v1 values during key rotation; accept if any
        # matches, and compare each in constant time.
        if any(hmac.compare_digest(expected, candidate) for candidate in provided):
            return True
    return False