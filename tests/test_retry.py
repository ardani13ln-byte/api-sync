"""Tests for retry, backoff and error classification."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apisync import (  # noqa: E402
    AuthError,
    Client,
    HttpError,
    RateLimitError,
    RetryPolicy,
    TransportError,
    _parse_retry_after,
)
from helpers import LocalServer, ScriptedTransport, transport_error  # noqa: E402


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class TestRetryPolicy(unittest.TestCase):
    def test_backoff_grows_exponentially_up_to_the_ceiling(self):
        policy = RetryPolicy(attempts=6, base=1.0, factor=2.0, max_delay=10.0, jitter=False)
        self.assertEqual(policy.delay_for(1), 1.0)
        self.assertEqual(policy.delay_for(2), 2.0)
        self.assertEqual(policy.delay_for(3), 4.0)
        self.assertEqual(policy.delay_for(4), 8.0)
        self.assertEqual(policy.delay_for(5), 10.0)  # capped

    def test_jitter_stays_within_the_full_window(self):
        policy = RetryPolicy(attempts=5, base=2.0, jitter=True)
        for _ in range(200):
            value = policy.delay_for(2)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 4.0)

    def test_attempt_budget_is_respected(self):
        policy = RetryPolicy(attempts=3)
        self.assertTrue(policy.should_retry(1, 500))
        self.assertTrue(policy.should_retry(2, 500))
        self.assertFalse(policy.should_retry(3, 500))

    def test_non_retryable_status_is_not_retried(self):
        policy = RetryPolicy(attempts=5)
        self.assertFalse(policy.should_retry(1, 400))
        self.assertFalse(policy.should_retry(1, 404))
        self.assertFalse(policy.should_retry(1, 422))

    def test_transport_errors_are_retried(self):
        self.assertTrue(RetryPolicy(attempts=3).should_retry(1, None))


class TestRetryAfterParsing(unittest.TestCase):
    def test_delay_seconds(self):
        self.assertEqual(_parse_retry_after("120"), 120.0)

    def test_http_date_in_the_future(self):
        import time
        from email.utils import formatdate

        value = _parse_retry_after(formatdate(time.time() + 60))
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    def test_http_date_in_the_past_clamps_to_zero(self):
        from email.utils import formatdate

        self.assertEqual(_parse_retry_after(formatdate(0)), 0.0)

    def test_missing_or_garbage_returns_none(self):
        self.assertIsNone(_parse_retry_after(None))
        self.assertIsNone(_parse_retry_after("soon"))
        self.assertIsNone(_parse_retry_after(""))


class TestRetryBehaviour(unittest.TestCase):
    def _client(self, script, **kw):
        sleep = FakeSleep()
        retries: list[tuple[int, float, str]] = []
        client = Client(
            "https://api.test",
            transport=ScriptedTransport(script=script),
            retry=kw.pop("retry", RetryPolicy(attempts=4, base=0.01, factor=2.0, jitter=False)),
            sleep=sleep,
            on_retry=lambda *args: retries.append(args),
            **kw,
        )
        return client, sleep, retries

    def test_transient_500_is_retried_then_succeeds(self):
        client, sleep, retries = self._client([(500, {"e": 1}), (200, {"ok": True})])
        self.assertEqual(client.get("/thing"), {"ok": True})
        self.assertEqual(len(sleep.calls), 1)
        self.assertEqual(len(retries), 1)

    def test_503_retry_budget_is_exhausted_then_raises(self):
        client, sleep, _ = self._client(
            [(503, {"e": 1})] * 4, retry=RetryPolicy(attempts=3, base=0.01, jitter=False)
        )
        with self.assertRaises(HttpError) as ctx:
            client.get("/thing")
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(len(sleep.calls), 2)  # attempts-1 waits

    def test_transport_error_is_retried_then_raises_http(self):
        client, sleep, _ = self._client(
            [transport_error(), transport_error(), (500, {"e": 1})],
            retry=RetryPolicy(attempts=3, base=0.01, jitter=False),
        )
        with self.assertRaises(HttpError):
            client.get("/thing")
        self.assertEqual(len(sleep.calls), 2)

    def test_404_is_not_retried(self):
        client, sleep, _ = self._client([(404, {"error": "missing"})])
        with self.assertRaises(HttpError) as ctx:
            client.get("/thing")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(sleep.calls, [])

    def test_401_raises_auth_error_without_retrying(self):
        client, sleep, _ = self._client([(401, {"error": "bad token"})])
        with self.assertRaises(AuthError):
            client.get("/thing")
        self.assertEqual(sleep.calls, [])

    def test_403_also_raises_auth_error(self):
        client, sleep, _ = self._client([(403, {"error": "forbidden"})])
        with self.assertRaises(AuthError):
            client.get("/thing")
        self.assertEqual(sleep.calls, [])

    def test_429_raises_rate_limit_error_and_honours_retry_after(self):
        # Retry-After: 0 is sent by ScriptedTransport, so the wait must be zero.
        client, sleep, _ = self._client(
            [(429, {"e": 1}), (429, {"e": 2})],
            retry=RetryPolicy(attempts=2, base=5.0, jitter=False),
        )
        with self.assertRaises(RateLimitError) as ctx:
            client.get("/thing")
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(sleep.calls, [0.0])  # header beat the 5s backoff

    def test_request_is_rebuilt_identically_on_retry(self):
        client, _, _ = self._client([(500, {}), (200, {"ok": True})])
        client.get("/thing", params={"page": 2})
        transport = client.transport
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(transport.calls[0].url, transport.calls[1].url)
        self.assertEqual(transport.calls[0].method, "GET")

    def test_post_body_is_replayed_on_retry(self):
        client, _, _ = self._client([(503, {}), (200, {"id": 5})])
        self.assertEqual(client.post("/orders", {"sku": "A"}), {"id": 5})
        first, second = client.transport.calls
        self.assertEqual(first.body, second.body)
        self.assertEqual(first.method, "POST")


class TestRetryAgainstRealServer(unittest.TestCase):
    """The retry loop over a real socket, not a double."""

    def test_real_503_then_success(self):
        with LocalServer() as server:
            server.route("GET", "/flaky")(server.flaky(2, {"ok": True}))
            # route() registers and returns the handler; do not call it again.
            client = Client(
                server.base_url,
                retry=RetryPolicy(attempts=4, base=0.01, jitter=False),
                sleep=lambda _: None,
            )
            self.assertEqual(client.get("/flaky"), {"ok": True})
        self.assertEqual(len(server.requests), 3)

    def test_real_404_raises_immediately(self):
        with LocalServer() as server:
            client = Client(server.base_url, sleep=lambda _: None)
            with self.assertRaises(HttpError) as ctx:
                client.get("/missing")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(len(server.requests), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)