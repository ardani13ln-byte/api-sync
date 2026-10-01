"""Tests for idempotent writes: the double-charge bug."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apisync import (  # noqa: E402
    Client,
    InMemoryIdempotencyStore,
    NullIdempotencyStore,
    RetryPolicy,
)
from helpers import LocalServer, ScriptedTransport  # noqa: E402


class TestIdempotencyStore(unittest.TestCase):
    def test_memory_store_roundtrip(self):
        store = InMemoryIdempotencyStore()
        self.assertFalse(store.seen("k"))
        store.remember("k", {"id": 9})
        self.assertTrue(store.seen("k"))
        self.assertEqual(store.result("k"), {"id": 9})

    def test_memory_store_evicts_beyond_capacity(self):
        store = InMemoryIdempotencyStore(max_entries=3)
        for i in range(5):
            store.remember(f"k{i}", i)
        self.assertFalse(store.seen("k0"))
        self.assertTrue(store.seen("k4"))

    def test_null_store_never_remembers(self):
        store = NullIdempotencyStore()
        store.remember("k", 1)
        self.assertFalse(store.seen("k"))
        self.assertIsNone(store.result("k"))


class TestIdempotentWrites(unittest.TestCase):
    def _client(self, script, store):
        return Client(
            "https://api.test",
            transport=ScriptedTransport(script=script),
            sleep=lambda _: None,
            retry=RetryPolicy(attempts=1),
            idempotency=store,
        )

    def test_repeated_key_short_circuits_without_a_second_request(self):
        store = InMemoryIdempotencyStore()
        client = self._client([(201, {"id": 7})], store)
        first = client.post("/orders", {"sku": "A"}, idempotency_key="order-1")
        second = client.post("/orders", {"sku": "A"}, idempotency_key="order-1")
        self.assertEqual(first, {"id": 7})
        self.assertEqual(second, {"id": 7})
        self.assertEqual(client.transport.call_count, 1)

    def test_key_header_is_sent_to_the_server(self):
        store = InMemoryIdempotencyStore()
        client = self._client([(201, {"id": 1})], store)
        client.post("/orders", {"sku": "A"}, idempotency_key="order-1")
        self.assertEqual(client.transport.calls[0].headers["Idempotency-Key"], "order-1")

    def test_different_keys_are_separate_requests(self):
        store = InMemoryIdempotencyStore()
        client = self._client([(201, {"id": 1}), (201, {"id": 2})], store)
        self.assertEqual(client.post("/o", {}, idempotency_key="a"), {"id": 1})
        self.assertEqual(client.post("/o", {}, idempotency_key="b"), {"id": 2})
        self.assertEqual(client.transport.call_count, 2)

    def test_failed_write_is_not_remembered(self):
        # If we cached on error, a retry after a 500 could never succeed.
        store = InMemoryIdempotencyStore()
        client = self._client([(500, {"e": 1})], store)
        with self.assertRaises(Exception):
            client.post("/orders", {"sku": "A"}, idempotency_key="order-1")
        self.assertFalse(store.seen("order-1"))

    def test_retried_write_is_still_remembered_once(self):
        store = InMemoryIdempotencyStore()
        client = self._client([(201, {"id": 42})], store)
        client.post("/orders", {}, idempotency_key="k")
        client.post("/orders", {}, idempotency_key="k")
        self.assertEqual(client.transport.call_count, 1)

    def test_get_is_never_deduplicated(self):
        store = InMemoryIdempotencyStore()
        client = self._client([(200, {"n": 1}), (200, {"n": 2})], store)
        self.assertEqual(client.get("/x"), {"n": 1})
        self.assertEqual(client.get("/x"), {"n": 2})
        self.assertEqual(client.transport.call_count, 2)

    def test_no_store_means_every_call_hits_the_network(self):
        client = self._client([(201, {"id": 1}), (201, {"id": 1})], NullIdempotencyStore())
        client.post("/o", {}, idempotency_key="k")
        client.post("/o", {}, idempotency_key="k")
        self.assertEqual(client.transport.call_count, 2)

    def test_idempotency_survives_a_real_http_retry(self):
        """The realistic failure: a 503 then a success, replayed once."""
        with LocalServer() as server:
            state = {"n": 0}

            def handler(record):
                state["n"] += 1
                if state["n"] == 1:
                    return 503, {"error": "try again"}
                return 201, {"id": state["n"]}

            server.route("POST", "/orders")(handler)
            store = InMemoryIdempotencyStore()
            client = Client(
                server.base_url,
                retry=RetryPolicy(attempts=3, base=0.01, jitter=False),
                sleep=lambda _: None,
                idempotency=store,
            )
            result = client.post("/orders", {"sku": "A"}, idempotency_key="order-1")
            self.assertEqual(result, {"id": 2})
            # Second call is served from the store, server never sees it again.
            self.assertEqual(client.post("/orders", {"sku": "A"}, idempotency_key="order-1"), {"id": 2})
            self.assertEqual(state["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)