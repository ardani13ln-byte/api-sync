"""Tests for headers, auth, payloads and URL construction."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apisync import Client, RetryPolicy  # noqa: E402
from helpers import LocalServer, ScriptedTransport  # noqa: E402


def make(script, **kw):
    return Client(
        "https://api.test",
        transport=ScriptedTransport(script=script),
        sleep=lambda _: None,
        retry=RetryPolicy(attempts=1),
        **kw,
    )


class TestAuth(unittest.TestCase):
    def test_bearer_token_header(self):
        client = make([(200, {})], token="secret")
        client.get("/x")
        self.assertEqual(client.transport.calls[0].headers["Authorization"], "Bearer secret")

    def test_custom_auth_scheme(self):
        client = make([(200, {})], token="abc", token_prefix="Token ")
        client.get("/x")
        self.assertEqual(client.transport.calls[0].headers["Authorization"], "Token abc")

    def test_custom_header_name(self):
        client = make([(200, {})], token="key", auth_header="X-API-Key", token_prefix="")
        client.get("/x")
        self.assertEqual(client.transport.calls[0].headers["X-API-Key"], "key")

    def test_no_token_means_no_auth_header(self):
        client = make([(200, {})])
        client.get("/x")
        self.assertNotIn("Authorization", client.transport.calls[0].headers)

    def test_per_request_header_overrides_default(self):
        client = make([(200, {})], headers={"X-Tenant": "eu"})
        client.get("/x", headers={"X-Tenant": "us"})
        self.assertEqual(client.transport.calls[0].headers["X-Tenant"], "us")


class TestUrlBuilding(unittest.TestCase):
    def test_relative_path_is_joined(self):
        client = make([(200, {})])
        client.get("orders")
        self.assertEqual(client.transport.calls[0].url, "https://api.test/orders")

    def test_absolute_url_is_used_verbatim(self):
        client = make([(200, {})])
        client.get("https://other.test/thing")
        self.assertEqual(client.transport.calls[0].url, "https://other.test/thing")

    def test_params_are_urlencoded(self):
        client = make([(200, {})])
        client.get("/x", params={"q": "a b", "limit": 10})
        self.assertIn("q=a+b", client.transport.calls[0].url)
        self.assertIn("limit=10", client.transport.calls[0].url)

    def test_none_params_are_dropped(self):
        client = make([(200, {})])
        client.get("/x", params={"a": None, "b": 1})
        self.assertNotIn("a=", client.transport.calls[0].url)

    def test_list_params_use_doseq(self):
        client = make([(200, {})])
        client.get("/x", params={"id": [1, 2, 3]})
        self.assertIn("id=1&id=2&id=3", client.transport.calls[0].url)

    def test_trailing_slash_on_base_is_normalised(self):
        client = Client("https://api.test/", transport=ScriptedTransport(script=[(200, {})]))
        client.get("/x")
        self.assertEqual(client.transport.calls[0].url, "https://api.test/x")


class TestPayloads(unittest.TestCase):
    def test_json_body_is_compact_and_typed(self):
        client = make([(200, {})])
        client.post("/x", {"a": 1, "b": [1, 2]})
        call = client.transport.calls[0]
        self.assertEqual(call.json_body, {"a": 1, "b": [1, 2]})
        self.assertEqual(call.headers["Content-Type"], "application/json")
        self.assertNotIn(b" ", call.body)  # compact separators

    def test_get_sends_no_body(self):
        client = make([(200, {})])
        client.get("/x")
        self.assertIsNone(client.transport.calls[0].body)

    def test_no_content_type_without_payload(self):
        client = make([(200, {})])
        client.post("/x")
        self.assertNotIn("Content-Type", client.transport.calls[0].headers)

    def test_empty_response_body_becomes_none(self):
        client = Client(
            "https://api.test",
            transport=ScriptedTransport(script=[(204, "")]),
            sleep=lambda _: None,
        )
        self.assertIsNone(client.delete("/x"))

    def test_verb_helpers_hit_the_right_methods(self):
        client = make([(200, {})] * 5)
        client.get("/x")
        client.post("/x", {})
        client.put("/x", {})
        client.patch("/x", {})
        client.delete("/x")
        self.assertEqual([c.method for c in client.transport.calls],
                         ["GET", "POST", "PUT", "PATCH", "DELETE"])

    def test_non_json_success_body_is_returned_raw(self):
        class RawTransport(ScriptedTransport):
            def request(self, method, url, headers, body, timeout):
                self.calls.append(type("C", (), {"headers": dict(headers)})())
                return 200, {"Content-Type": "text/csv"}, b"id,name\n1,Ana\n"

        client = Client("https://api.test", transport=RawTransport(script=[]))
        self.assertEqual(client.get("/export.csv"), b"id,name\n1,Ana\n")

    def test_default_accept_and_user_agent(self):
        client = make([(200, {})])
        client.get("/x")
        headers = client.transport.calls[0].headers
        self.assertEqual(headers["Accept"], "application/json")
        self.assertTrue(headers["User-Agent"].startswith("api-sync/"))


class TestAgainstRealServer(unittest.TestCase):
    def test_headers_and_body_arrive_intact(self):
        with LocalServer() as server:
            server.route("POST", "/echo")(lambda record: (200, {"body": record["body"].decode()}))
            client = Client(server.base_url, token="t", headers={"X-Trace": "abc"})
            result = client.post("/echo", {"hello": "world"}, headers={"X-Extra": "1"})
            self.assertEqual(result["body"], '{"hello":"world"}')
            sent = server.requests[0]["headers"]
            self.assertEqual(sent["Authorization"], "Bearer t")
            self.assertEqual(sent["X-Trace"], "abc")
            self.assertEqual(sent["X-Extra"], "1")

    def test_query_string_reaches_the_server(self):
        with LocalServer() as server:
            server.route("GET", "/search")(lambda record: (200, {"path": record["path"]}))
            client = Client(server.base_url)
            self.assertEqual(client.get("/search", params={"q": "x y"})["path"],
                             "/search?q=x+y")


if __name__ == "__main__":
    unittest.main(verbosity=2)