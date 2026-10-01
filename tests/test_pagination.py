"""Tests for cursor and page pagination, including the pathological cases.

These are the bugs that only show up in production: a server that echoes the
same cursor forever, an off-by-one that drops the last full page, a payload
whose items live under an unusual key.
"""

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


class TestCursorPagination(unittest.TestCase):
    def test_three_pages_yield_every_item_in_order(self):
        client = make([
            (200, {"items": [1, 2], "next_cursor": "c2"}),
            (200, {"items": [3, 4], "next_cursor": "c3"}),
            (200, {"items": [5], "next_cursor": None}),
        ])
        self.assertEqual(list(client.paginate("/items")), [1, 2, 3, 4, 5])

    def test_cursor_is_sent_on_follow_up_requests(self):
        client = make([
            (200, {"items": [1], "next_cursor": "abc"}),
            (200, {"items": [2], "next_cursor": None}),
        ])
        list(client.paginate("/items"))
        calls = client.transport.calls
        self.assertNotIn("cursor", calls[0].query)
        self.assertEqual(calls[1].query["cursor"], ["abc"])

    def test_empty_next_cursor_string_stops(self):
        client = make([(200, {"items": [1], "next_cursor": ""})])
        self.assertEqual(list(client.paginate("/items")), [1])

    def test_repeated_cursor_terminates_instead_of_looping(self):
        # Server bug: echoes the same cursor. Without a guard this loops forever.
        client = make([(200, {"items": [1], "next_cursor": "same"})] * 3)
        items = list(client.paginate("/items"))
        self.assertEqual(len(items), 2)  # stops on the repeat
        self.assertEqual(client.transport.call_count, 2)

    def test_explicit_cursor_path(self):
        client = make([
            (200, {"data": {"page": {"values": [1]}, "link": "l2"}}),
            (200, {"data": {"page": {"values": [2]}, "link": None}}),
        ])
        items = list(
            client.paginate("/items", items_path="data.page.values", cursor_path="data.link")
        )
        self.assertEqual(items, [1, 2])

    def test_alternative_cursor_keys_are_recognised(self):
        for key in ("next_cursor", "cursor", "nextCursor", "next", "after", "page_token"):
            client = make([
                (200, {"items": [1], key: "x"}),
                (200, {"items": [2], key: None}),
            ])
            self.assertEqual(list(client.paginate("/items")), [1, 2], f"key={key}")

    def test_custom_cursor_param_name(self):
        client = make([
            (200, {"items": [1], "next": "n2"}),
            (200, {"items": [], "next": None}),
        ], )
        list(client.paginate("/items", cursor_param="starting_after"))
        self.assertEqual(client.transport.calls[1].query["starting_after"], ["n2"])

    def test_max_pages_bound(self):
        # Distinct cursors, so the repeat guard is not what stops us.
        pages = [{"items": [n], "next_cursor": f"c{n}"} for n in range(10)]
        client = make([(200, p) for p in pages])
        items = list(client.paginate("/items", max_pages=3))
        self.assertEqual(client.transport.call_count, 3)
        self.assertEqual(items, [0, 1, 2])

    def test_max_items_bound_stops_mid_page(self):
        client = make([(200, {"items": [1, 2, 3, 4], "next_cursor": "x"})] * 3)
        self.assertEqual(list(client.paginate("/items", max_items=2)), [1, 2])
        self.assertEqual(client.transport.call_count, 1)


class TestPagePagination(unittest.TestCase):
    def test_pages_increment_until_a_short_page(self):
        client = make([
            (200, {"items": [1, 2]}),
            (200, {"items": [3, 4]}),
            (200, {"items": [5]}),
        ])
        self.assertEqual(list(client.paginate("/items", mode="page", page_size=2)), [1, 2, 3, 4, 5])

    def test_page_size_is_sent_as_limit(self):
        client = make([(200, {"items": [1]})])
        list(client.paginate("/items", mode="page", page_size=50))
        self.assertEqual(client.transport.calls[0].query["limit"], ["50"])

    def test_full_final_page_terminates_on_empty_next_page(self):
        client = make([
            (200, {"items": [1, 2]}),
            (200, {"items": []}),
        ])
        self.assertEqual(list(client.paginate("/items", mode="page", page_size=2)), [1, 2])

    def test_page_mode_without_page_size_stops_after_one_page(self):
        # No page_size means no reliable termination signal for page mode.
        client = make([(200, {"items": [1, 2, 3]})])
        self.assertEqual(list(client.paginate("/items", mode="page")), [1, 2, 3])
        self.assertEqual(client.transport.call_count, 1)

    def test_max_pages_bounds_page_mode(self):
        client = make([(200, {"items": [1, 2]})] * 10)
        list(client.paginate("/items", mode="page", page_size=2, max_pages=2))
        self.assertEqual(client.transport.call_count, 2)


class TestItemExtraction(unittest.TestCase):
    def test_common_container_keys(self):
        for key in ("items", "data", "results", "records", "rows", "values"):
            client = make([(200, {key: [7, 8], "next_cursor": None})])
            self.assertEqual(list(client.paginate("/x")), [7, 8], f"key={key}")

    def test_bare_list_response(self):
        client = make([(200, [1, 2, 3])])
        self.assertEqual(list(client.paginate("/x")), [1, 2, 3])

    def test_single_object_response_becomes_one_item(self):
        client = make([(200, {"id": 1, "name": "Ana"})])
        self.assertEqual(list(client.paginate("/x")), [{"id": 1, "name": "Ana"}])

    def test_items_path_miss_returns_empty_not_crash(self):
        client = make([(200, {"unexpected": True})])
        self.assertEqual(list(client.paginate("/x", items_path="a.b.c")), [])


class TestPaginationAgainstRealServer(unittest.TestCase):
    def test_cursor_flow_over_http(self):
        with LocalServer() as server:
            pages = [
                {"items": [1, 2], "next_cursor": "c2"},
                {"items": [3], "next_cursor": None},
            ]

            def handler(record):
                from urllib.parse import parse_qs, urlsplit

                cursor = parse_qs(urlsplit(record["path"]).query).get("cursor", [None])[0]
                return 200, pages[0] if cursor is None else pages[1]

            server.route("GET", "/items")(handler)
            client = Client(server.base_url, sleep=lambda _: None)
            self.assertEqual(list(client.paginate("/items")), [1, 2, 3])
            self.assertEqual(len(server.requests), 2)

    def test_page_flow_over_http(self):
        with LocalServer() as server:
            state = {"n": 0}

            def handler(_record):
                state["n"] += 1
                if state["n"] == 1:
                    return 200, {"items": [1, 2]}
                if state["n"] == 2:
                    return 200, {"items": [3]}
                return 200, {"items": []}

            server.route("GET", "/items")(handler)
            client = Client(server.base_url, sleep=lambda _: None)
            self.assertEqual(list(client.paginate("/items", mode="page", page_size=2)), [1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)