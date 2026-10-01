"""End-to-end demo: a real two-stage sync against a real local server.

Mirrors the job a client actually hires for: pull orders from one API,
write them to another, and survive the API being unhealthy halfway through.

Run it directly to watch the behaviour, including the retry delays:

    python3 examples/sync_demo.py --delays
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from apisync import Client, HttpError, InMemoryIdempotencyStore, RetryPolicy  # noqa: E402
from helpers import LocalServer  # noqa: E402


def run(delays: bool = False, verbose: bool = True) -> dict:
    """One pass of the sync. Returns a summary dict."""
    retries: list[tuple[int, float, str]] = []

    with LocalServer() as source, LocalServer() as sink:
        source.route("GET", "/orders")(_fail_once_then_pages())
        received: list[dict] = []

        def accept(record):
            received.append(json.loads(record["body"].decode()))
            return 201, {"ok": True}

        sink.route("POST", "/inbound")(accept)

        api = Client(
            source.base_url,
            retry=RetryPolicy(attempts=4, base=0.2, factor=2.0, jitter=False),
            sleep=(time.sleep if delays else (lambda _s: None)),
            on_retry=lambda attempt, delay, reason: retries.append((attempt, delay, reason)),
        )
        warehouse = Client(sink.base_url, retry=RetryPolicy(attempts=3, base=0.1, jitter=False),
                           sleep=lambda _s: None, idempotency=InMemoryIdempotencyStore())

        def log(message: str) -> None:
            if verbose:
                print(message)

        log(f"source : {source.base_url}")
        log(f"sink   : {sink.base_url}")
        log("")
        log("1. pulling orders with cursor pagination (source 503s on the first call)")

        synced = 0
        for order in api.paginate("/orders", page_size=2):
            if order["status"] != "paid":
                log(f"   skip {order['id']} (status={order['status']})")
                continue
            # Idempotency key derived from the record id: a replay of this sync
            # cannot double-create the warehouse order.
            key = f"order-{order['id']}"
            warehouse.post("/inbound", {"ref": order["id"], "amount": order["total"]},
                           idempotency_key=key)
            synced += 1
            log(f"   synced {order['id']} ${order['total']:.2f}")

        log("")
        log("2. replaying the last write with the same idempotency key")
        before = len(received)
        warehouse.post("/inbound", {"ref": "o_2", "amount": 32.00}, idempotency_key="order-o_2")
        deduped = len(received) == before
        log(f"   sink received {len(received)} request(s) for 2 distinct orders "
            f"({'dedupe works' if deduped else 'DUPLICATE CREATED'})")

        log("")
        log("3. hitting an endpoint that does not exist")
        try:
            api.get("/nope")
        except HttpError as exc:
            log(f"   raised HttpError(status={exc.status}) without retrying")

        log("")
        log(f"retries performed: {len(retries)}")
        for attempt, delay, reason in retries:
            log(f"   attempt {attempt}: waited {delay:.2f}s ({reason})")

        return {
            "orders_synced": synced,
            "sink_requests": len(received),
            "retries": len(retries),
            "dedupe_works": deduped,
        }


def _fail_once_then_pages():
    state = {"n": 0}
    pages = [
        {"items": [
            {"id": "o_1", "total": 149.90, "status": "paid"},
            {"id": "o_2", "total": 32.00, "status": "paid"},
        ], "next_cursor": "c2"},
        {"items": [{"id": "o_3", "total": 78.25, "status": "refunded"}],
         "next_cursor": None},
    ]

    def handler(record):
        from urllib.parse import parse_qs, urlsplit

        state["n"] += 1
        if state["n"] == 1:
            return 503, {"error": "upstream warming up"}
        cursor = parse_qs(urlsplit(record["path"]).query).get("cursor", [None])[0]
        return 200, pages[0] if cursor is None else pages[1]

    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays", action="store_true",
                        help="actually sleep between retries instead of fast-forwarding")
    args = parser.parse_args()
    summary = run(delays=args.delays)
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())