# api-sync

A REST client for Python that behaves when the other side is unhealthy.
**Zero dependencies** — the standard library ships everything needed
(`urllib`, `hmac`, `hashlib`). Python 3.10+.

```bash
git clone https://github.com/ardani13ln-byte/api-sync
cd api-sync
python3 examples/sync_demo.py       # watch it work, no setup
python3 -m unittest discover -s tests
```

The hard part of calling someone else's API is not the happy path. It is the
429 at 2am, the cursor pagination that silently truncates your results, and the
webhook retry that processes the same order twice. This library makes those
cases explicit and tested.

## The four bugs this prevents

| What goes wrong in production | What api-sync does |
|---|---|
| `requests` retry loop that hammers a struggling API with fixed 1s intervals | Bounded exponential backoff with **full jitter** (`uniform(0, backoff)`) — synchronised retries are how a recovering API gets knocked over again |
| A 401 retried four times, wasting 30s and the rate-limit budget | `AuthError` raised immediately; only 408/425/429/5xx are retryable |
| Cursor pagination that loops forever when the server echoes the same cursor | Repeated cursors detected and the loop stops |
| A timed-out POST retried blindly, creating the order twice | `idempotency_key` deduplicates writes — locally **and** via the `Idempotency-Key` header |
| Webhook signature checked loosely, or with a naive `==` | HMAC-SHA256 with a **replay window**, **constant-time** compare, and support for multiple `v1` values during key rotation |

## Retry

```python
from apisync import Client, RetryPolicy

api = Client(
    "https://api.example.com",
    token=os.environ["API_TOKEN"],
    retry=RetryPolicy(attempts=5, base=0.5, factor=2.0, max_delay=30.0),
    on_retry=lambda attempt, delay, reason: log.warning("retry %s in %.1fs: %s", attempt, delay, reason),
)
```

- Retries `408, 425, 429, 500, 502, 503, 504` plus transport failures (DNS,
  reset, timeout).
- Never retries `4xx` other than the two above — a malformed request will not
  become well-formed.
- `401`/`403` raise `AuthError` on the first response.
- `Retry-After` wins over the computed backoff, in **both** its forms: delay
  seconds and HTTP-date. A server that tells you to wait 30s is obeyed.
- Attempts are bounded (`attempts=4` means at most 3 waits). Backoff is capped
  by `max_delay`.
- `on_retry` fires with `(attempt, delay, reason)` so you can log or alert.

## Pagination

Both styles, one signature. The generator holds nothing but the current page.

```python
# Cursor / token based — stops when the cursor is absent, null or repeated
for order in api.paginate("/orders", page_size=100):
    sync(order)

# Page-number based — stops on a short page
for row in api.paginate("/rows", mode="page", page_size=500):
    store(row)
```

- Auto-detects items under `items`, `data`, `results`, `records`, `rows`,
  `values`, or at the top level; a bare list or a single object also work.
- Auto-detects cursors under `next_cursor`, `cursor`, `nextCursor`, `next`,
  `after`, `page_token`.
- Override with `items_path="data.page.values"` and
  `cursor_path="meta.link"` for unusual envelopes.
- **Always bounded**: `max_pages` and `max_items` are there for the API you
  have not debugged yet.
- A server that keeps returning the same cursor is a real failure mode; the
  loop stops instead of hanging.

## Idempotency

```python
from apisync import InMemoryIdempotencyStore

warehouse = Client("https://warehouse.example.com",
                   idempotency=InMemoryIdempotencyStore())

for order in api.paginate("/orders"):
    warehouse.post("/inbound", order, idempotency_key=f"order-{order['id']}")

# Re-running this job creates nothing twice — even across a 503 + retry.
```

The key is sent as the `Idempotency-Key` header (so the server also
deduplicates) *and* checked against the store (so nothing is sent at all).
Failed requests are **not** cached: a 500 must stay retryable.
`InMemoryIdempotencyStore` is bounded with FIFO eviction; swap in your own
subclass backed by Redis or Postgres for multi-process jobs.

## Webhooks

```python
from apisync import verify_signature, InMemoryIdempotencyStore

def receive(request):
    if not verify_signature(WEBHOOK_SECRET, request.body, request.headers["X-Signature"]):
        return 400  # never trust an unverified body
    event = json.loads(request.body)
    if store.seen(event["id"]):
        return 200  # already processed; ack so they stop retrying
    store.remember(event["id"])
    handle(event)
    return 200
```

`verify_signature` fails closed on a missing or malformed header, rejects
timestamps outside the tolerance (default 300s) in **either** direction, and
accepts the event if **any** of several `v1` signatures match — the format
providers use while rotating keys.

## Testing your own integration

`Transport` is a one-method seam, so retry and pagination logic is testable
without a network or a single `sleep`:

```python
from apisync import Client, Transport

class Scripted(Transport):
    def __init__(self, script): self.script, self.calls = script, []
    def request(self, method, url, headers, body, timeout):
        self.calls.append((method, url))
        return self.script.pop(0)

api = Client("https://api.test", transport=Scripted([(503, {}), (200, {"ok": True})]),
             sleep=lambda _: None)
assert api.get("/x") == {"ok": True}
```

`tests/helpers.py` also ships `LocalServer`, a real `http.server` on a random
port, so the wire behaviour (headers, query strings, status codes) is verified
for real and not only against a double.

## Tests

```
88 tests, ~3s, no network required.
```

The suite covers the failure modes rather than the happy path:

- backoff growth, jitter bounds, attempt budget, `Retry-After` in both formats
- transient failures recovering, budgets exhausted, 401/403/404 not retried,
  request body replayed identically on retry
- cursor pagination across pages, repeated-cursor termination, alternative
  cursor keys, `max_pages`/`max_items` bounds, page mode
- idempotency dedupe, failed writes not cached, `GET` never deduplicated,
  dedupe surviving a real HTTP 503-then-201
- signature verification: tampering, wrong secret, stale and future timestamps,
  missing parts, key rotation
- the same retry and pagination logic again over a real HTTP socket

## Error types

```
ApiError
├── TransportError      no response at all (DNS, reset, timeout)
└── HttpError           a non-2xx status
    ├── RateLimitError  429, carries the server's headers
    └── AuthError       401 / 403, never retried
```

Every `HttpError` carries `.status`, `.url`, `.body` and `.headers`, so a
failure in your logs has everything needed to diagnose it later.

## Design decisions worth stating

- **Full jitter, not fixed backoff.** Fixed intervals resynchronise every
  retrying client after an outage. `uniform(0, backoff)` spreads them.
- **Non-seekable input is normal.** The `Transport` seam takes a method and
  returns a tuple, which is how this library stays dependency-free and testable
  at the same time.
- **`sleep` is injectable.** Tests never wait; demos can with `--delays`.
- **Idempotency is opt-in.** Silently caching writes in a shared client object
  hides bugs, so `NullIdempotencyStore` is the default.

## License

MIT