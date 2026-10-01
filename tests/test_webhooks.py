"""Tests for webhook signature verification.

HMAC mistakes fail in the worst direction: too loose and anyone who finds your
endpoint can forge events, too strict and you silently drop real ones. Every
branch below is a mistake that has shipped somewhere.
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apisync import InMemoryIdempotencyStore, sign_payload, verify_signature  # noqa: E402

SECRET = "whsec_test_123"
PAYLOAD = '{"id":"evt_1","type":"order.created"}'


class TestSignPayload(unittest.TestCase):
    def test_signature_shape(self):
        header = sign_payload(SECRET, PAYLOAD, timestamp="1700000000")
        self.assertTrue(header.startswith("t=1700000000,v1="))
        digest = header.split("v1=", 1)[1]
        self.assertEqual(len(digest), 64)  # sha256 hex

    def test_deterministic_for_same_timestamp(self):
        self.assertEqual(
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
        )

    def test_timestamp_changes_the_signature(self):
        self.assertNotEqual(
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
            sign_payload(SECRET, PAYLOAD, timestamp="1700000001"),
        )

    def test_body_changes_the_signature(self):
        self.assertNotEqual(
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
            sign_payload(SECRET, PAYLOAD + " ", timestamp="1700000000"),
        )

    def test_secret_changes_the_signature(self):
        self.assertNotEqual(
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
            sign_payload("other", PAYLOAD, timestamp="1700000000"),
        )

    def test_bytes_and_str_payloads_agree(self):
        self.assertEqual(
            sign_payload(SECRET, PAYLOAD, timestamp="1700000000"),
            sign_payload(SECRET, PAYLOAD.encode(), timestamp="1700000000"),
        )

    def test_signature_covers_the_timestamp_not_just_the_body(self):
        # Prevents replaying a captured valid header against a new body.
        header = sign_payload(SECRET, PAYLOAD, timestamp="1700000000")
        self.assertFalse(verify_signature(SECRET, PAYLOAD.replace("evt_1", "evt_2"), header,
                                          tolerance=0))


class TestVerifySignature(unittest.TestCase):
    def test_valid_signature_passes(self):
        header = sign_payload(SECRET, PAYLOAD)
        self.assertTrue(verify_signature(SECRET, PAYLOAD, header))

    def test_wrong_secret_fails(self):
        header = sign_payload(SECRET, PAYLOAD)
        self.assertFalse(verify_signature("wrong", PAYLOAD, header))

    def test_tampered_body_fails(self):
        header = sign_payload(SECRET, PAYLOAD)
        self.assertFalse(verify_signature(SECRET, PAYLOAD.replace("created", "deleted"), header))

    def test_missing_parts_fail_closed(self):
        self.assertFalse(verify_signature(SECRET, PAYLOAD, ""))
        self.assertFalse(verify_signature(SECRET, PAYLOAD, "t=123"))
        self.assertFalse(verify_signature(SECRET, PAYLOAD, "v1=deadbeef"))
        self.assertFalse(verify_signature(SECRET, PAYLOAD, "garbage-without-equals"))

    def test_stale_timestamp_is_rejected(self):
        old = str(int(time.time()) - 3600)
        header = sign_payload(SECRET, PAYLOAD, timestamp=old)
        self.assertFalse(verify_signature(SECRET, PAYLOAD, header, tolerance=300))

    def test_tolerance_disabled_accepts_any_timestamp(self):
        old = str(int(time.time()) - 3600)
        header = sign_payload(SECRET, PAYLOAD, timestamp=old)
        self.assertTrue(verify_signature(SECRET, PAYLOAD, header, tolerance=0))

    def test_future_timestamp_also_rejected(self):
        ahead = str(int(time.time()) + 3600)
        header = sign_payload(SECRET, PAYLOAD, timestamp=ahead)
        self.assertFalse(verify_signature(SECRET, PAYLOAD, header, tolerance=300))

    def test_extra_header_elements_are_ignored(self):
        header = sign_payload(SECRET, PAYLOAD) + ",v1=decoy"
        self.assertTrue(verify_signature(SECRET, PAYLOAD, header))

    def test_comparison_is_constant_time_shape(self):
        # Not asserting timing, just that a same-length wrong digest is handled.
        stamp = str(int(time.time()))
        good = sign_payload(SECRET, PAYLOAD, timestamp=stamp).split("v1=", 1)[1]
        bad = ("0" if good[0] != "0" else "1") + good[1:]
        self.assertFalse(verify_signature(SECRET, PAYLOAD, f"t={stamp},v1={bad}"))


class TestWebhookDedupPattern(unittest.TestCase):
    """The pattern the signature tests protect: process each event once."""

    def test_replayed_event_is_processed_once(self):
        store = InMemoryIdempotencyStore()
        processed = []
        for event_id in ("evt_1", "evt_1", "evt_2", "evt_1"):
            if store.seen(event_id):
                continue
            store.remember(event_id, True)
            processed.append(event_id)
        self.assertEqual(processed, ["evt_1", "evt_2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)