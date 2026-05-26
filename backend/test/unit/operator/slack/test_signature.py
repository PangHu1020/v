"""Unit tests for ``backend.app.operator.slack.signature``."""

from __future__ import annotations

import time

from backend.app.operator.slack.signature import compute_signature, verify_signature

SECRET = "test-signing-secret"


class TestComputeSignature:
    def test_format(self) -> None:
        sig = compute_signature(SECRET, "1700000000", b"body")
        assert sig.startswith("v0=")
        assert len(sig) == 3 + 64  # v0= plus sha256 hex

    def test_deterministic(self) -> None:
        a = compute_signature(SECRET, "1700000000", b"body")
        b = compute_signature(SECRET, "1700000000", b"body")
        assert a == b

    def test_changes_with_body(self) -> None:
        a = compute_signature(SECRET, "1700000000", b"body-a")
        b = compute_signature(SECRET, "1700000000", b"body-b")
        assert a != b

    def test_changes_with_timestamp(self) -> None:
        a = compute_signature(SECRET, "1700000000", b"body")
        b = compute_signature(SECRET, "1700000001", b"body")
        assert a != b

    def test_changes_with_secret(self) -> None:
        a = compute_signature(SECRET, "1700000000", b"body")
        b = compute_signature("other-secret", "1700000000", b"body")
        assert a != b


class TestVerifySignature:
    def test_accepts_correct(self) -> None:
        ts = str(int(time.time()))
        sig = compute_signature(SECRET, ts, b"body")
        assert verify_signature(signing_secret=SECRET, timestamp=ts, body=b"body", signature=sig)

    def test_rejects_wrong_signature(self) -> None:
        ts = str(int(time.time()))
        assert not verify_signature(
            signing_secret=SECRET,
            timestamp=ts,
            body=b"body",
            signature="v0=" + "0" * 64,
        )

    def test_rejects_stale_timestamp(self) -> None:
        # Timestamp 10 minutes old.
        old_ts = str(int(time.time()) - 600)
        sig = compute_signature(SECRET, old_ts, b"body")
        assert not verify_signature(
            signing_secret=SECRET, timestamp=old_ts, body=b"body", signature=sig
        )

    def test_rejects_future_timestamp(self) -> None:
        future_ts = str(int(time.time()) + 600)
        sig = compute_signature(SECRET, future_ts, b"body")
        assert not verify_signature(
            signing_secret=SECRET, timestamp=future_ts, body=b"body", signature=sig
        )

    def test_rejects_non_numeric_timestamp(self) -> None:
        assert not verify_signature(
            signing_secret=SECRET,
            timestamp="not-a-number",
            body=b"body",
            signature="v0=ignored",
        )

    def test_within_window_with_explicit_now(self) -> None:
        ts = "1700000000"
        sig = compute_signature(SECRET, ts, b"body")
        # ``now`` close to ``ts`` — accepted.
        assert verify_signature(
            signing_secret=SECRET,
            timestamp=ts,
            body=b"body",
            signature=sig,
            now=1700000060,
        )
        # ``now`` far from ``ts`` — rejected.
        assert not verify_signature(
            signing_secret=SECRET,
            timestamp=ts,
            body=b"body",
            signature=sig,
            now=1700001000,
        )
