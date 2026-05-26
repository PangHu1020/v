"""Slack request signature verification.

Slack signs every inbound request with a v0 HMAC-SHA256 signature derived
from the request body and a per-app signing secret. The header layout is:

    X-Slack-Request-Timestamp: <unix-seconds>
    X-Slack-Signature: v0=<hex-sha256>

The signature base string is ``v0:{timestamp}:{body}``. Slack also
recommends rejecting requests whose timestamp is more than 5 minutes off
the current clock — that drops simple replay attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import time

_REPLAY_WINDOW_SECONDS = 5 * 60


def compute_signature(signing_secret: str, timestamp: str, body: bytes) -> str:
    """Return the canonical ``v0=<hex>`` signature string for the request."""
    base = b"v0:" + timestamp.encode("ascii") + b":" + body
    digest = hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def verify_signature(
    *,
    signing_secret: str,
    timestamp: str,
    body: bytes,
    signature: str,
    now: float | None = None,
    replay_window_seconds: int = _REPLAY_WINDOW_SECONDS,
) -> bool:
    """Return ``True`` iff signature matches and timestamp is within the replay window.

    Args:
        signing_secret: App-level Slack signing secret.
        timestamp: ``X-Slack-Request-Timestamp`` header.
        body: Raw request body bytes.
        signature: ``X-Slack-Signature`` header (full ``v0=<hex>`` string).
        now: Override "current time" for tests; defaults to ``time.time()``.
        replay_window_seconds: How stale a timestamp may be. Default 5 min.
    """
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False

    current = now if now is not None else time.time()
    if abs(current - ts) > replay_window_seconds:
        return False

    expected = compute_signature(signing_secret, timestamp, body)
    return hmac.compare_digest(expected, signature)
