"""WeCom callback signature.

WeCom's signature algorithm:

    sig = sha1( sort([token, timestamp, nonce, encrypted_body]).join("") )

The four inputs are sorted lexicographically, concatenated, then SHA-1 hashed.
The result hex digest is sent as the ``msg_signature`` query parameter on
inbound POSTs and as ``signature`` on the GET echo for URL verification.
"""

from __future__ import annotations

import hashlib
import hmac


def compute_signature(token: str, timestamp: str, nonce: str, encrypted: str) -> str:
    """Return the WeCom SHA-1 signature for the given fields.

    Args:
        token: Application-configured callback token.
        timestamp: Webhook ``timestamp`` query parameter.
        nonce: Webhook ``nonce`` query parameter.
        encrypted: For POST events, the inner ``<Encrypt>...</Encrypt>``
            body. For the GET echo, the ``echostr`` query parameter.

    Returns:
        Lowercase hex SHA-1 digest.
    """
    parts = sorted([token, timestamp, nonce, encrypted])
    digest = hashlib.sha1(("".join(parts)).encode("utf-8"))  # noqa: S324  # algorithm fixed by WeCom protocol
    return digest.hexdigest()


def verify_signature(
    *,
    token: str,
    timestamp: str,
    nonce: str,
    encrypted: str,
    signature: str,
) -> bool:
    """Constant-time comparison of the computed signature against the provided one."""
    expected = compute_signature(token, timestamp, nonce, encrypted)
    return hmac.compare_digest(expected, signature)
