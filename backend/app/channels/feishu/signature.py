"""Feishu callback signature.

Feishu's signature (v2):

    sig = sha256( timestamp + nonce + encrypt_key + body )

The four inputs are concatenated in that exact order as a UTF-8 byte string,
then SHA-256 hashed. The result hex digest is sent as the
``X-Lark-Signature`` header. Feishu also sends ``X-Lark-Request-Timestamp``
and ``X-Lark-Request-Nonce`` headers carrying the values used in the
signature.
"""

from __future__ import annotations

import hashlib
import hmac


def compute_signature(timestamp: str, nonce: str, encrypt_key: str, body: bytes) -> str:
    """Return the Feishu SHA-256 signature for the given request.

    Args:
        timestamp: ``X-Lark-Request-Timestamp`` header value.
        nonce: ``X-Lark-Request-Nonce`` header value.
        encrypt_key: Application-configured callback encrypt key.
        body: Raw request body bytes.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(timestamp.encode("utf-8"))
    digest.update(nonce.encode("utf-8"))
    digest.update(encrypt_key.encode("utf-8"))
    digest.update(body)
    return digest.hexdigest()


def verify_signature(
    *,
    timestamp: str,
    nonce: str,
    encrypt_key: str,
    body: bytes,
    signature: str,
) -> bool:
    """Constant-time comparison of the computed signature against the provided one."""
    expected = compute_signature(timestamp, nonce, encrypt_key, body)
    return hmac.compare_digest(expected, signature)
