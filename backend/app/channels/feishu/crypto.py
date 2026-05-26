"""Feishu callback envelope encryption (AES-256-CBC, PKCS7).

The encrypted body arrives as ``{"encrypt": "<base64>"}``. The base64-decoded
ciphertext layout is:

    iv(16 bytes) | encrypted_payload(...)

The AES-256 key is derived as ``sha256(encrypt_key)``. After AES-CBC decrypt
with that key and the embedded IV, the plaintext is the original JSON event
body (UTF-8). PKCS7 padding is stripped.
"""

from __future__ import annotations

import base64
import hashlib
import secrets

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class FeishuCryptoError(Exception):
    """Raised when decryption fails."""


class FeishuCrypto:
    """Encrypt and decrypt Feishu callback envelopes."""

    def __init__(self, encrypt_key: str) -> None:
        if not encrypt_key:
            raise FeishuCryptoError("encrypt_key must be a non-empty string")
        self._key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()

    @staticmethod
    def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
        pad_len = block_size - (len(data) % block_size)
        return data + bytes([pad_len]) * pad_len

    @staticmethod
    def _pkcs7_unpad(data: bytes) -> bytes:
        if not data:
            raise FeishuCryptoError("empty plaintext")
        pad_len = data[-1]
        if pad_len < 1 or pad_len > 16:
            raise FeishuCryptoError("invalid pkcs7 padding")
        return data[:-pad_len]

    def encrypt(self, plaintext: str) -> str:
        """Encrypt ``plaintext`` and return the base64-encoded envelope.

        Used in tests; production callers only invoke :meth:`decrypt`.
        """
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        padded = self._pkcs7_pad(plaintext.encode("utf-8"))
        ct = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(iv + ct).decode("ascii")

    def decrypt(self, encrypted_b64: str) -> str:
        """Decrypt the envelope and return the inner UTF-8 plaintext."""
        try:
            blob = base64.b64decode(encrypted_b64)
        except Exception as exc:
            raise FeishuCryptoError(f"ciphertext is not valid base64: {exc}") from exc
        if len(blob) < 32:
            raise FeishuCryptoError("ciphertext too short")

        iv = blob[:16]
        ct = blob[16:]
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        try:
            padded = decryptor.update(ct) + decryptor.finalize()
        except Exception as exc:
            raise FeishuCryptoError(f"AES-CBC decryption failed: {exc}") from exc

        plaintext = self._pkcs7_unpad(padded)
        return plaintext.decode("utf-8")
