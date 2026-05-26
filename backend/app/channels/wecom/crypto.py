"""WeCom callback envelope encryption (AES-256-CBC, PKCS7).

The ``EncodingAESKey`` is 43 base64-safe characters; appending ``"="`` yields
valid base64 that decodes to 32 bytes — the AES-256 key. The IV is the
first 16 bytes of that key.

Ciphertext format (after base64 decode and AES-CBC decrypt):

    random(16) | msg_len_be32(4) | msg_bytes | corp_id_bytes

The leading 16 random bytes are discarded. ``msg_len`` is the length in
bytes of ``msg`` (the inner XML/JSON content). ``corp_id`` is the receiver's
corp id and MUST match the configured value; this is the integrity check.
"""

from __future__ import annotations

import base64
import secrets
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class WecomCryptoError(Exception):
    """Raised when decryption or integrity checks fail."""


class WecomCrypto:
    """Encrypt and decrypt WeCom callback envelopes."""

    def __init__(self, encoding_aes_key: str, corp_id: str) -> None:
        if len(encoding_aes_key) != 43:
            raise WecomCryptoError(f"EncodingAESKey must be 43 chars; got {len(encoding_aes_key)}")
        try:
            self._key = base64.b64decode(encoding_aes_key + "=")
        except Exception as exc:
            raise WecomCryptoError(f"EncodingAESKey is not valid base64: {exc}") from exc
        if len(self._key) != 32:
            raise WecomCryptoError(f"Decoded EncodingAESKey must be 32 bytes; got {len(self._key)}")
        self._iv = self._key[:16]
        self._corp_id = corp_id.encode("utf-8")

    @staticmethod
    def _pkcs7_pad(data: bytes, block_size: int = 32) -> bytes:
        pad_len = block_size - (len(data) % block_size)
        return data + bytes([pad_len]) * pad_len

    @staticmethod
    def _pkcs7_unpad(data: bytes) -> bytes:
        if not data:
            raise WecomCryptoError("empty plaintext")
        pad_len = data[-1]
        if pad_len < 1 or pad_len > 32:
            raise WecomCryptoError("invalid pkcs7 padding")
        return data[:-pad_len]

    def encrypt(self, msg: str) -> str:
        """Build the base64 ciphertext envelope for ``msg``.

        Used in tests; the production gateway only calls :meth:`decrypt`.
        """
        msg_bytes = msg.encode("utf-8")
        random_prefix = secrets.token_bytes(16)
        msg_len = struct.pack(">I", len(msg_bytes))
        plaintext = random_prefix + msg_len + msg_bytes + self._corp_id
        padded = self._pkcs7_pad(plaintext)

        cipher = Cipher(algorithms.AES(self._key), modes.CBC(self._iv))
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(ct).decode("ascii")

    def decrypt(self, encrypted_b64: str) -> str:
        """Decrypt the envelope and return the inner ``msg`` string.

        Raises:
            WecomCryptoError: on bad base64, padding, length prefix mismatch,
                or corp_id mismatch.
        """
        try:
            ct = base64.b64decode(encrypted_b64)
        except Exception as exc:
            raise WecomCryptoError(f"ciphertext is not valid base64: {exc}") from exc

        cipher = Cipher(algorithms.AES(self._key), modes.CBC(self._iv))
        decryptor = cipher.decryptor()
        try:
            padded = decryptor.update(ct) + decryptor.finalize()
        except Exception as exc:
            raise WecomCryptoError(f"AES-CBC decryption failed: {exc}") from exc

        plaintext = self._pkcs7_unpad(padded)
        if len(plaintext) < 20:
            raise WecomCryptoError("plaintext shorter than header")

        msg_len = struct.unpack(">I", plaintext[16:20])[0]
        msg_end = 20 + msg_len
        if msg_end > len(plaintext):
            raise WecomCryptoError("declared msg length exceeds plaintext")

        msg = plaintext[20:msg_end]
        recv_corp_id = plaintext[msg_end:]
        if recv_corp_id != self._corp_id:
            raise WecomCryptoError("corp_id mismatch")

        return msg.decode("utf-8")
