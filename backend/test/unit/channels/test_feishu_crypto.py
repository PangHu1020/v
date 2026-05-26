"""Unit tests for Feishu signature + AES envelope crypto."""

from __future__ import annotations

import json

import pytest

from backend.app.channels.feishu.crypto import FeishuCrypto, FeishuCryptoError
from backend.app.channels.feishu.signature import compute_signature, verify_signature

TEST_ENCRYPT_KEY = "test-encrypt-key-2024"


class TestSignature:
    def test_deterministic(self) -> None:
        body = b'{"event":"hello"}'
        a = compute_signature("1700000000", "n1", TEST_ENCRYPT_KEY, body)
        b = compute_signature("1700000000", "n1", TEST_ENCRYPT_KEY, body)
        assert a == b

    def test_changes_with_each_field(self) -> None:
        body = b'{"x": 1}'
        baseline = compute_signature("1", "n", TEST_ENCRYPT_KEY, body)
        assert compute_signature("2", "n", TEST_ENCRYPT_KEY, body) != baseline
        assert compute_signature("1", "m", TEST_ENCRYPT_KEY, body) != baseline
        assert compute_signature("1", "n", "different", body) != baseline
        assert compute_signature("1", "n", TEST_ENCRYPT_KEY, b"{}") != baseline

    def test_verify_accepts_correct(self) -> None:
        body = b'{"a": 1}'
        sig = compute_signature("1", "n", TEST_ENCRYPT_KEY, body)
        assert verify_signature(
            timestamp="1",
            nonce="n",
            encrypt_key=TEST_ENCRYPT_KEY,
            body=body,
            signature=sig,
        )

    def test_verify_rejects_wrong(self) -> None:
        body = b'{"a": 1}'
        assert not verify_signature(
            timestamp="1",
            nonce="n",
            encrypt_key=TEST_ENCRYPT_KEY,
            body=body,
            signature="0" * 64,
        )


class TestCryptoConstruction:
    def test_empty_key_rejected(self) -> None:
        with pytest.raises(FeishuCryptoError):
            FeishuCrypto("")


class TestEncryptDecryptRoundTrip:
    @pytest.fixture
    def crypto(self) -> FeishuCrypto:
        return FeishuCrypto(TEST_ENCRYPT_KEY)

    def test_round_trip_ascii(self, crypto: FeishuCrypto) -> None:
        payload = json.dumps({"event": "im.message.receive_v1", "message": "hi"})
        ct = crypto.encrypt(payload)
        assert crypto.decrypt(ct) == payload

    def test_round_trip_chinese(self, crypto: FeishuCrypto) -> None:
        payload = json.dumps({"text": "客户咨询订单ORD123状态"}, ensure_ascii=False)
        ct = crypto.encrypt(payload)
        assert crypto.decrypt(ct) == payload

    def test_round_trip_long_payload(self, crypto: FeishuCrypto) -> None:
        payload = "x" * 8192
        ct = crypto.encrypt(payload)
        assert crypto.decrypt(ct) == payload

    def test_each_encryption_uses_fresh_iv(self, crypto: FeishuCrypto) -> None:
        ct1 = crypto.encrypt("same")
        ct2 = crypto.encrypt("same")
        assert ct1 != ct2


class TestDecryptFailures:
    @pytest.fixture
    def crypto(self) -> FeishuCrypto:
        return FeishuCrypto(TEST_ENCRYPT_KEY)

    def test_bad_base64_rejected(self, crypto: FeishuCrypto) -> None:
        with pytest.raises(FeishuCryptoError):
            crypto.decrypt("@@@not-base64@@@")

    def test_short_payload_rejected(self, crypto: FeishuCrypto) -> None:
        import base64

        with pytest.raises(FeishuCryptoError):
            crypto.decrypt(base64.b64encode(b"abc").decode("ascii"))

    def test_wrong_key_yields_garbage_or_padding_error(self) -> None:
        a = FeishuCrypto("key-a")
        b = FeishuCrypto("key-b")
        ct = a.encrypt('{"v": 1}')
        # Either the decryption raises a padding error, or it returns
        # corrupted bytes that won't match the original. Both are acceptable.
        try:
            assert b.decrypt(ct) != '{"v": 1}'
        except FeishuCryptoError:
            pass
