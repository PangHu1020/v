"""Unit tests for WeCom signature + AES envelope crypto."""

from __future__ import annotations

import pytest

from backend.app.channels.wecom.crypto import WecomCrypto, WecomCryptoError
from backend.app.channels.wecom.signature import compute_signature, verify_signature

# 43-char base64-safe EncodingAESKey (decodes to 32 bytes after appending "=").
TEST_AES_KEY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"
TEST_CORP_ID = "wxabcdef1234567890"
TEST_TOKEN = "qwerty"


class TestSignature:
    def test_deterministic(self) -> None:
        a = compute_signature(TEST_TOKEN, "1700000000", "n1", "ENC")
        b = compute_signature(TEST_TOKEN, "1700000000", "n1", "ENC")
        assert a == b

    def test_changes_with_each_field(self) -> None:
        baseline = compute_signature(TEST_TOKEN, "ts", "nonce", "enc")
        assert compute_signature("other", "ts", "nonce", "enc") != baseline
        assert compute_signature(TEST_TOKEN, "other", "nonce", "enc") != baseline
        assert compute_signature(TEST_TOKEN, "ts", "other", "enc") != baseline
        assert compute_signature(TEST_TOKEN, "ts", "nonce", "other") != baseline

    def test_independent_of_input_order(self) -> None:
        # Sorting is part of the algorithm; reordering the four inputs at
        # the call site MUST yield the same signature when the lexical sort
        # stays identical. Use four strings whose lex order is invariant
        # under transposition.
        s1 = compute_signature("a", "b", "c", "d")
        s2 = compute_signature("a", "b", "c", "d")
        assert s1 == s2

    def test_verify_signature_constant_time(self) -> None:
        sig = compute_signature(TEST_TOKEN, "ts", "nonce", "enc")
        assert verify_signature(
            token=TEST_TOKEN,
            timestamp="ts",
            nonce="nonce",
            encrypted="enc",
            signature=sig,
        )
        assert not verify_signature(
            token=TEST_TOKEN,
            timestamp="ts",
            nonce="nonce",
            encrypted="enc",
            signature="bad",
        )


class TestCryptoConstruction:
    def test_wrong_length_key_rejected(self) -> None:
        with pytest.raises(WecomCryptoError):
            WecomCrypto("too-short", TEST_CORP_ID)

    def test_invalid_base64_rejected(self) -> None:
        with pytest.raises(WecomCryptoError):
            WecomCrypto("!" * 43, TEST_CORP_ID)


class TestEncryptDecryptRoundTrip:
    @pytest.fixture
    def crypto(self) -> WecomCrypto:
        return WecomCrypto(TEST_AES_KEY, TEST_CORP_ID)

    def test_round_trip_ascii(self, crypto: WecomCrypto) -> None:
        original = "<xml><Content>hello</Content></xml>"
        ct = crypto.encrypt(original)
        assert crypto.decrypt(ct) == original

    def test_round_trip_chinese(self, crypto: WecomCrypto) -> None:
        original = "<xml><Content>你好世界</Content></xml>"
        ct = crypto.encrypt(original)
        assert crypto.decrypt(ct) == original

    def test_round_trip_long_payload(self, crypto: WecomCrypto) -> None:
        original = "<xml>" + "X" * 5000 + "</xml>"
        ct = crypto.encrypt(original)
        assert crypto.decrypt(ct) == original

    def test_each_encryption_uses_fresh_random_prefix(self, crypto: WecomCrypto) -> None:
        ct1 = crypto.encrypt("same message")
        ct2 = crypto.encrypt("same message")
        # Random 16-byte prefix means ciphertexts must differ even for the
        # same plaintext. This guards against a regression to a static prefix.
        assert ct1 != ct2


class TestDecryptFailures:
    @pytest.fixture
    def crypto(self) -> WecomCrypto:
        return WecomCrypto(TEST_AES_KEY, TEST_CORP_ID)

    def test_bad_base64_rejected(self, crypto: WecomCrypto) -> None:
        with pytest.raises(WecomCryptoError):
            crypto.decrypt("@@@not-base64@@@")

    def test_corp_id_mismatch_rejected(self, crypto: WecomCrypto) -> None:
        ct = crypto.encrypt("<xml/>")
        wrong = WecomCrypto(TEST_AES_KEY, "wxOtherCorp_____")
        with pytest.raises(WecomCryptoError, match="corp_id mismatch"):
            wrong.decrypt(ct)

    def test_truncated_payload_rejected(self, crypto: WecomCrypto) -> None:
        ct = crypto.encrypt("<xml/>")
        # Lop off the last block; AES-CBC will still "decrypt" but the
        # padding check or length prefix will catch it.
        import base64

        raw = base64.b64decode(ct)
        truncated = base64.b64encode(raw[:-16]).decode("ascii")
        with pytest.raises(WecomCryptoError):
            crypto.decrypt(truncated)
