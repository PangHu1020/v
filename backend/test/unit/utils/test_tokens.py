"""Unit tests for ``backend.v.utils.tokens``."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.v.utils.tokens import (
    _content_to_text,
    count_message_tokens,
    count_messages_tokens,
    count_text_tokens,
    reset_encoder_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_encoder_cache()


class TestCountTextTokens:
    def test_empty_string(self) -> None:
        assert count_text_tokens("") == 0

    def test_ascii_text_via_tiktoken(self) -> None:
        # cl100k_base encodes "hello world" as 2 tokens.
        assert count_text_tokens("hello world") == 2

    def test_chinese_text(self) -> None:
        # cl100k_base encodes "你好世界" as 5 tokens (each Chinese char is
        # ~1 BPE chunk plus a few combining bytes).
        assert count_text_tokens("你好世界") == 5

    def test_unknown_model_falls_back_to_cl100k(self) -> None:
        # An unknown model name resolves to cl100k_base, so we still get
        # a real count instead of the character heuristic.
        n = count_text_tokens("hello world", model="deepseek-chat-v4-pro")
        assert n == 2

    def test_known_model(self) -> None:
        assert count_text_tokens("hello", model="gpt-4") == 1

    def test_falls_back_to_char_heuristic_when_tiktoken_unavailable(self) -> None:
        # Patch the encoder factory to return None (no tiktoken).
        with patch("backend.v.utils.tokens._encoder_for", return_value=None):
            reset_encoder_cache()
            n = count_text_tokens("hello world")  # 11 chars // 2 = 5
        assert n == 5

    def test_char_heuristic_short_string(self) -> None:
        # ``len(text) // 2`` is 0 for 1-char strings; we floor at 1.
        with patch("backend.v.utils.tokens._encoder_for", return_value=None):
            reset_encoder_cache()
            assert count_text_tokens("x") == 1

    def test_encoder_failure_falls_back_to_heuristic(self) -> None:
        class _BadEncoder:
            def encode(self, text):
                raise RuntimeError("bad")

        with patch("backend.v.utils.tokens._encoder_for", return_value=_BadEncoder()):
            reset_encoder_cache()
            # 22 chars // 2 = 11
            assert count_text_tokens("twenty-two characters!") == 11


class TestContentToText:
    def test_string(self) -> None:
        assert _content_to_text("hello") == "hello"

    def test_list_of_strings(self) -> None:
        assert _content_to_text(["a", "b"]) == "a\nb"

    def test_list_of_dicts_with_text(self) -> None:
        parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert _content_to_text(parts) == "hello\nworld"

    def test_list_with_non_text_dict(self) -> None:
        # Dicts without a ``text`` field should not crash; they contribute
        # an empty string and get filtered.
        parts = [{"type": "image_url", "image_url": "..."}, {"type": "text", "text": "hi"}]
        assert _content_to_text(parts) == "hi"

    def test_none(self) -> None:
        assert _content_to_text(None) == ""

    def test_other_type(self) -> None:
        assert _content_to_text(42) == "42"


class TestCountMessageTokens:
    def test_simple_human_message(self) -> None:
        n = count_message_tokens(HumanMessage(content="hello world"))
        assert n == 2

    def test_system_message_chinese(self) -> None:
        n = count_message_tokens(SystemMessage(content="你好"))
        assert n >= 1

    def test_prefers_usage_metadata(self) -> None:
        # If usage_metadata.output_tokens is set, trust it over text encoding.
        msg = AIMessage(
            content="this is a long reply that would tokenize to many tokens",
            usage_metadata={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
        )
        assert count_message_tokens(msg) == 7

    def test_falls_back_when_usage_metadata_missing(self) -> None:
        msg = AIMessage(content="hello")
        # No metadata -> encode the content. cl100k_base("hello") == 1.
        assert count_message_tokens(msg) == 1

    def test_falls_back_when_usage_metadata_zero(self) -> None:
        # An ``output_tokens`` of 0 is treated as missing.
        msg = AIMessage(
            content="hello world",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        assert count_message_tokens(msg) == 2

    def test_response_metadata_token_usage(self) -> None:
        # Some providers stash counts under response_metadata.token_usage.
        msg = AIMessage(content="anything")
        msg.response_metadata = {"token_usage": {"completion_tokens": 11}}  # type: ignore[attr-defined]
        assert count_message_tokens(msg) == 11

    def test_message_with_list_content(self) -> None:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        # "hello\nworld" -> 3 tokens via cl100k_base
        assert count_message_tokens(msg) == 3


class TestCountMessagesTokens:
    def test_sums_across_messages(self) -> None:
        msgs = [
            SystemMessage(content="you are a helpful agent"),  # 5
            HumanMessage(content="hello"),  # 1
            AIMessage(content="hi there"),  # 2
        ]
        total = count_messages_tokens(msgs)
        assert total == 5 + 1 + 2

    def test_empty_list(self) -> None:
        assert count_messages_tokens([]) == 0

    def test_uses_metadata_where_present(self) -> None:
        msgs = [
            HumanMessage(content="hello"),
            AIMessage(
                content="ignored long content here",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 3,
                    "total_tokens": 53,
                },
            ),
        ]
        # 1 (hello via tiktoken) + 3 (from metadata)
        assert count_messages_tokens(msgs) == 4


class TestEncoderCache:
    def test_reset_clears_cache(self) -> None:
        from backend.v.utils.tokens import _encoder_for

        _encoder_for("gpt-4")
        cache_info_before = _encoder_for.cache_info()
        assert cache_info_before.currsize >= 1
        reset_encoder_cache()
        assert _encoder_for.cache_info().currsize == 0
