"""Unit tests for the reply-segmentation helper."""

from __future__ import annotations

from backend.v.utils.reply import split_reply_segments


class TestSplitReplySegments:
    def test_empty_returns_empty_list(self) -> None:
        assert split_reply_segments("") == []

    def test_single_paragraph_one_segment(self) -> None:
        assert split_reply_segments("您好，已为您查询订单") == ["您好，已为您查询订单"]

    def test_blank_line_splits_into_two(self) -> None:
        reply = "第一步：登录账户\n\n第二步：进入订单页"
        assert split_reply_segments(reply) == ["第一步：登录账户", "第二步：进入订单页"]

    def test_multiple_blank_lines_collapse(self) -> None:
        reply = "段一\n\n\n\n段二"
        assert split_reply_segments(reply) == ["段一", "段二"]

    def test_trims_each_segment(self) -> None:
        reply = "  hello  \n\n  world  "
        assert split_reply_segments(reply) == ["hello", "world"]

    def test_single_newline_within_paragraph_preserved(self) -> None:
        reply = "line1\nline2\n\nnext"
        assert split_reply_segments(reply) == ["line1\nline2", "next"]

    def test_blank_only_segments_dropped(self) -> None:
        reply = "\n\n实际内容\n\n   \n\n"
        assert split_reply_segments(reply) == ["实际内容"]
