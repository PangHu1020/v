"""Unit tests for conversational RAGAS scoring helpers (pure functions only).

The ``score_cases`` coroutine hits a live judge endpoint, so it is exercised by
the offline eval, not here. These tests lock the deterministic helpers:
context extraction, reference synthesis, and the scorable gate.
"""

from __future__ import annotations

from backend.eval.common import Product
from backend.eval.conversational.ragas_score import (
    build_reference,
    extract_contexts,
    scorable,
)


class TestExtractContexts:
    def test_pulls_text_after_marker(self) -> None:
        out = "检索结果（按相关度排序）：\n- [product:P001] iPhone 15 便宜手机\n- [product:P002] 小米 续航强"
        assert extract_contexts([out]) == ["iPhone 15 便宜手机", "小米 续航强"]

    def test_dedupes_across_tool_calls(self) -> None:
        a = "- [product:P001] iPhone"
        b = "- [product:P001] iPhone\n- [product:P003] 华为"
        assert extract_contexts([a, b]) == ["iPhone", "华为"]

    def test_keeps_faq_lines(self) -> None:
        out = "- [faq:FAQ001] 退货政策7天无理由"
        assert extract_contexts([out]) == ["退货政策7天无理由"]

    def test_ignores_header_and_blank(self) -> None:
        assert extract_contexts(["检索结果（按相关度排序）："]) == []
        assert extract_contexts(["", "（未找到相关条目）"]) == []

    def test_empty_input(self) -> None:
        assert extract_contexts([]) == []


def _prod(pid: str, name: str, price: float, desc: str = "好用") -> Product:
    return Product(
        product_id=pid,
        product_name=name,
        category="手机数码",
        brand="某牌",
        price=price,
        description=desc,
    )


class TestBuildReference:
    def test_lists_gold_products(self) -> None:
        prods = {"P001": _prod("P001", "iPhone", 5999.0, "拍照强")}
        ref = build_reference(["P001"], prods)
        assert "iPhone" in ref
        assert "5999" in ref
        assert "拍照强" in ref
        assert ref.startswith("符合客户需求的商品")

    def test_skips_unknown_ids(self) -> None:
        prods = {"P001": _prod("P001", "iPhone", 5999.0)}
        ref = build_reference(["P001", "P999"], prods)
        assert "iPhone" in ref
        assert ref.count("- ") == 1  # only the known product

    def test_empty_when_no_gold_resolves(self) -> None:
        assert build_reference(["P999"], {}) == ""
        assert build_reference([], {}) == ""


class TestScorable:
    def test_needs_response_and_contexts(self) -> None:
        assert scorable({"response": "r", "retrieved_contexts": ["c"]}) is True

    def test_no_response_unscorable(self) -> None:
        assert scorable({"response": "", "retrieved_contexts": ["c"]}) is False

    def test_no_contexts_unscorable(self) -> None:
        assert scorable({"response": "r", "retrieved_contexts": []}) is False

    def test_missing_keys_unscorable(self) -> None:
        assert scorable({}) is False
