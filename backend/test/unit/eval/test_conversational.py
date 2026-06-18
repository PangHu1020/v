"""Unit tests for conversational retrieval eval components."""

from __future__ import annotations

from backend.eval.common import Product
from backend.eval.conversational.cases import enumerate_gold
from backend.eval.conversational.scoring import extract_retrieved_ids, score_hit


class TestEnumerateGold:
    """Test constraint-based gold enumeration (completeness is key)."""

    def test_category_only(self) -> None:
        products = [
            Product("P001", "A", "手机", "X", 1000, "a"),
            Product("P002", "B", "手机", "Y", 2000, "b"),
            Product("P003", "C", "鞋靴", "Z", 500, "c"),
        ]
        gold = enumerate_gold(products, {"category": "手机"})
        assert set(gold) == {"P001", "P002"}

    def test_price_band(self) -> None:
        products = [
            Product("P001", "A", "手机", "X", 1000, "a"),
            Product("P002", "B", "手机", "Y", 2000, "b"),
            Product("P003", "C", "手机", "Z", 3000, "c"),
        ]
        gold = enumerate_gold(products, {"price_min": 1500, "price_max": 2500})
        assert gold == ["P002"]

    def test_category_and_price(self) -> None:
        products = [
            Product("P001", "A", "手机", "X", 1000, "a"),
            Product("P002", "B", "手机", "Y", 2000, "b"),
            Product("P003", "C", "鞋靴", "Z", 1500, "c"),
        ]
        gold = enumerate_gold(products, {"category": "手机", "price_min": 500, "price_max": 1500})
        assert gold == ["P001"]

    def test_empty_gold(self) -> None:
        products = [Product("P001", "A", "手机", "X", 5000, "a")]
        gold = enumerate_gold(products, {"category": "手机", "price_min": 100, "price_max": 200})
        assert gold == []

    def test_price_band_inclusive_both_ends(self) -> None:
        # Band slicing in the generator uses [first.price, last.price]; the
        # endpoints MUST be included or the boundary products drop from gold.
        products = [
            Product("P001", "A", "手机", "X", 1000, "a"),
            Product("P002", "B", "手机", "Y", 2000, "b"),
            Product("P003", "C", "手机", "Z", 3000, "c"),
        ]
        gold = enumerate_gold(products, {"price_min": 1000, "price_max": 3000})
        assert set(gold) == {"P001", "P002", "P003"}  # both endpoints kept

    def test_preserves_catalogue_order(self) -> None:
        products = [
            Product("P003", "C", "手机", "Z", 3000, "c"),
            Product("P001", "A", "手机", "X", 1000, "a"),
        ]
        gold = enumerate_gold(products, {"category": "手机"})
        assert gold == ["P003", "P001"]  # input order, not sorted


class TestScoring:
    """Test source_id extraction and any-of hit scoring."""

    def test_extract_retrieved_ids(self) -> None:
        tool_outputs = [
            "检索结果：\n- [product:P001] 商品A\n- [product:P002] 商品B",
            "- [faq:FAQ001] 问答",
            "未找到",
            "- [product:P003] 商品C\n- [product:P001] 商品A重复",
        ]
        ids = extract_retrieved_ids(tool_outputs, source_type="product")
        # de-duplicated, first-seen order
        assert ids == ["P001", "P002", "P003"]

    def test_extract_no_products(self) -> None:
        tool_outputs = ["未找到", "- [faq:FAQ001] 问答"]
        ids = extract_retrieved_ids(tool_outputs, source_type="product")
        assert ids == []

    def test_score_hit_any_of(self) -> None:
        assert score_hit(["P001", "P002"], ["P001", "P003"]) is True  # P001 in gold
        assert score_hit(["P001"], ["P002", "P003"]) is False
        assert score_hit([], ["P001"]) is False

    def test_score_hit_empty_gold(self) -> None:
        # edge case: empty gold should never hit
        assert score_hit(["P001"], []) is False
