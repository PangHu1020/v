"""Unit tests for ``backend.eval.metrics`` (pure retrieval metrics)."""

from __future__ import annotations

import math

from backend.eval.retrieval.metrics import (
    aggregate,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestHitAtK:
    def test_hit_in_top_k(self) -> None:
        assert hit_at_k(["a", "b", "c"], {"b"}, 3) == 1.0

    def test_miss_outside_k(self) -> None:
        assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_empty_retrieved(self) -> None:
        assert hit_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_single_gold_found(self) -> None:
        assert recall_at_k(["a", "b"], {"a"}, 2) == 1.0

    def test_partial_multi_gold(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_empty_gold_is_zero(self) -> None:
        assert recall_at_k(["a"], set(), 2) == 0.0


class TestMrrAtK:
    def test_rank_one(self) -> None:
        assert mrr_at_k(["a", "b"], {"a"}, 2) == 1.0

    def test_rank_three(self) -> None:
        assert mrr_at_k(["x", "y", "a"], {"a"}, 3) == 1.0 / 3

    def test_outside_k_is_zero(self) -> None:
        assert mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


class TestNdcgAtK:
    def test_perfect_rank_one(self) -> None:
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == 1.0

    def test_discounted_lower_rank(self) -> None:
        # gold at rank 2: dcg = 1/log2(3); idcg = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / 1.0
        assert abs(ndcg_at_k(["x", "a", "y"], {"a"}, 3) - expected) < 1e-9

    def test_empty_gold(self) -> None:
        assert ndcg_at_k(["a"], set(), 3) == 0.0

    def test_two_gold_ideal(self) -> None:
        # both gold at top-2 -> perfect ordering -> 1.0
        assert abs(ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) - 1.0) < 1e-9


class TestAggregate:
    def test_averages_across_rows(self) -> None:
        rows = [
            {"retrieved": ["a", "b"], "gold": ["a"]},  # hit@1 = 1
            {"retrieved": ["x", "a"], "gold": ["a"]},  # hit@1 = 0, hit@3 = 1
        ]
        out = aggregate(rows, [1, 3])
        assert out["hit@1"] == 0.5
        assert out["hit@3"] == 1.0
        assert out["mrr@3"] == (1.0 + 0.5) / 2

    def test_empty_rows(self) -> None:
        assert aggregate([], [1, 3]) == {}
