"""Unit tests for the cascade confidence gate (pure, no IO)."""

from __future__ import annotations

from backend.v.rag.gate import GateParams, confidence_signals, should_exit


class TestConfidenceSignals:
    def test_empty(self) -> None:
        s = confidence_signals([])
        assert s == {"top1": 0.0, "top2": 0.0, "margin": 0.0, "rel_margin": 0.0, "mean": 0.0}

    def test_single_result_is_max_separated(self) -> None:
        s = confidence_signals([0.8])
        assert s["top1"] == 0.8
        assert s["top2"] == 0.0
        assert s["rel_margin"] > 0.999  # lone result → fully separated (1.0 modulo eps)

    def test_margin_and_rel_margin(self) -> None:
        s = confidence_signals([0.9, 0.6, 0.5])
        assert abs(s["margin"] - 0.3) < 1e-9
        assert abs(s["rel_margin"] - (0.3 / 0.9)) < 1e-9

    def test_unsorted_input(self) -> None:
        s = confidence_signals([0.5, 0.9, 0.6])
        assert s["top1"] == 0.9
        assert s["top2"] == 0.6


class TestShouldExit:
    def test_exits_on_clear_separation(self) -> None:
        g = GateParams(floor=0.5, rel_margin=0.1, min_k=1)
        assert should_exit([0.9, 0.6], g) is True  # rel_margin ~0.33 >= 0.1

    def test_escalates_when_ambiguous(self) -> None:
        g = GateParams(floor=0.5, rel_margin=0.2, min_k=1)
        # top1=0.82 top2=0.80 → rel_margin ~0.024 < 0.2 → escalate
        assert should_exit([0.82, 0.80, 0.79], g) is False

    def test_escalates_below_floor(self) -> None:
        g = GateParams(floor=0.7, rel_margin=0.0, min_k=1)
        assert should_exit([0.5], g) is False  # below floor

    def test_floor_only_when_margin_zero(self) -> None:
        g = GateParams(floor=0.6, rel_margin=0.0, min_k=1)
        # rel_margin=0 disables separation check; floor passes
        assert should_exit([0.65, 0.64], g) is True

    def test_min_k_blocks_exit(self) -> None:
        g = GateParams(floor=0.0, rel_margin=0.0, min_k=3)
        assert should_exit([0.9, 0.8], g) is False  # only 2 results < min_k
        assert should_exit([0.9, 0.8, 0.7], g) is True

    def test_scale_invariance(self) -> None:
        # Same relative separation at different absolute scales → same decision.
        g = GateParams(floor=0.0, rel_margin=0.15, min_k=1)
        assert should_exit([0.90, 0.72], g) == should_exit([0.45, 0.36], g)
