"""Unit tests for ``backend.v.agents.emotion`` (Phase-3 Group F)."""

from __future__ import annotations

import pytest

from backend.v.agents.emotion import (
    HttpEmotionDetector,
    KeywordEmotionDetector,
    should_preempt_handoff,
)


@pytest.fixture
def detector() -> KeywordEmotionDetector:
    return KeywordEmotionDetector()


class TestKeywordEmotionDetector:
    async def test_empty_string(self, detector: KeywordEmotionDetector) -> None:
        assert await detector.score("") == 0.0

    async def test_polite_message(self, detector: KeywordEmotionDetector) -> None:
        # Standard customer enquiry — should not trigger.
        score = await detector.score("请问我的订单 ORD123 的物流状态是什么？")
        assert score < 0.4

    async def test_explicit_profanity_high_score(self, detector: KeywordEmotionDetector) -> None:
        # Clear compound profanity — matches `操你|草你妈|...`.
        score = await detector.score("你们草你妈的服务")
        assert score >= 0.8

    async def test_exclamation_burst(self, detector: KeywordEmotionDetector) -> None:
        # Three or more exclamation marks triggers a medium signal.
        score = await detector.score("快点退款!!!")
        assert score >= 0.5

    async def test_complaint_word(self, detector: KeywordEmotionDetector) -> None:
        score = await detector.score("我要投诉你们")
        assert score >= 0.4

    async def test_courtesy_offset_reduces_score(self, detector: KeywordEmotionDetector) -> None:
        # "谢谢" alongside a mild complaint should lower the score.
        rude = await detector.score("差评")
        polite_rude = await detector.score("差评，谢谢你")
        assert polite_rude < rude

    async def test_score_clamped_to_1(self, detector: KeywordEmotionDetector) -> None:
        # Multiple anger patterns should not push above 1.
        score = await detector.score("草你妈骗人!!!")
        assert score <= 1.0

    async def test_score_clamped_to_0(self, detector: KeywordEmotionDetector) -> None:
        # Heavy courtesy offsetting a weak signal should not go below 0.
        score = await detector.score("差评 谢谢 感谢 麻烦你")
        assert score >= 0.0


class TestShouldPreemptHandoff:
    async def test_above_threshold_preempts(self, detector: KeywordEmotionDetector) -> None:
        # Profanity → score ≥ 0.8 → should preempt.
        assert (
            await should_preempt_handoff(
                "草你妈你们太差了",
                detector=detector,
                threshold=0.80,
            )
            is True
        )

    async def test_below_threshold_does_not_preempt(self, detector: KeywordEmotionDetector) -> None:
        assert (
            await should_preempt_handoff(
                "请问怎么退款？",
                detector=detector,
                threshold=0.80,
            )
            is False
        )

    async def test_threshold_1_never_preempts(self, detector: KeywordEmotionDetector) -> None:
        # threshold=1.0 disables pre-emption.
        assert (
            await should_preempt_handoff(
                "草你妈",
                detector=detector,
                threshold=1.0,
            )
            is False
        )

    async def test_threshold_0_always_preempts(self, detector: KeywordEmotionDetector) -> None:
        # threshold=0.0 would preempt on any non-zero score.
        # A mild complaint should have score > 0.
        assert (
            await should_preempt_handoff(
                "差评",
                detector=detector,
                threshold=0.0,
            )
            is True
        )


class TestHttpEmotionDetector:
    async def test_unreachable_endpoint_returns_zero(self) -> None:
        # No server on port 1 — the connection refused error should be
        # swallowed and a safe 0.0 returned.
        det = HttpEmotionDetector("http://127.0.0.1:1", timeout=0.05)
        assert await det.score("anything") == 0.0

    def test_url_trailing_slash_stripped(self) -> None:
        det = HttpEmotionDetector("http://x/", timeout=0.05)
        assert det._base_url == "http://x"
