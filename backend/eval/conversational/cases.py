"""Data structures for conversational retrieval eval cases.

A :class:`ConvCase` is one golden test case. Unlike the single-turn ``QaItem``,
it does not carry a fixed query — the query *emerges* from a multi-turn dialogue
the user simulator drives from ``persona`` + ``hidden_need`` + ``noise``. The
gold set is enumerated by :func:`enumerate_gold` from a machine-checkable
``constraint`` (so any-of scoring is reliable: the gold set is complete).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.eval.common import DATA_DIR, Product

CONV_CASES_PATH = DATA_DIR / "conv_cases.jsonl"

# Customer persona archetypes — drive the user simulator's tone/behaviour.
# Validated against the seed dialogues (SEED_DIALOGUES.md).
PERSONAS = [
    "budget_tight",  # 预算紧张，反复比价、强调别太贵
    "in_a_hurry",  # 急需，没耐心，信息挤牙膏
    "clueless",  # 不懂行，一堆生活细节，需客服引导
    "frustrated",  # 带怨气（上次买贵了/被坑），强调性价比
    "gifting",  # 给别人买，需求飘，自己不懂
    "terse",  # 惜字如金，全程被动，一个字一个字挤
    "rambler",  # 闲聊跑题严重，正经诉求埋在废话里
    "wishy_washy",  # 边聊边改主意，需求前后变化
]


@dataclass
class ConvCase:
    """One conversational retrieval golden case.

    Attributes:
        case_id: Stable id, e.g. ``CONV0001``.
        constraint: Machine-checkable gold predicate. Keys: ``category`` (str),
            ``price_min`` (float), ``price_max`` (float). Gold = all products
            satisfying it (see :func:`enumerate_gold`).
        gold_source_ids: Product ids satisfying ``constraint`` (the complete
            any-of gold set). Filled by the generator via enumeration.
        persona: One of :data:`PERSONAS`; drives simulator tone.
        hidden_need: The customer's real ask in plain words — fed to the
            simulator, but the simulator must NOT name a specific product and
            should reveal the need gradually.
        noise: Off-topic / emotional chatter the simulator weaves in.
        max_turns: Hard cap on customer turns before the dialogue is force-ended.
    """

    case_id: str
    constraint: dict[str, Any]
    gold_source_ids: list[str]
    persona: str
    hidden_need: str
    noise: str = ""
    max_turns: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "constraint": self.constraint,
            "gold_source_ids": self.gold_source_ids,
            "persona": self.persona,
            "hidden_need": self.hidden_need,
            "noise": self.noise,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConvCase:
        return cls(
            case_id=d["case_id"],
            constraint=d["constraint"],
            gold_source_ids=list(d["gold_source_ids"]),
            persona=d["persona"],
            hidden_need=d["hidden_need"],
            noise=d.get("noise", ""),
            max_turns=int(d.get("max_turns", 6)),
        )


def enumerate_gold(products: list[Product], constraint: dict[str, Any]) -> list[str]:
    """Return all product ids satisfying ``constraint`` (the complete gold set).

    Constraint keys (all optional, AND-combined):
        category: exact category match.
        price_min / price_max: inclusive price band.

    This is the heart of reliable any-of scoring: the gold set is whatever the
    catalogue actually contains under the predicate, enumerated by code — never
    hand-picked or LLM-judged, so it cannot silently under-label.
    """
    category = constraint.get("category")
    price_min = constraint.get("price_min")
    price_max = constraint.get("price_max")
    out: list[str] = []
    for p in products:
        if category is not None and p.category != category:
            continue
        if price_min is not None and p.price < float(price_min):
            continue
        if price_max is not None and p.price > float(price_max):
            continue
        out.append(p.product_id)
    return out


@dataclass
class ConvResult:
    """Outcome of running one ConvCase through the agent."""

    case_id: str
    persona: str
    category: str
    gold_source_ids: list[str]
    retrieved_source_ids: list[str]  # union across all search calls in the dialogue
    hit: bool  # retrieved ∩ gold ≠ ∅
    hit_turn: int  # 1-based customer turn where gold first retrieved, 0 if never
    total_turns: int
    # ── feedback-loop metrics (one graph run yields all of these) ──────────
    task_completed: bool = False  # LLM-judge: was the customer's need satisfied?
    task_reason: str = ""  # judge's one-line rationale
    tool_calls: int = 0  # tool invocations across the dialogue
    tool_errors: int = 0  # of those, how many returned an error
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0  # wall time for the whole multi-turn case
    transcript: list[dict[str, str]] = field(default_factory=list)  # role/content per turn

    @property
    def tool_success_rate(self) -> float:
        """Fraction of tool calls that did NOT error. 1.0 when no calls."""
        if self.tool_calls == 0:
            return 1.0
        return (self.tool_calls - self.tool_errors) / self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "persona": self.persona,
            "category": self.category,
            "gold_source_ids": self.gold_source_ids,
            "retrieved_source_ids": self.retrieved_source_ids,
            "hit": self.hit,
            "hit_turn": self.hit_turn,
            "total_turns": self.total_turns,
            "task_completed": self.task_completed,
            "task_reason": self.task_reason,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "transcript": self.transcript,
        }


def load_conv_cases(path: Path = CONV_CASES_PATH) -> list[ConvCase]:
    from backend.eval.common import load_jsonl

    return [ConvCase.from_dict(d) for d in load_jsonl(path)]
