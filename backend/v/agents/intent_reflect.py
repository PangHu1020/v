"""Intent classification + reflection nodes.

**Intent node** — cheap LLM call (flash tier) that scores the customer's
latest message as a probability distribution over five base intents, then
*derives* the routing state from that distribution + per-category thresholds:

- single: top intent clears its threshold and leads top-2 by ``ambiguity_margin``.
- mix: top TWO both clear their thresholds → handle + reflect.
- ambiguous: top-1 below threshold, or top-2 within margin → clarify (capped),
  else fall back to acting on top-1.

**Clarify node** — when ambiguous and under the clarify cap, sets a transient
directive telling the agent to answer the top-1 guess AND confirm intent at the
end (the directive is injected for one LLM call only, never persisted to
``messages``, so the cache-stable prefix is untouched).

**Reflection node** — after the agent produces a reply, checks whether it
references facts NOT present in any ToolMessage from this turn. Retries up to
``MAX_REFLECTION_RETRIES`` times before passing through.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from backend.v.agents.prompts import (
    INTENT_CLARIFY_DIRECTIVE,
    INTENT_MIX_DIRECTIVE,
    INTENT_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
)
from backend.v.agents.state import CustomerServiceState
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import get_logger

_log = get_logger("agents.intent_reflect")

MAX_REFLECTION_RETRIES = 2

Intent = Literal["refund", "logistics", "complaint", "general", "chitchat"]
# Single intents that warrant hallucination reflection on their own
# (high-stakes, tool-backed). chitchat/general never reflect; mix always does.
_REFLECTION_INTENTS = frozenset({"refund", "logistics"})
# Human-readable labels for the clarifying question.
_INTENT_CN = {
    "refund": "退款售后",
    "logistics": "物流查询",
    "complaint": "投诉反馈",
    "general": "商品咨询",
    "chitchat": "随便聊聊",
}

# Fallback thresholds/margin when no IntentSettings is on config (e.g. tests).
_DEFAULT_THRESHOLDS = {
    "refund": 0.6,
    "logistics": 0.55,
    "complaint": 0.6,
    "general": 0.45,
    "chitchat": 0.5,
}
_DEFAULT_MARGIN = 0.15
_DEFAULT_MAX_CLARIFY = 1


class IntentScores(BaseModel):
    """Probability distribution over the five base intents (LLM output)."""

    refund: float = Field(default=0.0, ge=0.0, le=1.0)
    logistics: float = Field(default=0.0, ge=0.0, le=1.0)
    complaint: float = Field(default=0.0, ge=0.0, le=1.0)
    general: float = Field(default=0.0, ge=0.0, le=1.0)
    chitchat: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


def _ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Intents sorted by probability desc (name asc tiebreak for determinism)."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _resolve_intent_params(cfg: dict) -> tuple[dict[str, float], float, int]:
    """Pull (thresholds, margin, max_clarify) from settings.intent or defaults."""
    settings = cfg.get("settings")
    intent_cfg = getattr(settings, "intent", None)
    if intent_cfg is None:
        return dict(_DEFAULT_THRESHOLDS), _DEFAULT_MARGIN, _DEFAULT_MAX_CLARIFY
    thresholds = {k: intent_cfg.threshold_for(k) for k in _DEFAULT_THRESHOLDS}
    return thresholds, float(intent_cfg.ambiguity_margin), int(intent_cfg.max_clarify_turns)


def _derive_routing(
    scores: dict[str, float],
    *,
    thresholds: dict[str, float],
    margin: float,
    clarify_count: int,
    max_clarify: int,
) -> dict[str, Any]:
    """Derive routing state from the distribution. Pure function (easy to test).

    Priority: ambiguous(top-1 below threshold) → mix(top-2 both clear) →
    ambiguous(close margin) → single. Returns intent / secondary_intent /
    is_mix / needs_clarify / needs_reflection.
    """
    ranked = _ranked(scores)
    (top1, p1), (top2, p2) = ranked[0], ranked[1]

    can_clarify = clarify_count < max_clarify
    top1_clears = p1 >= thresholds[top1]
    top2_clears = p2 >= thresholds[top2]
    close = (p1 - p2) < margin

    if not top1_clears:
        ambiguous, is_mix = True, False
    elif top2_clears:
        ambiguous, is_mix = False, True
    elif close:
        ambiguous, is_mix = True, False
    else:
        ambiguous, is_mix = False, False

    needs_clarify = ambiguous and can_clarify
    needs_reflection = is_mix or top1 in _REFLECTION_INTENTS
    return {
        "intent": top1,
        "secondary_intent": top2,
        "is_mix": is_mix,
        "needs_clarify": needs_clarify,
        "needs_reflection": needs_reflection,
    }


async def intent_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Score the latest message and derive mix / ambiguous / single routing."""
    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    # Reset any stale directive from a prior turn so it never leaks forward.
    base: dict[str, Any] = {"intent_directive": ""}
    if caller is None:
        return {**base, "intent": "general"}

    started = time.perf_counter()
    messages = state.get("messages", [])
    latest_text = ""
    for m in reversed(messages):
        if not isinstance(m, (SystemMessage, AIMessage, ToolMessage)):
            latest_text = m.content if isinstance(m.content, str) else ""
            break
    if not latest_text:
        return {**base, "intent": "general"}

    prompt = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"<message>\n{latest_text}\n</message>"),
    ]
    try:
        result = await caller.chat("summary", prompt, structured=IntentScores, config=config)
        scores_model = result.parsed if isinstance(result.parsed, IntentScores) else IntentScores()
    except Exception as exc:
        _log.warning("intent_node.failed", error=type(exc).__name__)
        return {**base, "intent": "general"}

    scores = scores_model.as_dict()
    thresholds, margin, max_clarify = _resolve_intent_params(cfg)
    routing = _derive_routing(
        scores,
        thresholds=thresholds,
        margin=margin,
        clarify_count=int(state.get("clarify_count") or 0),
        max_clarify=max_clarify,
    )

    _log.info(
        "intent_node.classified",
        intent=routing["intent"],
        secondary=routing["secondary_intent"],
        is_mix=routing["is_mix"],
        needs_clarify=routing["needs_clarify"],
        scores=scores,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    if routing["is_mix"]:
        base["intent_directive"] = INTENT_MIX_DIRECTIVE.format(
            top1_cn=_INTENT_CN.get(routing["intent"], routing["intent"]),
            top2_cn=_INTENT_CN.get(routing["secondary_intent"], routing["secondary_intent"]),
        )
    return {**base, **routing, "intent_scores": scores}


async def clarify_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Set a transient directive so the agent answers top-1 AND confirms intent.

    No LLM call here — the directive is consumed by ``agent_node`` for a single
    invocation. Increments ``clarify_count`` so the next ambiguous turn falls
    back to acting on top-1 rather than asking again.
    """
    top1 = state.get("intent", "general")
    top2 = state.get("secondary_intent", "general")
    directive = INTENT_CLARIFY_DIRECTIVE.format(
        top1=top1,
        top2=top2,
        top1_cn=_INTENT_CN.get(top1, top1),
        top2_cn=_INTENT_CN.get(top2, top2),
    )
    _log.info("clarify_node.directive_set", top1=top1, top2=top2)
    return {
        "intent_directive": directive,
        "clarify_count": int(state.get("clarify_count") or 0) + 1,
    }


class ReflectionResult(BaseModel):
    passes: bool = True
    issues: list[str] = Field(default_factory=list)


def _collect_tool_facts(messages: list) -> str:
    """Concatenate tool result payloads from the current turn.

    Handles both the JSON envelope emitted by :class:`ToolOrchestrator`
    (``{"ok": bool, "data": "..."}`` shape) and legacy plain-string content
    for backward compatibility.  Guard-block markers are always skipped.
    """
    import json as _json

    facts: list[str] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if not content:
            continue
        # Try JSON envelope first (new path).
        try:
            d = _json.loads(content)
            if isinstance(d, dict):
                if d.get("ok") and d.get("data"):
                    facts.append(d["data"])
                # ok=false → skip (error, not a fact to cite)
                continue
        except (_json.JSONDecodeError, TypeError):
            pass
        # Legacy plain string (no envelope).
        if not content.startswith("[tool_guard]"):
            facts.append(content)
    return "\n".join(facts) or "（本轮无工具调用结果，回复不应引用任何具体事实。）"


def _last_ai_reply(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else ""
    return ""


async def reflection_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Check the agent's reply for hallucinations; retry if needed."""
    retries = int(state.get("reflection_retries") or 0)
    if retries >= MAX_REFLECTION_RETRIES:
        _log.info("reflection_node.max_retries_reached")
        return {}

    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    if caller is None:
        return {}

    started = time.perf_counter()
    messages = state.get("messages", [])
    reply = _last_ai_reply(messages)
    if not reply:
        return {}

    tool_facts = _collect_tool_facts(messages)

    from langchain_core.messages import HumanMessage

    prompt = [
        SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"<tool_facts>\n{tool_facts}\n</tool_facts>\n\n<ai_reply>\n{reply}\n</ai_reply>"
            )
        ),
    ]
    try:
        result = await caller.chat("summary", prompt, structured=ReflectionResult, config=config)
        rr = result.parsed if isinstance(result.parsed, ReflectionResult) else ReflectionResult()
    except Exception as exc:
        _log.warning("reflection_node.failed", error=type(exc).__name__)
        return {}

    if rr.passes:
        _log.info(
            "reflection_node.passed",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return {}

    _log.warning(
        "reflection_node.failed_check",
        issues=rr.issues,
        retry=retries + 1,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"reflection_retries": retries + 1, "reflection_failed": True}
