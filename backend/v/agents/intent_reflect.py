"""Intent classification + reflection nodes.

**Intent node** — cheap LLM call (flash tier) that classifies the
customer's latest message into one of four intents.

**Reflection node** — after the agent produces a reply, checks whether
it references facts NOT present in any ToolMessage from this turn.
Retries up to ``MAX_REFLECTION_RETRIES`` times before passing through.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from backend.v.agents.prompts import INTENT_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from backend.v.agents.state import CustomerServiceState
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import get_logger

_log = get_logger("agents.intent_reflect")

MAX_REFLECTION_RETRIES = 2

Intent = Literal["refund", "logistics", "complaint", "general"]


class IntentResult(BaseModel):
    intent: Intent = "general"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


async def intent_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Classify the customer's latest message into one of four intents."""
    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    if caller is None:
        return {"intent": "general"}

    started = time.perf_counter()
    messages = state.get("messages", [])
    latest_text = ""
    for m in reversed(messages):
        if not isinstance(m, (SystemMessage, AIMessage, ToolMessage)):
            latest_text = m.content if isinstance(m.content, str) else ""
            break
    if not latest_text:
        return {"intent": "general"}

    from langchain_core.messages import HumanMessage

    prompt = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"<message>\n{latest_text}\n</message>"),
    ]
    try:
        result = await caller.chat("summary", prompt, structured=IntentResult, config=config)
        ir = result.parsed if isinstance(result.parsed, IntentResult) else IntentResult()
    except Exception as exc:
        _log.warning("intent_node.failed", error=type(exc).__name__)
        return {"intent": "general"}

    _log.info(
        "intent_node.classified",
        intent=ir.intent,
        confidence=ir.confidence,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return {
        "intent": ir.intent,
        "needs_reflection": ir.intent in {"refund", "logistics"},
    }


class ReflectionResult(BaseModel):
    passes: bool = True
    issues: list[str] = Field(default_factory=list)


def _collect_tool_facts(messages: list) -> str:
    """Concatenate all ToolMessage contents from the current turn."""
    facts: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content and not content.startswith("[tool_guard]"):
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
