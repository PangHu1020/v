"""LangGraph nodes: enter -> compress -> agent (with tools) -> exit.

``enter_node`` builds the per-turn system prompt by stacking layers:

1. Base persona + channel context (static, from agents/prompts).
2. ``<customer_profile>``  — full long-term profile (every populated
   canonical field + every extras key + notes if present).
3. ``<recent_events>``     — most-recent N rows of ``agent.event_memory``.
4. ``<session_memory>``    — current session's working-memory entries.
5. ``<sops>``              — matched skills / SOPs.

Layers stacked in that order so the prompt-level rule "later layers
override earlier ones on conflicts" lines up with the storage tier
freshness: profile (oldest) < events (medium-term) < session memory
(this session) < SOPs (current intent guidance).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.prompts import build_main_system_prompt
from backend.v.agents.state import CustomerServiceState
from backend.v.memory.prompts import (
    render_recent_events_for_prompt,
    render_session_memory_for_prompt,
)
from backend.v.memory.types import MemoryEntry, UserProfile
from backend.v.models.llm_caller import LLMCaller
from backend.v.skills import SkillRegistry
from backend.v.tools import calculator, recall_memory, search, subagent
from backend.v.utils.logging import get_logger

_log = get_logger("agents.nodes")

AGENT_TOOLS: list = [calculator, search, recall_memory, subagent]
"""Tools bound to the main agent."""


def _latest_human_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else ""
    return ""


def _render_profile(profile: dict[str, Any] | None) -> str:
    """Render the full user_profile as a ``<customer_profile>`` block.

    Every populated canonical field becomes its own sub-tag; ``extras``
    becomes one ``<extras>`` block of ``<extra key="...">`` entries;
    ``notes`` becomes a ``<notes>`` block. Missing values are omitted
    so an empty profile yields an empty string and the caller can
    drop the section cleanly.

    Validates against :class:`UserProfile` first so an unexpected
    profile shape doesn't blow up the turn — we fall back to a flat
    JSONB-ish dump in that case.
    """
    if not profile:
        return ""

    try:
        validated = UserProfile.model_validate(profile)
    except Exception:
        # Render whatever keys are present; the layered shadowing rule
        # in the prompt still applies.
        kvs = [f"  <{k}>{v}</{k}>" for k, v in profile.items() if v not in (None, "", [], {})]
        return "<customer_profile>\n" + "\n".join(kvs) + "\n</customer_profile>" if kvs else ""

    bits: list[str] = []
    canonical = (
        ("customer_name", validated.customer_name),
        ("preferred_salutation", validated.preferred_salutation),
        ("preferred_language", validated.preferred_language),
        ("member_level", validated.member_level),
        ("response_style", validated.response_style),
    )
    for tag, value in canonical:
        if value:
            bits.append(f"  <{tag}>{value}</{tag}>")
    if validated.risk_flags:
        flags = ",".join(validated.risk_flags)
        bits.append(f"  <risk_flags>{flags}</risk_flags>")
    if validated.extras:
        extra_lines = [
            f'    <extra key="{k}">{v}</extra>'
            for k, v in validated.extras.items()
            if v not in (None, "")
        ]
        if extra_lines:
            bits.append("  <extras>\n" + "\n".join(extra_lines) + "\n  </extras>")
    if validated.notes:
        bits.append(f"  <notes>{validated.notes}</notes>")

    if not bits:
        return ""
    return "<customer_profile>\n" + "\n".join(bits) + "\n</customer_profile>"


def _maybe_skill_section(
    skill_registry: SkillRegistry | None,
    *,
    query: str,
    channel: str,
    top_k: int,
) -> str:
    """Return the rendered skills block (possibly empty) for appending."""
    if skill_registry is None or top_k <= 0 or not query:
        return ""
    matched = skill_registry.match(query, channel=channel, top_k=top_k)
    if not matched:
        return ""
    return skill_registry.render_for_prompt(matched)


async def enter_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Prepend the layered system prompt on first turn of a thread.

    On subsequent turns the messages already include a leading
    SystemMessage (the checkpointer keeps it) and we leave them
    untouched.
    """
    messages = state.get("messages", [])
    if any(isinstance(m, SystemMessage) for m in messages):
        return {}
    cfg = config.get("configurable", {}) if config else {}

    sections: list[str] = [build_main_system_prompt(state.get("channel", ""))]

    profile_block = _render_profile(state.get("user_profile"))
    if profile_block:
        sections.append(profile_block)

    events_block = render_recent_events_for_prompt(state.get("recent_events") or [])
    if events_block:
        sections.append(f"<recent_events>\n{events_block}\n</recent_events>")

    working: list[MemoryEntry] = state.get("working_memory") or []
    session_memory_block = render_session_memory_for_prompt(working)
    if session_memory_block:
        sections.append(session_memory_block)

    skill_section = _maybe_skill_section(
        cfg.get("skill_registry"),
        query=_latest_human_text(messages),
        channel=state.get("channel", ""),
        top_k=int(cfg.get("skill_top_k", 3)),
    )
    if skill_section:
        sections.append(f"<sops>\n{skill_section}\n</sops>")

    full_prompt = "\n\n".join(sections)
    _log.info(
        "agents.enter_node.injected",
        prompt_len=len(full_prompt),
        has_profile=bool(profile_block),
        events=len(state.get("recent_events") or []),
        working=len(working),
        has_skills=bool(skill_section),
    )
    return {"messages": [SystemMessage(content=full_prompt)]}


async def agent_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
    *,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Invoke the main-tier LLM and append its reply to ``messages``."""
    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    if caller is None:
        raise RuntimeError("agent_node requires config['configurable']['llm_caller']")

    started = time.perf_counter()
    bound_tools = tools if tools is not None else AGENT_TOOLS
    result = await caller.chat(
        "main_primary",
        list(state.get("messages", [])),
        tools=bound_tools,
        config=config,
    )
    _log.info(
        "agents.agent_node.replied",
        model=result.model,
        fallback_used=result.fallback_used,
        latency_ms=result.latency_ms,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        has_tool_calls=bool(getattr(result.message, "tool_calls", None)),
        tool_count=len(bound_tools),
    )
    return {"messages": [result.message]}


async def agent_fast_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Tool-free agent for general intent — no tool binding, lower latency."""
    return await agent_node(state, config, tools=[])


async def exit_node(state: CustomerServiceState) -> dict[str, Any]:
    """Terminal log point."""
    _log.debug(
        "agents.exit_node",
        session_id=state.get("session_id"),
        message_count=len(state.get("messages", [])),
    )
    return {}
