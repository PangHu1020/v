"""LangGraph nodes: enter -> compress -> agent (with tools) -> exit.

``enter_node`` assembles the prompt in **cold → hot volatility tiers** so the
leading token prefix stays byte-identical across turns (and, for the cold
tier, across sessions) — maximizing DeepSeek/Qwen automatic prefix-cache hits.

Message layout after ``enter_node`` (turn 1):

    [0] COLD  SystemMessage — base persona + channel + <available_skills> catalog.
              Customer-independent; identical across every session on this
              channel → shared prefix cache.
    [1] WARM  SystemMessage — <customer_profile> + <recent_events> +
              <session_memory>. Customer-specific; frozen for the session until
              a compression event rebuilds it.
    [2..] HOT conversation history — append-only.

Tool / MCP schemas ride in the request's ``tools`` param (bound once per build),
which sits in the same cached prefix. Skills use **progressive disclosure**: the
cold catalog advertises names+descriptions; the model pulls a full SOP body on
demand via the ``load_skill`` tool, which appends to history (hot) rather than
mutating the frozen system prefix.

Conflict-resolution rule (unchanged): hotter content shadows colder on
conflicts — session memory > recent events > profile.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
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
from backend.v.tools import calculator, load_skill, recall_memory, search, subagent
from backend.v.utils.logging import get_logger

_log = get_logger("agents.nodes")

AGENT_TOOLS: list = [calculator, search, recall_memory, subagent, load_skill]
"""Tools bound to the main agent."""


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


def _build_cold_layer(channel: str, skill_registry: SkillRegistry | None) -> str:
    """Customer-independent system text: persona + channel + skill catalog.

    Identical for every customer on the same channel, so the token prefix is
    shared across sessions in the provider's cache.
    """
    sections = [build_main_system_prompt(channel)]
    if skill_registry is not None and len(skill_registry):
        catalog = skill_registry.render_catalog(channel=channel)
        if catalog:
            sections.append(f"<available_skills>\n{catalog}\n</available_skills>")
    return "\n\n".join(sections)


def _build_warm_layer(state: CustomerServiceState) -> str:
    """Customer-specific memory text: profile + recent events + session memory."""
    sections: list[str] = []
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
    return "\n\n".join(sections)


async def enter_node(
    state: CustomerServiceState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Inject the cold + warm system layers on the first turn of a thread.

    Emits two ordered ``SystemMessage``s (cold, then warm) and reorders the
    turn's existing messages to sit *after* them, so the cache-stable system
    prefix leads the request. On later turns a leading SystemMessage already
    exists (kept by the checkpointer) and we no-op — the prefix stays frozen
    until a compression event rebuilds the warm layer.
    """
    messages = list(state.get("messages", []))
    if any(isinstance(m, SystemMessage) for m in messages):
        return {}
    cfg = config.get("configurable", {}) if config else {}
    channel = state.get("channel", "")

    cold_text = _build_cold_layer(channel, cfg.get("skill_registry"))
    warm_text = _build_warm_layer(state)

    new_messages: list[Any] = [SystemMessage(content=cold_text)]
    if warm_text:
        new_messages.append(SystemMessage(content=warm_text))

    # Reorder so the system layers lead: drop the existing (pre-system) messages
    # and re-append them after, with fresh ids (add_messages assigns new uuids).
    removals = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)]
    readded = [m.model_copy(update={"id": None}) for m in messages]

    _log.info(
        "agents.enter_node.injected",
        cold_len=len(cold_text),
        warm_len=len(warm_text),
        reordered=len(messages),
    )
    return {"messages": [*removals, *new_messages, *readded]}


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

    # A clarify directive (set by clarify_node) is injected for THIS call only —
    # never written back to state, so the persisted history stays clean.
    # Appended at the TAIL (append-only) to keep the prompt prefix byte-stable
    # for KV-cache reuse — inserting it near the front would shift every later
    # token and invalidate the cached prefix. It's a HumanMessage, not a
    # SystemMessage: the system block is frozen at the head (cold+warm only),
    # and strict chat templates (vllm-hosted Qwen) reject a SystemMessage that
    # follows a Human/AI turn ("System message must be at the beginning").
    call_messages = list(state.get("messages", []))
    directive = state.get("intent_directive")
    if directive:
        call_messages.append(HumanMessage(content=directive))

    result = await caller.chat(
        "main_primary",
        call_messages,
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
