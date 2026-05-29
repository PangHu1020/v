"""LangGraph nodes: enter -> compress -> agent (with tools) -> exit.

Phase-2 P0 binds the ``transfer_to_human`` tool so the agent can suspend
the graph via :func:`langgraph.types.interrupt` when handoff is needed.
The conditional edge in :mod:`backend.v.agents.graph` routes the agent's
tool calls through ``ToolNode`` and back; tool-less responses go to exit.

Phase-2 P4: ``enter_node`` consults the optional skill registry passed
through ``config['configurable']['skill_registry']`` and inlines any
matching SOPs into the system prompt.

Phase-3 Group C: ``enter_node`` also renders ``state['recent_events']``
(the last few medium-term ``agent.session_memory`` rows for this
identity) so the agent has cross-session continuity.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.state import CustomerServiceState
from backend.v.memory.event_memory import render_recent_events_for_prompt
from backend.v.models.llm_caller import LLMCaller
from backend.v.skills import SkillRegistry
from backend.v.tools import recall_memory, subagent, transfer_to_human
from backend.v.utils.logging import get_logger

_log = get_logger("agents.nodes")

AGENT_TOOLS: list = [transfer_to_human, recall_memory, subagent]
"""Tools bound to the main agent. Phase-2 P0 added ``transfer_to_human``;
Phase-2 P3 added ``recall_memory`` and ``subagent``."""


def _system_prompt(profile: dict[str, Any] | None, channel: str) -> str:
    """Render the base per-turn system prompt without skill / events injection.

    Returns an XML-tagged document. Tags are the LLM-friendly way to keep
    role, behavioral guidance, channel-specific context, and customer
    metadata visually separated — the model attends to them more reliably
    than to a wall of prose.
    """
    profile_block = ""
    if profile:
        bits: list[str] = []
        if name := profile.get("customer_name"):
            bits.append(f"  <name>{name}</name>")
        if level := profile.get("member_level"):
            bits.append(f"  <member_level>{level}</member_level>")
        if pref := profile.get("preferred_language"):
            bits.append(f"  <preferred_language>{pref}</preferred_language>")
        if bits:
            profile_block = "\n<customer_profile>\n" + "\n".join(bits) + "\n</customer_profile>"

    channel_text = channel or "未知渠道"

    return f"""<role>
你是一名外部客户服务助理，正在通过 {channel_text} 与客户对话。
</role>

<goal>
准确、礼貌、高效地解决客户的咨询；当问题超出能力或客户明确要求人工时，
主动调用 transfer_to_human 工具转接。
</goal>

<capabilities>
- 回答订单 / 物流 / 退换货 / 会员权益等常规咨询。
- 调用工具：calculator（数值计算）、search（外部检索）、
  recall_memory（按语义召回历史会话片段）、subagent（NL2SQL 查业务数据）、
  transfer_to_human（转人工，会让对话进入人工接管态）。
- 在工具结果支撑下给出结论；没有支撑时不要编造单号、价格、时间等具体事实。
</capabilities>

<style>
- 简短、口语化、不堆砌套话。
- 涉及具体业务时给出明确步骤而不是泛泛而谈。
- 不暴露内部实现（"调用工具"、"检索 RAG"等技术名词）。
- 默认中文；客户档案标注偏好语言时按其偏好。
</style>

<constraints>
- 严禁伪造任���具体数字、时间、单号、商品 SKU。
- 涉及退款 / 投诉 / 情绪激烈时优先转人工，不要自作主张承诺补偿。
- 工具结果与客户陈述冲突时以工具结果为准，并礼貌指出差异。
</constraints>{profile_block}"""


def _latest_human_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else ""
    return ""


def _maybe_skill_section(
    skill_registry: SkillRegistry | None,
    *,
    query: str,
    channel: str,
    top_k: int,
) -> str:
    """Return the rendered skills block (possibly empty) for prepending."""
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
    """Prepend a system prompt with profile / events / skills on first turn.

    On subsequent turns within the same thread (the checkpointer's
    ``thread_id``) the messages already include a leading SystemMessage
    and we leave them untouched.

    Sections rendered (in order):

    1. Base persona + channel context
    2. Long-term ``user_profile`` snippets
    3. Recent medium-term events (cross-session continuity)
    4. Matched skills / SOPs (if a registry is present and the latest
       human message hits intent keywords)
    """
    messages = state.get("messages", [])
    if any(isinstance(m, SystemMessage) for m in messages):
        return {}
    cfg = config.get("configurable", {}) if config else {}

    sections: list[str] = [_system_prompt(state.get("user_profile"), state.get("channel", ""))]

    events_block = render_recent_events_for_prompt(state.get("recent_events") or [])
    if events_block:
        sections.append(f"<recent_context>\n{events_block}\n</recent_context>")

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
        events=len(state.get("recent_events") or []),
        has_profile=bool(state.get("user_profile")),
        has_skills=bool(skill_section),
    )
    return {"messages": [SystemMessage(content=full_prompt)]}


async def agent_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
    *,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Invoke the main-tier LLM and append its reply to ``messages``.

    Phase-3 Group D: when ``state["force_handoff"]`` is set by the tool
    guard, skip the LLM and inject a ``transfer_to_human`` tool call
    directly so the interrupt fires on the next ToolNode pass.
    """
    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    if caller is None:
        raise RuntimeError("agent_node requires config['configurable']['llm_caller']")

    started = time.perf_counter()
    if state.get("force_handoff"):
        _log.warning("agents.agent_node.force_handoff")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "force-handoff-0",
                            "name": "transfer_to_human",
                            "args": {"reason": "工具安全机制触发强制转人工"},
                        }
                    ],
                )
            ]
        }

    bound_tools = tools if tools is not None else AGENT_TOOLS
    result = await caller.chat(
        "main_primary",
        list(state.get("messages", [])),
        tools=bound_tools,
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


async def exit_node(state: CustomerServiceState) -> dict[str, Any]:
    """Terminal log point. Phase-2 may emit accounting / metrics here."""
    _log.debug(
        "agents.exit_node",
        session_id=state.get("session_id"),
        message_count=len(state.get("messages", [])),
    )
    return {}
