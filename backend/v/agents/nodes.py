"""LangGraph nodes: enter -> agent (with tools) -> exit.

Phase-2 P0 binds the ``transfer_to_human`` tool so the agent can suspend
the graph via :func:`langgraph.types.interrupt` when handoff is needed.
The conditional edge in :mod:`backend.v.agents.graph` routes the agent's
tool calls through ``ToolNode`` and back; tool-less responses go to exit.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.state import CustomerServiceState
from backend.v.models.llm_caller import LLMCaller
from backend.v.tools import transfer_to_human
from backend.v.utils.logging import get_logger

_log = get_logger("agents.nodes")

AGENT_TOOLS: list = [transfer_to_human]
"""Tools bound to the main agent. Phase-2 P3 adds ``recall_memory``;
Phase-2 P2 adds ``subagent``."""


def _system_prompt(profile: dict[str, Any] | None, channel: str) -> str:
    """Render the per-turn system prompt.

    Phase-1 keeps this concise; Phase-2 will pull SOPs from the Skill loader
    and inject them here based on the current intent.
    """
    base = (
        "你是一名外部客户服务助理，正在通过 "
        f"{channel} 与客户对话。回复要简短、礼貌、准确；"
        "若涉及订单、物流、退货等具体业务，按 SOP 给出明确步骤。"
        "若问题超出你的能力或客户明确要求人工，请说明会转接人工。"
    )
    if not profile:
        return base
    bits = []
    if name := profile.get("customer_name"):
        bits.append(f"客户姓名：{name}")
    if level := profile.get("member_level"):
        bits.append(f"会员等级：{level}")
    if pref := profile.get("preferred_language"):
        bits.append(f"偏好语言：{pref}")
    if not bits:
        return base
    return base + "\n\n已知客户档案：\n" + "\n".join(f"- {b}" for b in bits)


async def enter_node(state: CustomerServiceState) -> dict[str, Any]:
    """Prepend a system prompt with the user_profile on the first turn.

    On subsequent turns within the same thread (the checkpointer's
    ``thread_id``) the messages already include a leading SystemMessage and
    we leave them untouched.
    """
    messages = state.get("messages", [])
    if any(isinstance(m, SystemMessage) for m in messages):
        return {}
    prompt = _system_prompt(state.get("user_profile"), state.get("channel", ""))
    return {"messages": [SystemMessage(content=prompt)]}


async def agent_node(
    state: CustomerServiceState,
    config: RunnableConfig,
    *,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Invoke the main-tier LLM and append its reply to ``messages``.

    The tool list is provided either explicitly (via :func:`functools.partial`
    in :func:`backend.v.agents.graph.build_graph`) or implicitly via
    :data:`AGENT_TOOLS`. The graph wires both the LLM binding and the
    downstream ``ToolNode`` from the same list so what the LLM sees and
    what gets dispatched are guaranteed identical.
    """
    cfg = config.get("configurable", {}) if config else {}
    caller: LLMCaller | None = cfg.get("llm_caller")
    if caller is None:
        raise RuntimeError("agent_node requires config['configurable']['llm_caller']")

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
