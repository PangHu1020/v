"""Generic subagent tool.

Lets the main agent delegate one focused sub-task to a fresh LLM call.
Two context modes:

- ``independent`` (default): the sub-agent sees only ``task``. Best for
  tasks where the main conversation would be noise — research,
  classification, transformations of caller-supplied input.
- ``shared``: the sub-agent additionally sees the last few messages from
  the main agent's conversation. Best when continuity matters
  (summarize-what-we've-discussed, "based on the customer's earlier
  statement, ...").

Implemented as a single LLM round-trip with no nested tool calls. That
keeps the surface narrow and the cost predictable; if a task genuinely
needs a tool-using sub-loop, the main agent should call the necessary
tools itself rather than delegate.

Why this is one tool with a flag instead of two separate primitives:
the same problem (focused work that benefits from a fresh frame) can
need either mode depending on the task. Forcing the agent to pick
between two named tools makes the choice harder and the prompt
descriptions duplicate. One tool + one parameter mirrors how a human
delegates: "please summarize the call so far" vs "please research X
fresh, no need to read the chat".
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from backend.v.tools.prompts import SUBAGENT_SYSTEM_PROMPT
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("tools.subagent")

DEFAULT_SHARED_TURN_COUNT = 10
"""When ``context_mode='shared'``, how many of the parent's most recent
messages to forward. Bounds the token budget so a long conversation
doesn't make the subagent prompt explode."""

_SUBAGENT_SYSTEM_PROMPT = SUBAGENT_SYSTEM_PROMPT


def _slice_parent_messages(state: dict[str, Any], limit: int) -> list[BaseMessage]:
    """Return the last ``limit`` non-system messages from the parent state."""
    messages = state.get("messages", []) if isinstance(state, dict) else []
    # Skip the system prompt (always first); take the tail of the rest.
    body = [m for m in messages if not isinstance(m, SystemMessage)]
    return body[-limit:] if limit > 0 else []


@tool("subagent")
async def subagent(
    task: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    context_mode: Literal["independent", "shared"] = "independent",
) -> str:
    """Delegate a focused sub-task to a fresh LLM call.

    Use this when:
    - You need to think hard about one specific question without the main
      conversation cluttering the prompt (``context_mode='independent'``).
    - You need a structured output for an internal step (summary,
      classification, extraction) that the customer shouldn't see verbatim.
    - You need to summarize / refer back to recent dialogue
      (``context_mode='shared'``).

    Don't use this to:
    - Call other tools (the subagent has no tools of its own).
    - Run long multi-step research (it's a single LLM call).

    Args:
        task: A self-contained Chinese description of what you want done.
            Be specific about input, expected output, and constraints.
        context_mode: ``"independent"`` (default) or ``"shared"``. Shared
            includes the last few parent messages.
    """
    cfg = config.get("configurable", {}) if config else {}
    llm_caller = cfg.get("llm_caller")
    if llm_caller is None:
        return "[subagent_error] missing llm_caller in config"

    messages: list[BaseMessage] = [SystemMessage(content=_SUBAGENT_SYSTEM_PROMPT)]
    if context_mode == "shared":
        messages.extend(_slice_parent_messages(state, DEFAULT_SHARED_TURN_COUNT))
    messages.append(HumanMessage(content=f"<task>\n{task}\n</task>"))

    with bind_request(subagent_mode=context_mode):
        try:
            result = await llm_caller.chat("main_fallback", messages)
        except Exception as exc:
            _log.error("tools.subagent.llm_failed", error=type(exc).__name__)
            return f"[subagent_error] llm failed: {type(exc).__name__}"

        content = result.message.content
        text = content if isinstance(content, str) else str(content)
        _log.info(
            "tools.subagent.completed",
            mode=context_mode,
            task_len=len(task),
            output_len=len(text),
            model=result.model,
            fallback_used=result.fallback_used,
        )
        return text
