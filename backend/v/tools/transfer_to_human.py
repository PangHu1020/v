"""``transfer_to_human`` tool: hand the conversation off to an operator.

Calling this tool inside ``agent_node``'s LangChain tool-call loop invokes
LangGraph's :func:`langgraph.types.interrupt`, which suspends the graph at
this exact tool call. The graph's caller (the bus worker) detects the
interrupt and runs the ``on_interrupt`` hook to:

1. Migrate the thread's checkpoint from Redis (hot) to Postgres (cold).
2. Flip the session status to ``suspended`` so subsequent customer
   messages bypass the graph and forward into the Slack alert thread.
3. Post a handoff alert to Slack with a "Resume AI" button.

When the operator clicks "Resume AI", :func:`langgraph.types.Command` is
invoked with ``resume={"type": "resume", "operator_messages": [...]}``
which becomes the return value of ``interrupt(...)`` here. The tool then
returns a short summary that the agent sees as a regular tool result and
the conversation continues.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool("transfer_to_human", parse_docstring=True)
def transfer_to_human(reason: str) -> str:
    """Transfer the conversation to a human operator.

    Use this when:
    - The customer explicitly asks for a human.
    - The question is outside your competence (legal, refunds beyond
      authority, complex disputes).
    - The customer is frustrated and AI replies aren't helping.

    Args:
        reason: A short Chinese sentence explaining why a human is needed.
            Will be shown to the operator in the Slack alert.

    Returns:
        Short summary of the operator's decision. The agent should
        incorporate this into its next reply and then proceed normally.
    """
    decision: Any = interrupt({"type": "transfer_to_human", "reason": reason})

    if isinstance(decision, dict):
        msgs = decision.get("operator_messages") or []
        if msgs:
            joined = "\n".join(f"- {m}" for m in msgs)
            return f"人工已接管并给出指引：\n{joined}\n请基于以上信息继续回应客户。"
        return "人工已确认让 AI 继续。"
    return f"人工已确认：{decision}"
