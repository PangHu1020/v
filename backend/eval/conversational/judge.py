"""Task-completion judge for conversational eval.

After a dialogue runs, an LLM judges whether the customer's underlying need
(``ConvCase.hidden_need``) was actually satisfied by the agent — a layer above
retrieval hit. "Did the agent retrieve a gold product" (hit) and "did the agent
resolve the customer's request" (completion) diverge often: the agent can
surface the right product yet answer the wrong question, or never commit to a
recommendation. This judge captures the second.

LLM-as-judge: subjective and costs one extra LLM call per case, but it's the
only way to score "答非所问" / unresolved dialogues that rule-based hit can't see.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.conversational.cases import ConvCase
from backend.v.utils.logging import get_logger

_log = get_logger("eval.judge")

_SYS = (
    "你是一名客服对话质检员。给定客户的【真实诉求】和一段【客服对话】，"
    "判断对话结束时客户的诉求是否被实质满足。\n"
    "满足的标准：客服理解了诉求，并给出了对应的、可用的答复（如推荐了符合条件的商品、"
    "给出了正确的政策/操作）。\n"
    "不满足：答非所问、绕圈子没给结论、信息错误、或对话结束时诉求仍悬而未决。\n"
    "只看是否解决诉求，不评价语气。只输出 JSON：completed(bool)、reason(一句话中文)。"
)


class _Verdict(BaseModel):
    completed: bool = Field(description="客户诉求是否被实质满足")
    reason: str = Field(description="一句话理由")


async def judge_task_completion(
    llm, *, case: ConvCase, transcript: list[dict[str, str]]
) -> tuple[bool, str]:
    """Return (completed, reason). Degrades to (False, "<error>") on failure.

    Args:
        llm: LLMCaller instance.
        case: The case (carries ``hidden_need``).
        transcript: Dialogue as [{role, content}, ...]; role ∈ {user, agent}.
    """
    if not transcript:
        return False, "空对话"
    lines = []
    for t in transcript:
        label = "客户" if t["role"] == "user" else "客服"
        lines.append(f"{label}: {t['content']}")
    prompt = [
        SystemMessage(content=_SYS),
        HumanMessage(
            content=(
                f"<真实诉求>\n{case.hidden_need}\n</真实诉求>\n\n"
                f"<对话>\n" + "\n".join(lines) + "\n</对话>"
            )
        ),
    ]
    try:
        res = await llm.chat("summary", prompt, structured=_Verdict)
        v = res.parsed if isinstance(res.parsed, _Verdict) else None
    except Exception as exc:
        _log.warning("eval.judge.failed", case_id=case.case_id, error=type(exc).__name__)
        return False, f"<judge_error:{type(exc).__name__}>"
    if v is None:
        return False, "<judge_parse_failed>"
    return v.completed, v.reason.strip()
