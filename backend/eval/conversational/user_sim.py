"""User simulator for conversational retrieval eval (stage 3).

An LLM plays the customer: given a case's ``persona`` + ``hidden_need`` + ``noise``
and the dialogue so far, it generates the next customer utterance. The prompt
enforces colloquial, emotional, info-dripping behavior (reveal constraints across
multiple turns, not all at once) and weaves in the noise. The simulator also
decides when the customer is satisfied and ends the dialogue.

The simulator is *stateless*: it re-reads the full transcript each turn and emits
one customer message. The harness orchestrates the turn loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.conversational.cases import ConvCase
from backend.v.utils.logging import get_logger

_log = get_logger("eval.user_sim")

_PERSONA_STYLE = {
    "budget_tight": "反复强调预算，追问价格，对比性价比，语气紧张。",
    "in_a_hurry": "回答简短生硬，没耐心，追问'快点'，能不说就不说。",
    "clueless": "用大白话问，提一堆生活场景而非参数，模糊不清，需客服引导。",
    "frustrated": "带怨气，吐槽上次体验差，强调'这次不能再坑我'，语气冲。",
    "gifting": "给别人买，自己不太懂，需求飘，可能顺嘴问有没有活动/优惠。",
    "terse": "惜字如金，一两个字一句（'嗯''就这个''多少钱'），全程被动。",
    "rambler": "先扯一堆废话（天气/抱怨/闲聊），正经需求埋在废话里，啰嗦。",
    "wishy_washy": "边聊边改主意，预算或偏好前后变化，让客服跟着调整。",
}

_SYS = (
    "你在模拟一个真实客户与客服多轮对话。给定客户的【真实诉求 hidden_need】、"
    "【画像 persona】、【噪声话题 noise】和当前对话历史，生成客户的下一句话。\n"
    "\n"
    "规则（严格遵守）：\n"
    "1. 信息碎片化：不要一次性说清全部需求（类目+预算+偏好）。客户第一句可能只说个大概"
    "（'想买手机'），等客服问了再一点点补充预算、用途等。像真人那样慢慢漏信息。\n"
    "2. 画像一致：你的语气、啰嗦度、耐心必须符合 persona 风格（见下文）。\n"
    "3. 噪声自然夹带：如果 noise 非空，要在对话过程中自然地插入（比如 rambler 先扯天气、"
    "frustrated 吐槽上次被坑），但不要所有 noise 一口气说完——分几轮。\n"
    "4. 绝不点名具体商品：你可以说'两三千的手机''跑鞋''空气炸锅那种'，但【绝对不能】"
    "说出任何具体型号品牌（不能出现'iPhone 15''小米 Civi 3''索尼 WH-1000XM5'）。\n"
    "5. 配合客服追问：如果客服问预算/用途/偏好，你要回答（但仍然碎片化，别一下全说完）。\n"
    "6. 判断结束：如果客服已经推荐了合适商品、你觉得满意或者不想再聊了，"
    "输出 done=true；否则 done=false 继续对话。\n"
    "\n"
    "输出 JSON：\n"
    "  utterance: 客户下一句话（口语化、符合画像）。\n"
    "  done: 布尔值，true=对话结束（满意或放弃），false=继续。"
)


class _SimOutput(BaseModel):
    """User simulator's next turn."""

    utterance: str = Field(description="客户下一句话，口语化、符合画像、信息碎片化")
    done: bool = Field(description="对话是否结束（客户满意或不想再聊）")


async def simulate_user(
    llm,
    *,
    case: ConvCase,
    transcript: list[dict[str, str]],
) -> tuple[str, bool]:
    """Generate the customer's next utterance and done flag.

    Args:
        llm: LLMCaller instance.
        case: The case being evaluated (carries persona + hidden_need + noise).
        transcript: Dialogue so far, as [{role, content}, ...]. role ∈ {user, agent}.

    Returns:
        (utterance, done): The next customer message and whether the dialogue ends.
    """
    # Build the prompt: system + case spec + transcript.
    persona_guide = _PERSONA_STYLE[case.persona]
    user_prompt = (
        f"<case>\n"
        f"画像: {case.persona} — {persona_guide}\n"
        f"真实诉求: {case.hidden_need}\n"
        f"噪声话题: {case.noise or '（无）'}\n"
        f"</case>\n\n"
        f"<对话历史>\n"
    )
    if not transcript:
        user_prompt += "（还没开始，你要说第一句话）\n"
    else:
        for t in transcript:
            role_label = "客户" if t["role"] == "user" else "客服"
            user_prompt += f"{role_label}: {t['content']}\n"
    user_prompt += "</对话历史>\n\n生成客户下一句话："

    messages = [SystemMessage(content=_SYS), HumanMessage(content=user_prompt)]
    try:
        res = await llm.chat("main_primary", messages, structured=_SimOutput)
        parsed: _SimOutput | None = res.parsed if isinstance(res.parsed, _SimOutput) else None
    except Exception as exc:
        # Structured-output validation can fail when the model returns JSON
        # that doesn't fit _SimOutput (missing field, bad shape). Don't let
        # one bad simulator turn kill the whole case — degrade to a generic
        # continuation so the dialogue (and the retrieval being measured)
        # proceeds. (Genuine API errors like arrears still propagate from the
        # agent side, where they should fail the case.)
        _log.warning("user_sim.parse_failed", error=type(exc).__name__)
        return "嗯，你看着推荐吧", False
    if not parsed or not parsed.utterance.strip():
        # Fallback: the simulator failed, just say something generic.
        return "嗯...", False
    return parsed.utterance.strip(), parsed.done
