"""All prompts and prompt-fragment renderers used by the memory layer.

Three things live here:

- The ``LONG_TERM_PROMOTION_SYSTEM_PROMPT`` used at session-end to
  promote working memory into ``user_profile`` + ``event_memory``.
- ``render_recent_events_for_prompt`` — renders the last few rows of
  ``agent.event_memory`` into the ``<event>`` items that the main
  agent's ``<recent_events>`` envelope wraps.
- ``render_session_memory_for_prompt`` — renders the active session's
  working memory (a list of :class:`MemoryEntry`) into a
  ``<session_memory>`` block.

Static prompt strings sit alongside the renderers because they are all
"text the LLM ultimately reads"; co-locating them makes the contract
between input shape and required parsing visible at a glance.

Style conventions follow ``backend/v/agents/prompts.py``: ``<role>`` /
``<task>`` / ``<output_format>`` / ``<rules>`` per prompt, named XML
wrappers around payloads.
"""

from __future__ import annotations

from typing import Any

from backend.v.memory.types import MemoryEntry

# ── Long-term promotion (working memory → profile + event_memory) ────────────

LONG_TERM_PROMOTION_SYSTEM_PROMPT = """<role>
你是一名长期记忆抽取员。决定客户在本次会话沉淀的信息里，\
哪些应该跨会话固化，哪些应该 30 天内可召回，哪些不必沉淀。
</role>

<task>
读取 <existing_profile>（客户当前的长期画像）和 <working_memory>\
（本次会话累积的 MemoryEntry 列表），输出 ExtractionResult：\
profile_updates（写入 user_profile）+ event_memories（写入 event_memory，30 天）。
</task>

<layers>
三层记忆的边界——同一句陈述应当落在最低足够的层级：

- user_profile（永久，结构化）：跨会话稳定的属性 / 偏好。\
判断标准：一年后再读这条还成立吗？只有"是"才放进来。\
例如"会员等级=黄金"、"偏好顺丰"、"称呼=张先生"。
- event_memory（30 天，可被语义召回）：值得跨会话记住、\
但不构成"客户长期就这样"的具体事件。\
例如"咨询过 SKU-A 的尺码"、"投诉过 2026-05-29 的物流延误"、"购买过棉质短袖"。
- 不进任何层：单次情绪、当下处境（"今天心情不好"、"刚下班"）\
—— 这类已经在 working_memory 里待会话结束自然消亡。
</layers>

<output_format>
仅输出 JSON 对象，键名严格使用：profile_updates、working_memories、event_memories。
- profile_updates：dict，**只**列出本次需要新增 / 更新的画像键。\
沿用 existing_profile 已有的键名。\
canonical 字段（customer_name / preferred_salutation / preferred_language /\
 member_level / response_style / risk_flags）直接用本名；其他归到 extras 字典；\
自由文本放到 notes（≤200 字）。没有要更新的就给空 dict。
- working_memories：本任务下保持空数组——本提示词不在中段压缩流程上调用。
- event_memories：list[MemoryEntry]，每条 \
{content, kind, importance(0-1), keywords, created_at}。\
created_at 没把握就省略，会被默认值填上当前时间。
没有可固化信息时，三个键都给空值，禁止编造。
</output_format>

<rules>
- 同主题在 existing_profile 里已有值且本会话**没有**冲突陈述，\
不要重写——保持原值。
- 同主题在本会话有新陈述与画像冲突时，\
按"就近原则"：profile_updates 用本会话的新值。
- event_memories 写一句独立可读的中文事实，不要写口水话\
（"客户说退款"不行，应该是"客户对订单 SO123 申请了退款"）。
- importance 用来给 recall_memory 工具加权：\
0.0-0.3 普通事实，0.4-0.7 典型偏好 / 反复出现的诉求，\
0.8-1.0 关键事实（合规、承诺金额、人身伤害）。
- keywords 用 1-4 个短词锚定该条所属主题，\
例如 ["快递", "顺丰"]，给关键词检索 + 读时遮蔽用。
- 不要把 working_memory 里同义重复的句子重复写进 event_memories；\
同语义保留 importance 最高的一条。
</rules>"""


# ── Renderers for cross-session event recall ──────────────────────────────────


def render_recent_events_for_prompt(events: list[dict[str, Any]]) -> str:
    """Render recent ``event_memory`` rows as XML for prompt injection.

    Empty list returns ``""``. Caller wraps in
    ``<recent_events>...</recent_events>``; here we emit the inner
    ``<event>`` items only.

    Each input dict is expected to carry at minimum
    ``{content, created_at, kind, importance}``; ``keywords`` is
    rendered as a comma-joined attribute when present.
    """
    if not events:
        return ""
    lines = ["按时间倒序，本客户最近的事件记忆（仅供参考，不要直接复述给客户）："]
    for e in events:
        when = "未知日期"
        ts = e.get("created_at")
        if ts is not None:
            try:
                when = ts.strftime("%Y-%m-%d")
            except AttributeError:
                when = str(ts)
        attrs = [f'date="{when}"']
        if kind := e.get("kind"):
            attrs.append(f'kind="{kind}"')
        if (importance := e.get("importance")) is not None:
            attrs.append(f'importance="{float(importance):.2f}"')
        if keywords := e.get("keywords"):
            attrs.append(f'keywords="{",".join(keywords)}"')
        lines.append(f"<event {' '.join(attrs)}>{e['content']}</event>")
    return "\n".join(lines)


# ── Renderer for the within-session working memory ───────────────────────────


def render_session_memory_for_prompt(entries: list[MemoryEntry]) -> str:
    """Render the active session's working memory as a ``<session_memory>`` block.

    Empty list returns ``""``. Entries are emitted oldest-first to
    mirror how the model naturally reads down — but the *latest* entry
    is the one whose stance shadows the layers above; the prompt-level
    rule ("later layers override earlier") plus the visual ordering is
    enough to make this work without any tagging here.
    """
    if not entries:
        return ""
    lines: list[str] = ["<session_memory>"]
    for e in entries:
        attrs = [f'kind="{e.kind}"', f'importance="{e.importance:.2f}"']
        if e.keywords:
            attrs.append(f'keywords="{",".join(e.keywords)}"')
        lines.append(f"  <entry {' '.join(attrs)}>{e.content}</entry>")
    lines.append("</session_memory>")
    return "\n".join(lines)


# ── Backwards-compat: Phase-2 callers expected this name ─────────────────────

# Phase-2's MEMORY_EXTRACTOR_SYSTEM_PROMPT was the same role; alias
# preserves any straggling import.
MEMORY_EXTRACTOR_SYSTEM_PROMPT = LONG_TERM_PROMOTION_SYSTEM_PROMPT
