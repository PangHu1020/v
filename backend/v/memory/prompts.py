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

from backend.v.memory.types import ConversationState, MemoryEntry

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


# ── Session-end V2 extraction (working memory + transcript → MemoryExtraction) ─

SESSION_END_EXTRACTION_SYSTEM_PROMPT = """<role>
你是一名长期记忆抽取员。会话结束时，从本次会话沉淀里区分两类记忆：\
关于客户"是什么样的人"（用户记忆，可覆盖），和"发生过什么事"（情节记忆，只追加）。
</role>

<task>
读取 <existing_user_memory>（客户当前已知的属性）和 <session_input>\
（本次会话的工作记忆 / 对话），输出 MemoryExtraction：\
user_candidates（写入用户记忆）+ episodic_candidates（写入情节记忆）。
</task>

<boundaries>
两类记忆的边界——这是最关键的判断：

- 用户记忆 user_candidates（属性 / 当前状态，会被新值覆盖）：\
判断标准是"这描述客户长期是什么样，而非某次发生的事"。三种 kind：
  - preference（软偏好）：偏好顺丰、喜欢简洁回复、常买棉质。
  - constraint（硬约束，必须遵守）：对花生过敏、不要电话联系、只收工作日件。
  - pattern（反复行为模式）：每月初下单、经常退货、习惯先问价。
  每条必须给 attr_key（如 preferred_courier / allergy / order_pattern）\
+ attr_value（值）+ kind。同一属性后续会话给新值会覆盖旧值。

- 情节记忆 episodic_candidates（发生过的事，带时间，不冲突只追加）：\
具体事件，如"投诉了 2026-06-09 的物流延误"、"咨询��� SKU-A 的尺码"、\
"对订单 SO123 申请了退款"。每条尽量给 subject（订单号 / SKU / 话题锚点），\
用于后续按主题归并。

- 都不进：单次情绪、当下处境（"今天心情不好"）——会话结束自然消亡，不要抽取。
</boundaries>

<output_format>
仅输出 JSON，键名：user_candidates、episodic_candidates。
- user_candidates：list，每条 {content, importance(0-1), source, confidence(0-1),\
 attr_key, attr_value, kind}。
- episodic_candidates：list，每条 {content, importance(0-1), source, confidence(0-1),\
 keywords, subject}。
没有可抽取的就给空数组，禁止编造。
</output_format>

<rules>
- source：客户明说的填 "stated"，你从上下文推断的填 "inferred"。\
confidence 反映你对该判断的把握（0-1）。这两个字段决定冲突时谁覆盖谁，务必如实。
- attr_value 用最简洁的规范值（"顺丰" 而非 "客户说他喜欢用顺丰快递寄东西"）。\
content 才是可读句子。
- 同一属性在 existing_user_memory 已有且本次无变化：不要重复输出。\
有变化（客户改口）才输出新的 user_candidate。
- 别把同一信息既塞 user 又塞 episodic：属性型进 user，事件型进 episodic。
- importance：0.0-0.3 普通，0.4-0.7 典型偏好 / 诉求，0.8-1.0 关键（合规、承诺、过敏）。
</rules>"""


# ── Mid-session extraction (compression) ─────────────────────────────────────

MID_SESSION_EXTRACTION_SYSTEM_PROMPT = """<role>
你是一名会话记忆整理员。在对话中段（上下文压缩时），既要提炼可沉淀的记忆，\
又要产出一份"对话状态快照"，让压缩掉旧消息后对话仍能无缝衔接。
</role>

<task>
读取 <conversation>（即将被截断的历史对话），输出 ExtractionResult：
- conversation_state：本次对话的结构化状态快照（用于压缩后无缝衔接，最关键）。
- working_memories：本次会话内仍有用的短期记忆，下轮注入 system prompt。
- event_memories：值得 30 天内跨会话召回的具体事实。
- profile_updates：保持空 dict——会话未结束，不修改长期画像。
</task>

<output_format>
仅输出 JSON，键名：conversation_state、working_memories、event_memories、\
profile_updates（空 dict）。
conversation_state 字段：
- current_topic：一句话说明"现在在聊什么"。
- events：按顺序列出发生过的关键事件（客户问了X、确认了Y、AI查了Z）。
- actions_taken：AI 已经执行的动作（调用过的工具、给出的信息、做出的承诺）。
- unresolved_questions：尚未解决的问题 / 待办。
- key_facts：已确认的具体事实（订单号、金额、SKU、日期、运单号等）。
每条 MemoryEntry：{content, kind, importance(0-1), keywords}。
</output_format>

<rules>
- conversation_state 要让接手者读完就能继续对话，不丢上下文；用简短中文，不要客套。
- key_facts 必须是对话里真实出现的具体值，禁止编造。
- working_memories：客户本次表达的偏好、待处理诉求、临时背景信息。importance >= 0.4。
- event_memories：具体可引用的事实（订单号、投诉内容、特殊需求）。importance >= 0.3。
- 不要重复两个记忆列表里的相同信息；偏好类放 working，事件类放 event。
- 没有可提取的记忆时给空数组；但 conversation_state 应尽量填充，除非对话确实无实质内容。
</rules>"""


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


# ── Renderer for the structured conversation-state summary ───────────────────


def render_conversation_state(state: ConversationState) -> str:
    """Render a :class:`ConversationState` as a ``<conversation_state>`` block.

    Injected by the compression node in place of the dropped turns so the agent
    resumes seamlessly. Empty instance returns ``""``.
    """
    if state.is_empty():
        return ""
    lines: list[str] = ["<conversation_state>"]
    if state.current_topic:
        lines.append(f"  <current_topic>{state.current_topic}</current_topic>")

    def _list_block(tag: str, items: list[str]) -> None:
        if items:
            lines.append(f"  <{tag}>")
            lines.extend(f"    <item>{it}</item>" for it in items)
            lines.append(f"  </{tag}>")

    _list_block("events", state.events)
    _list_block("actions_taken", state.actions_taken)
    _list_block("unresolved_questions", state.unresolved_questions)
    _list_block("key_facts", state.key_facts)
    lines.append("</conversation_state>")
    return "\n".join(lines)


# ── Backwards-compat: Phase-2 callers expected this name ─────────────────────

# Phase-2's MEMORY_EXTRACTOR_SYSTEM_PROMPT was the same role; alias
# preserves any straggling import.
MEMORY_EXTRACTOR_SYSTEM_PROMPT = LONG_TERM_PROMOTION_SYSTEM_PROMPT


# ── Monthly consolidation (raw episodic cluster → one summary) ────────────────

CONSOLIDATION_SUMMARY_SYSTEM_PROMPT = """<role>
你是一名记忆归并员。把同一主题、同一月份的若干条情节记忆，\
压缩成一条信息无损的月度摘要，供长期低成本召回。
</role>

<task>
读取 <episodes>（同一 subject、同一月份的原始情节记忆列表），\
输出一条 JSON：{"summary": "...", "importance": 0-1}。
summary 用一句到三句中文概括这批事件，保留关键具体值\
（订单号、金额、SKU、日期、投诉/诉求要点），丢弃口水与重复。
importance 取这批里最高的那条（关键事件不能在归并后被降权）。
</task>

<rules>
- 不要编造 episodes 里没有的事实；具体值必须来自原文。
- 若这批事件指向同一件事的多次往复，concise 成一条主线，\
但保留"反复出现"这一信号（如"就 SO123 多次催促物流"）。
- 仅输出 JSON，键名：summary、importance。
</rules>"""
