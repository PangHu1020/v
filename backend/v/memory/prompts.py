"""All prompts and prompt-fragment renderers used by the memory layer.

Three things live here:

- The ``MEMORY_EXTRACTOR_SYSTEM_PROMPT`` used when promoting a session's
  short-term memory into the long-term ``user_profile`` + episodes.
- ``render_recent_events_for_prompt`` — turns the last few medium-term
  ``agent.session_memory`` rows into the ``<event>`` items that the main
  agent's ``<recent_context>`` envelope wraps.
- ``render_session_memory_for_prompt`` — turns a single Redis-side
  session-memory record into a self-contained ``<session_memory>`` XML
  block for injection into the system prompt or the compression marker.

Static prompt strings sit alongside the renderers because they are all
"text the LLM ultimately reads"; co-locating them makes the contract
between input shape and required parsing visible at a glance.
"""

from __future__ import annotations

from typing import Any

# ── Long-term memory promotion (session memory → user_profile + episodes) ────

MEMORY_EXTRACTOR_SYSTEM_PROMPT = """<role>
你是一名长期记忆抽取员，决定哪些信息应该跨会话固化，哪些只该停留在当前会话。
</role>

<task>
读取 <existing_profile>（客户当前的长期画像）和 <session_memory>\
（本次会话的短期偏好 / 观察），输出更新后的长期画像和值得长期召回的事件列表。
</task>

<output_format>
仅输出 JSON 对象，键名严格使用：profile、episodes。
- profile：合并后的长期画像 JSON 对象。\
保留 existing_profile 的字段，仅在新信息明确支持时增改对应键。
- episodes：列表，每项是一句独立的中文事实，例如"客户偏好夜间���货"、"曾投诉物流延误"。\
供后续语义召回。

如果没有任何可固化信息：profile 原样返回 existing_profile，episodes 为空数组。
</output_format>

<rules>
- 跨会话标准（必须同时满足才能写进 profile）：
  1) 描述的是该客户的稳定属性 / 长期偏好（语言、称呼、收货习惯、忠诚等级、产品偏好）。
  2) 不依赖某次会话的临时上下文（"今天心情不好"、"今晚要发货"不算）。
  3) 与既有 profile 不冲突，或新信息更新 / 替代旧字段时有明确依据。
- preferences 中能反映"客户长期就这样"的项可升级到 profile；\
observations 几乎都是当下情境，原则上不升��。
- episodes 是事件而不是属性："客户偏好顺丰" → 写进 profile.preferred_courier；\
"客户咨询过 SKU-123 的尺码" → 写进 episodes。
- 不要编造 existing_profile 中没有、session_memory 中也未出现的信息。
- profile 字段名优先沿用 existing_profile 已有的命名，不要新造同义键。
</rules>"""


# ── Renderers for cross-session event recall ──────────────────────────────────


def render_recent_events_for_prompt(events: list[dict[str, Any]]) -> str:
    """Render recent cross-session events as XML for system-prompt injection.

    Empty list returns ``""`` so the caller can drop the section cleanly.
    The caller wraps the returned text in ``<recent_context>...</recent_context>``;
    here we emit the inner ``<event>`` items only.
    """
    if not events:
        return ""
    lines = ["以下是该客户最近几次会话的要点（按时间倒序，仅供参考，不要直接复述）："]
    for e in events:
        when = e["created_at"].strftime("%Y-%m-%d") if e.get("created_at") else "未知日期"
        lines.append(f'<event date="{when}">{e["summary"]}</event>')
    return "\n".join(lines)


# ── Renderer for the within-session record ────────────────────────────────────


def render_session_memory_for_prompt(record: dict[str, Any] | None) -> str:
    """Render a session-memory record as a self-contained ``<session_memory>`` block.

    Returns ``""`` for empty records so the caller can drop the section.
    Emitted shape::

        <session_memory>
          <preferences>
            <pref key="...">...</pref>
          </preferences>
          <observations>
            <observation>...</observation>
          </observations>
        </session_memory>
    """
    if not record:
        return ""
    prefs = record.get("preferences") or {}
    obs = record.get("observations") or []
    if not prefs and not obs:
        return ""
    lines: list[str] = ["<session_memory>"]
    if prefs:
        lines.append("  <preferences>")
        for k, v in prefs.items():
            lines.append(f'    <pref key="{k}">{v}</pref>')
        lines.append("  </preferences>")
    if obs:
        lines.append("  <observations>")
        for item in obs:
            lines.append(f"    <observation>{item}</observation>")
        lines.append("  </observations>")
    lines.append("</session_memory>")
    return "\n".join(lines)
