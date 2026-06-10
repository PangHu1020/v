"""``load_skill``: progressive-disclosure tool for SOP skills.

The system prompt's cold layer advertises a frozen ``<available_skills>``
catalog (names + descriptions only). When the model decides a SOP is relevant
to the current turn, it calls ``load_skill(name)`` and the full SOP body is
returned as a ToolMessage — appended to the conversation (hot, append-only)
rather than baked into the system prompt. This keeps the cached system prefix
stable while still giving the model the full instructions on demand.

The :class:`~backend.v.skills.registry.SkillRegistry` is provided via
``config["configurable"]["skill_registry"]``; the channel scopes which skills
are visible.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from backend.v.utils.logging import get_logger

_log = get_logger("tools.load_skill")


@tool("load_skill", parse_docstring=True)
async def load_skill(name: str, config: RunnableConfig) -> str:
    """Load the full SOP body for a skill listed in <available_skills>.

    Call this when the current conversation needs the detailed steps of a
    specific SOP. Pass the exact skill ``name`` from the catalog. The returned
    text is the authoritative procedure — follow it; do not paraphrase it back
    to the customer verbatim.

    Args:
        name: The skill's exact name as shown in the <available_skills> catalog.
    """
    cfg = config.get("configurable", {}) if config else {}
    registry = cfg.get("skill_registry")
    channel = cfg.get("channel")
    if registry is None:
        return "（无法加载 SOP：缺少运行上下文）"
    skill = registry.get(name, channel=channel)
    if skill is None:
        return f"（未找到名为 {name!r} 的 SOP）"
    _log.info("tools.load_skill.loaded", name=name)
    return f'<sop name="{skill.name}">\n{skill.body.strip()}\n</sop>'
