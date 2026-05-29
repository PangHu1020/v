"""Prompts and rendering helpers for the skills layer.

Skills aren't rendered as a system prompt themselves — they're
fragments injected into the main agent's system prompt under the
``<sops>`` envelope. This module owns the static intro line and the
per-skill XML rendering so ``SkillRegistry.render_for_prompt`` stays
focused on data flow rather than markup.

See ``backend/v/agents/prompts.py`` for the project's prompt style
conventions; this file follows the same pattern.
"""

from __future__ import annotations

from backend.v.skills.model import Skill

SOPS_INTRO_LINE = "以下 SOP 与本次咨询相关，按优先级从高到低排列；遵循其指引而不是逐字复述："


def render_sops_for_prompt(skills: list[Skill]) -> str:
    """Render matched skills as XML for system-prompt injection.

    Empty list returns ``""``. The caller wraps in ``<sops>...</sops>``;
    here we emit only the intro line and the inner ``<sop>`` items.
    """
    if not skills:
        return ""
    lines: list[str] = [SOPS_INTRO_LINE]
    for s in skills:
        attrs = f' name="{s.name}" priority="{s.priority}"'
        if s.description:
            attrs += f' description="{s.description}"'
        lines.append(f"<sop{attrs}>")
        lines.append(s.body.strip())
        lines.append("</sop>")
    return "\n".join(lines)
