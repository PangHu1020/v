"""Prompts and rendering helpers for the skills layer.

Skills use Anthropic-style progressive disclosure: the cold system layer
carries a frozen ``<available_skills>`` *catalog* (names + descriptions only),
and the model pulls a full SOP body on demand via the ``load_skill`` tool.
This module owns the catalog intro line and the per-skill XML rendering.

See ``backend/v/agents/prompts.py`` for the project's prompt style
conventions; this file follows the same pattern.
"""

from __future__ import annotations

from backend.v.skills.model import Skill

CATALOG_INTRO_LINE = (
    "以下是可用的 SOP 技能清单（仅名称与适用场景）。"
    "当本次对话需要某个 SOP 的完整指引时，调用 load_skill(name) 工具取回正文，再据其执行。"
)


def render_skill_catalog(skills: list[Skill]) -> str:
    """Render the frozen skill *catalog* (schemas only) for the cold layer.

    Anthropic-style progressive disclosure: the catalog lists each skill's
    name + description (NOT the body). The model decides which to pull via the
    ``load_skill`` tool; the loaded body then lands in the conversation
    (append-only, hot layer). Keeping only schemas here means this block is
    cheap and frozen for the whole session — cache-friendly.

    Empty list returns ``""``. Caller wraps in ``<available_skills>...</>``.
    """
    if not skills:
        return ""
    lines: list[str] = [CATALOG_INTRO_LINE]
    for s in skills:
        desc = s.description or "（无描述）"
        lines.append(f'<skill name="{s.name}" priority="{s.priority}">{desc}</skill>')
    return "\n".join(lines)
