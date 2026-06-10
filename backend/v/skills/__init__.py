"""Skill loader.

Skills are reusable SOPs surfaced to the agent via Anthropic-style
progressive disclosure: a frozen ``<available_skills>`` catalog (names +
descriptions) sits in the cold system layer, and the model pulls a full body
on demand with the ``load_skill`` tool. Markdown-only, filesystem-only for
now; ``SKILL.md`` + executable scripts (with sandboxing), community-registry
sync, and embedding-based selection are deferred.

A skill file is a markdown document with YAML frontmatter::

    ---
    name: refund_sop
    description: 标准退款流程
    intents: [退款, 退货, 退钱, refund]
    priority: 5
    channels: [wecom, feishu]   # optional; omit for all channels
    ---

    # 退款流程

    1. 询问订单号 ...

``description`` is what the catalog advertises; ``channels`` scopes which
channels list the skill. ``intents`` is retained on the model for future
ranking but no longer gates catalog membership — the catalog is uncapped so
the cold-layer prefix stays cache-stable.
"""

from backend.v.skills.loader import load_skills
from backend.v.skills.model import Skill
from backend.v.skills.registry import SkillRegistry

__all__ = ["Skill", "SkillRegistry", "load_skills"]
