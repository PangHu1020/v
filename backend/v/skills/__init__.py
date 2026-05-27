"""Skill loader (Phase-2 P4).

Skills are reusable SOPs the agent can prepend to its system prompt
when the customer's intent matches. Phase-2 P4 ships markdown-only,
filesystem-only, with keyword-based matching. Phase-3 will add
``SKILL.md`` + executable scripts (with sandboxing), community-registry
sync, and embedding-based semantic match.

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

The frontmatter ``intents`` are case-insensitive substrings the loader
matches against the customer's most recent message; the highest-scoring
``max_skills_per_turn`` skills are inlined into the system prompt by
``enter_node``.
"""

from backend.v.skills.loader import load_skills
from backend.v.skills.model import Skill
from backend.v.skills.registry import SkillRegistry

__all__ = ["Skill", "SkillRegistry", "load_skills"]
