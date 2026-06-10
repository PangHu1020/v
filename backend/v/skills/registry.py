"""Skill matcher / registry.

Builds an in-memory index of loaded skills and ranks them against a
customer's most recent text. Phase-2 P4 uses keyword overlap; Phase-3
will add embedding-based semantic match (the embedder is already
available via :func:`backend.v.models.factory.get_embedding`).

Scoring (per skill, given query text + channel):

1. Filter: skip if ``skill.channels`` is set and ``channel`` isn't in it.
2. Match: count how many of the skill's ``intents`` appear (case-
   insensitive substring) in the query.
3. Tiebreak: skills with equal match counts sort by ``priority`` desc,
   then by ``name`` asc for determinism.

Skills with zero intent matches are excluded — we don't inject SOPs
the customer didn't ask about.
"""

from __future__ import annotations

from backend.v.skills.model import Skill
from backend.v.utils.logging import get_logger

_log = get_logger("skills.registry")


def _score(skill: Skill, query_lower: str) -> int:
    if not skill.intents:
        return 0
    return sum(1 for intent in skill.intents if intent.lower() in query_lower)


class SkillRegistry:
    """Holds all loaded skills + answers ``match`` queries."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = list(skills)

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def get(self, name: str, *, channel: str | None = None) -> Skill | None:
        """Return the skill with this exact ``name`` (channel-filtered), else None.

        Used by the ``load_skill`` tool for progressive disclosure: the cold-layer
        catalog advertises the name, the model calls the tool, this returns the body.
        """
        for skill in self._skills:
            if skill.name == name and (channel is None or skill.applies_to_channel(channel)):
                return skill
        return None

    def catalog(self, *, channel: str) -> list[Skill]:
        """All skills available on ``channel`` (for the frozen cold-layer catalog).

        Unlike :meth:`match`, this is query-independent — it lists the whole
        toolbox so the prompt prefix stays stable across turns.
        """
        return [s for s in self._skills if s.applies_to_channel(channel)]

    def render_catalog(self, *, channel: str) -> str:
        """Render the channel's skill catalog (schemas only) for the cold layer."""
        from backend.v.skills.prompts import render_skill_catalog

        return render_skill_catalog(self.catalog(channel=channel))

    def match(
        self,
        query: str,
        *,
        channel: str,
        top_k: int = 3,
    ) -> list[Skill]:
        """Return up to ``top_k`` skills whose intents match the query.

        Args:
            query: Latest customer message.
            channel: Customer channel slug; skills with ``channels=[...]``
                that exclude this channel are filtered out.
            top_k: Cap on returned skills. ``0`` returns ``[]``.
        """
        if top_k <= 0 or not query:
            return []
        q = query.lower()
        scored: list[tuple[int, int, str, Skill]] = []
        for skill in self._skills:
            if not skill.applies_to_channel(channel):
                continue
            score = _score(skill, q)
            if score == 0:
                continue
            # Sort key: higher score, higher priority, lower name (alpha).
            scored.append((-score, -skill.priority, skill.name, skill))
        scored.sort()
        result = [s for _, _, _, s in scored[:top_k]]
        if result:
            _log.debug(
                "skills.registry.matched",
                count=len(result),
                names=[s.name for s in result],
            )
        return result

    def render_for_prompt(self, skills: list[Skill]) -> str:
        """Render matched skills as XML for system-prompt injection.

        Delegates to :mod:`backend.v.skills.prompts` so the markup
        sits next to every other prompt in the project.
        """
        from backend.v.skills.prompts import render_sops_for_prompt

        return render_sops_for_prompt(skills)
