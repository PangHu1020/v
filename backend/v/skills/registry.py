"""Skill registry.

Builds an in-memory index of loaded skills and exposes them to the agent
via Anthropic-style progressive disclosure: the cold system layer carries a
frozen ``<available_skills>`` catalog (names + descriptions only, via
:meth:`SkillRegistry.render_catalog`), and the model pulls a full SOP body on
demand with the ``load_skill`` tool (which calls :meth:`SkillRegistry.get`).

Catalog membership is channel-scoped: a skill with ``channels=[...]`` is only
listed on those channels. The catalog is intentionally uncapped — it is the
frozen cold layer, so listing every available skill keeps the cached prompt
prefix stable across turns.
"""

from __future__ import annotations

from backend.v.skills.model import Skill


class SkillRegistry:
    """Holds all loaded skills + serves the catalog / body-on-demand."""

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
