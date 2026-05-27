"""Skill data model.

A :class:`Skill` is the parsed result of loading one markdown file. The
frontmatter fields control matching + ranking; the body is the SOP text
that gets inlined into the system prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SkillKind = Literal["markdown"]
"""Phase-2 P4 ships only the markdown SOP kind. ``executable`` is
reserved for the Phase-3 SKILL.md + exec format."""


class Skill(BaseModel):
    """One loaded skill."""

    model_config = ConfigDict(extra="forbid")

    kind: SkillKind = "markdown"
    name: str = Field(min_length=1, description="Stable identifier (file stem).")
    description: str = ""
    intents: list[str] = Field(
        default_factory=list,
        description=(
            "Case-insensitive substrings the matcher looks for in the customer's "
            "most recent message. Order doesn't matter; more matches = higher score."
        ),
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Tiebreaker when intent-match counts are equal. 10 = critical "
            "compliance / safety; 3 = normal SOP; 1 = nice-to-have hint."
        ),
    )
    channels: list[str] | None = Field(
        default=None,
        description=(
            "Restrict this skill to a subset of customer channels. "
            "``None`` (omitted in frontmatter) means all channels."
        ),
    )
    body: str = Field(min_length=1, description="The SOP markdown text.")
    source_path: str = ""
    """Original filesystem path; useful for log messages and tests."""

    def applies_to_channel(self, channel: str) -> bool:
        return self.channels is None or channel in self.channels
