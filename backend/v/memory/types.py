"""Pydantic models that describe memory layers.

The three runtime memory tiers and the LLM extraction output all share a
single sentence-shaped entry: :class:`MemoryEntry`. Differences between
tiers are about **storage and lifetime**, not shape.

- Working memory (Redis hash, TTL = working memory window) — list of
  :class:`MemoryEntry`. Lives only while the session is active.
- Event memory (Postgres ``agent.event_memory``, 30-day TTL,
  pgvector(1024)) — one row per :class:`MemoryEntry`. Read-time
  injection takes the most-recent N; the ``recall_memory`` tool fishes
  older entries via ANN over the embedding column.
- User profile (Postgres ``agent.user_profile``, JSONB, no TTL) —
  shaped by :class:`UserProfile`: a small canonical core plus open
  ``extras`` and ``notes`` for the long tail. Always-injected, full.

Conflict resolution between tiers happens **at read time**, not write
time. ``enter_node`` lays the layers down profile → events → working
in that order; the prompt instructs the model to treat later layers as
shadowing earlier ones when they disagree on the same topic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Sentence kinds. Three is enough — finer slicing didn't survive
# discussion. ``preference`` and ``observation`` typically live in the
# working tier (immediate session); ``event`` typically lives in the
# event tier. The extractor decides per entry.
MemoryKind = Literal["preference", "observation", "event"]


class MemoryEntry(BaseModel):
    """A single sentence-shaped memory record.

    Same shape across working + event tiers. The pgvector column on
    ``agent.event_memory`` is computed at write time from ``content`` —
    the LLM does not produce it.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Self-contained Chinese sentence.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    importance: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = trivia, 1.0 = critical (compliance, payment promise).",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Topical anchors for keyword pre-filtering and read-time "
            'shadowing detection (e.g., ["快递", "顺丰"]).'
        ),
    )
    kind: MemoryKind = Field(description="Which tier this entry semantically belongs to.")


# ── User profile ─────────────────────────────────────────────────────────────


class UserProfile(BaseModel):
    """Long-term structured picture of one customer.

    Canonical fields are explicit so renderer / tests / migrations can
    rely on them. ``extras`` is an open dict for whatever additional
    keys the LLM judges worth long-term retention but that don't fit a
    canonical slot — common in e-commerce where the long tail of useful
    attributes (favorite_brand, allergic_to, gift_recipient_name, …) is
    too wide to enumerate.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Identity / locale ────────────────────────────────────────────────
    customer_name: str | None = None
    preferred_salutation: str | None = Field(
        default=None,
        description='Honorific the agent should use, e.g., "先生" / "女士" / "老板".',
    )
    preferred_language: str | None = Field(
        default=None,
        description='Language tag, e.g., "zh", "en", "粤语".',
    )

    # ── Commerce ─────────────────────────────────────────────────────────
    member_level: str | None = Field(
        default=None,
        description='Membership tier, e.g., "普通" / "白银" / "黄金" / "黑金".',
    )

    # ── Communication style ──────────────────────────────────────────────
    response_style: str | None = Field(
        default=None,
        description='Preferred reply style, e.g., "简洁" / "详尽" / "数据导向".',
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description='Operational labels, e.g., ["易投诉", "高价值"].',
    )

    # ── Open extension ──────────────────────────────────────────────────
    extras: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Stable attributes that don't fit a canonical slot. The LLM is "
            "free to choose key names; rendering shows them verbatim."
        ),
    )
    notes: str | None = Field(
        default=None,
        max_length=200,
        description="Free-form supplementary text (≤200 chars).",
    )


# ── Extractor output ─────────────────────────────────────────────────────────


class ExtractionResult(BaseModel):
    """LLM-produced consolidation output.

    Three buckets, written to three different stores:

    - ``profile_updates``: partial :class:`UserProfile` fields the
      consolidator wants merged into ``agent.user_profile``. Empty dict
      means "nothing to upsert".
    - ``working_memories``: sentences the consolidator believes belong
      in this session's working memory (Redis). Mid-session compression
      uses these to keep context across the dropped tail.
    - ``event_memories``: sentences worth persisting beyond this
      session (Postgres event_memory + embedding). Survive 30 days.
    """

    model_config = ConfigDict(extra="forbid")

    profile_updates: dict[str, Any] = Field(default_factory=dict)
    working_memories: list[MemoryEntry] = Field(default_factory=list)
    event_memories: list[MemoryEntry] = Field(default_factory=list)
