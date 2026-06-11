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
#
# NOTE (memory V2): this flat kind is the *legacy* shape. V2 splits memory
# into a semantic tier (:class:`UserMemory`, overwritable attributes) and an
# episodic tier (events). New extraction produces :class:`MemoryExtraction`
# (candidate lists); ``MemoryEntry`` + ``ExtractionResult`` are retained until
# the write pipeline (stage 3) migrates every consumer.
MemoryKind = Literal["preference", "observation", "event"]

# ── Memory V2 shared enums ────────────────────────────────────────────────────

# Semantic-tier attribute classes. ``preference`` = soft taste (顺丰/简洁回复);
# ``constraint`` = hard rule the agent must honour (过敏、不要电话联系);
# ``pattern`` = recurring behaviour (每月初下单、常退货).
UserMemoryKind = Literal["preference", "constraint", "pattern"]

# Provenance. ``stated`` = the customer said it outright; ``inferred`` = the
# model deduced it. Drives conflict arbitration (stated beats inferred) and
# whether the agent may repeat the fact back to the customer as established.
MemorySource = Literal["stated", "inferred"]


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


# ── Memory V2: semantic tier (user_memory) ────────────────────────────────────


class UserMemory(BaseModel):
    """One bi-temporal row of the semantic tier (``agent.user_memory``).

    A customer attribute with a single *current truth* per ``attr_key`` and an
    append-only supersession chain for audit. Overwriting closes the old row
    (``status='superseded'``, ``valid_to=now``, ``superseded_by``) and inserts a
    fresh active row — old rows are never deleted. No embedding: semantic memory
    is small, stable, and injected whole at session start.

    This mirrors the ``agent.user_memory`` schema (080_memory_v2.sql). It is the
    persisted/queried shape; the extractor emits :class:`MemoryCandidate`, which
    the write pipeline resolves into these rows.
    """

    model_config = ConfigDict(extra="forbid")

    attr_key: str = Field(
        min_length=1,
        description='Conflict key, e.g. "preferred_courier" / "allergy" / "member_level".',
    )
    attr_value: Any = Field(description="Scalar or list value. JSONB-encoded at the storage layer.")
    kind: UserMemoryKind = Field(description="preference | constraint | pattern.")
    source: MemorySource = Field(default="inferred")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Literal["active", "superseded"] = Field(default="active")
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_to: datetime | None = Field(
        default=None, description="NULL while current; set to now() when superseded."
    )
    last_confirmed_at: datetime | None = Field(
        default=None,
        description="When the customer last reaffirmed this value (staleness flag input).",
    )


# ── Memory V2: extraction candidates (pre-write intermediate) ─────────────────


class MemoryCandidate(BaseModel):
    """A single fact the extractor proposes, before the policy gate / merge.

    Tier-agnostic: ``content`` is the human-readable fact; the routing into the
    semantic vs episodic store is decided by which list it lands in on
    :class:`MemoryExtraction`. Carries provenance so the write pipeline can
    arbitrate conflicts (semantic) and weight recall (episodic).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Self-contained Chinese sentence.")
    importance: float = Field(ge=0.0, le=1.0)
    source: MemorySource = Field(default="inferred")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    # Semantic candidates set ``attr_key``/``attr_value`` so the pipeline can
    # key the upsert; episodic candidates leave them None and may set ``subject``.
    attr_key: str | None = Field(
        default=None, description="Semantic only: the conflict key to upsert under."
    )
    attr_value: Any = Field(default=None, description="Semantic only: the value for attr_key.")
    kind: UserMemoryKind | None = Field(
        default=None, description="Semantic only: preference | constraint | pattern."
    )
    subject: str | None = Field(
        default=None,
        description="Episodic only: entity/topic anchor (order id, SKU, topic) for clustering.",
    )


class MemoryExtraction(BaseModel):
    """The V2 extractor output: candidates split by tier.

    Replaces the role :class:`ExtractionResult` played at session-end. The write
    pipeline (stage 3) routes ``user_candidates`` into ``agent.user_memory``
    (bi-temporal upsert) and ``episodic_candidates`` into ``agent.event_memory``
    (append-only). ``ConversationState`` stays separate (mid-session continuity).
    """

    model_config = ConfigDict(extra="forbid")

    user_candidates: list[MemoryCandidate] = Field(default_factory=list)
    episodic_candidates: list[MemoryCandidate] = Field(default_factory=list)


# ── Conversation-state summary (mid-session compression) ──────────────────────


class ConversationState(BaseModel):
    """Structured snapshot of the *current dialogue* at a compression point.

    Distinct from memory (durable, cross-session facts about the customer):
    this is the **this-session continuity** artifact. When mid-session
    compression drops the older turns, this structured summary replaces them so
    the agent resumes seamlessly — it knows the topic, what it already did, what
    facts were established, and what is still open.

    Rendered into the ``<compressed_history>`` block by the compression node.
    All fields optional/empty-tolerant; an empty instance renders to nothing.
    """

    model_config = ConfigDict(extra="forbid")

    current_topic: str = Field(
        default="", description="What the conversation is about right now, one line."
    )
    events: list[str] = Field(
        default_factory=list,
        description="Key things that happened, in order (客户问了X、确认了Y).",
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="Actions the assistant already performed: tool calls, info, promises.",
    )
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="Open items / pending questions not yet answered.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Concrete facts surfaced: order numbers, amounts, SKUs, dates, tracking ids.",
    )

    def is_empty(self) -> bool:
        return not (
            self.current_topic
            or self.events
            or self.actions_taken
            or self.unresolved_questions
            or self.key_facts
        )


# ── Extractor output ─────────────────────────────────────────────────────────


class ExtractionResult(BaseModel):
    """LLM-produced consolidation output.

    Buckets, written to different stores / used for different purposes:

    - ``profile_updates``: partial :class:`UserProfile` fields the
      consolidator wants merged into ``agent.user_profile``. Empty dict
      means "nothing to upsert".
    - ``working_memories``: sentences the consolidator believes belong
      in this session's working memory (Redis). Mid-session compression
      uses these to keep context across the dropped tail.
    - ``event_memories``: sentences worth persisting beyond this
      session (Postgres event_memory + embedding). Survive 30 days.
    - ``conversation_state``: structured this-session continuity summary
      (mid-session compression only). ``None`` for session-end promotion.
    """

    model_config = ConfigDict(extra="forbid")

    profile_updates: dict[str, Any] = Field(default_factory=dict)
    working_memories: list[MemoryEntry] = Field(default_factory=list)
    event_memories: list[MemoryEntry] = Field(default_factory=list)
    conversation_state: ConversationState | None = Field(default=None)
