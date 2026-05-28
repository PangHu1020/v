"""Internal message format produced by channel adapters and consumed by ``/v/agents``.

Channel-specific normalizers translate platform-shaped payloads into a
:class:`SystemMessage`. Everything downstream of the bus speaks this shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChannelSlug = Literal["wecom", "feishu", "wecom_aibot"]


class SystemMessage(BaseModel):
    """Normalized inbound message after channel adapter and debounce.

    Attributes:
        channel: Source channel slug (``wecom`` / ``feishu``).
        channel_user_id: Per-channel external user identifier. With ``channel``
            this forms the identity primary key in MVP.
        text: Plain-text content of the message after channel normalization.
        attachments: Optional list of structured attachment descriptors
            (image, file, voice). Each entry is platform-agnostic.
        received_at: UTC timestamp when the channel layer accepted the
            (post-debounce) message.
        dedup_key: Stable de-duplication key, typically derived from the
            platform's own message id. Allows the bus consumer to drop
            replays without doing semantic comparison.
        raw_payload_ref: Optional opaque reference to the original platform
            payload (e.g., an object-store key) for audit / debugging. Phase-1
            keeps this empty; Phase-2 may persist raw payloads for replay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: ChannelSlug
    channel_user_id: str = Field(min_length=1)
    text: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dedup_key: str = Field(min_length=1)
    raw_payload_ref: str | None = None
