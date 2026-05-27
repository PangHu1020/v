"""Ad-hoc ad push.

Forwards a marketing-team-prepared message to a single customer. Phase-2
P2 keeps copy generation outside the agent — the campaign text is
authored elsewhere (or prepared by an LLM in a separate offline pipeline)
and arrives ready to send. Avoids burning tokens on per-recipient
generation for the common case where one piece of copy goes to many
customers.

Future phases may add an LLM-personalized variant that uses
``user_profile`` to tailor the greeting / discount tier.
"""

from __future__ import annotations

from typing import Any

from backend.v.cron.proactive import deliver_proactive

PURPOSE = "ad_hoc_ad"


async def push_ad(
    ctx: dict[str, Any],
    *,
    channel: str,
    channel_user_id: str,
    text: str,
) -> bool:
    """Deliver a prepared ad text to one customer."""
    return await deliver_proactive(
        channel=channel,
        channel_user_id=channel_user_id,
        text=text,
        purpose=PURPOSE,
        redis=ctx["redis"],
        sends=ctx["sends"],
    )
