"""Logistics delivery notification.

Triggered when an upstream logistics event indicates a parcel has been
signed for. Composes a brief Chinese reminder and pushes it through the
right channel adapter via :func:`deliver_proactive`.

Template-driven for Phase-2 P2 MVP. LLM personalization (greeting by
member tier, language preference, post-delivery cross-sell) is left for a
later phase that introduces a per-purpose prompt template registry.
"""

from __future__ import annotations

from typing import Any

from backend.v.cron.proactive import deliver_proactive

PURPOSE = "logistics_delivered"


def _format(order_id: str, tracking_number: str, courier: str | None) -> str:
    courier_text = f"承运商：{courier}\n" if courier else ""
    return (
        "您的订单已签收 ✅\n"
        f"订单号：{order_id}\n"
        f"运单号：{tracking_number}\n"
        f"{courier_text}"
        "如有问题随时找我，我帮您处理。"
    )


async def notify_logistics_delivered(
    ctx: dict[str, Any],
    *,
    channel: str,
    channel_user_id: str,
    order_id: str,
    tracking_number: str,
    courier: str | None = None,
) -> bool:
    """Deliver a logistics-signed-for notification to one customer."""
    text = _format(order_id, tracking_number, courier)
    return await deliver_proactive(
        channel=channel,
        channel_user_id=channel_user_id,
        text=text,
        purpose=PURPOSE,
        redis=ctx["redis"],
        sends=ctx["sends"],
    )
