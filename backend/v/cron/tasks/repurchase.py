"""Repurchase reminder.

Two-stage by design:

1. :func:`build_repurchase_targets` — periodic scan of ``dw.fact_order``
   for customers who bought repeatable items in a window (default 7-30
   days ago). Returns a list of target dicts that include the
   ``(channel, channel_user_id)`` joined from ``agent.user_profile``;
   customers without an agent identity are skipped (we have no way to
   reach them through the agent platform).

2. :func:`send_repurchase_reminders` — given a list of targets, call
   :func:`deliver_proactive` per target. Lets an upstream coordinator
   batch multiple categories or A/B test message variants without
   re-querying the warehouse.

A daily cron schedule binds these together by calling stage 1 then
stage 2 in :mod:`backend.app.cron_worker`.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from backend.v.cron.proactive import deliver_proactive
from backend.v.utils.logging import get_logger

_log = get_logger("cron.repurchase")
PURPOSE = "repurchase_reminder"

# Categories whose products are typically repurchased on a regular cadence.
DEFAULT_REPEATABLE_CATEGORIES = ("食品饮料", "休闲零食")
DEFAULT_WINDOW_DAYS_FROM = 7
DEFAULT_WINDOW_DAYS_TO = 30


async def build_repurchase_targets(
    pool: asyncpg.Pool,
    *,
    days_from: int = DEFAULT_WINDOW_DAYS_FROM,
    days_to: int = DEFAULT_WINDOW_DAYS_TO,
    categories: tuple[str, ...] = DEFAULT_REPEATABLE_CATEGORIES,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Scan ``dw.fact_order`` for repurchase candidates.

    The query joins the warehouse's ``customer_id`` against the agent's
    ``user_profile.profile`` JSON. The presence of ``customer_id`` in
    ``profile->>'dw_customer_id'`` is what bridges the two universes; rows
    without that mapping are not deliverable through the agent and are
    silently dropped.

    Args:
        pool: Postgres pool with codecs registered.
        days_from / days_to: Inclusive window of days-ago to consider.
        categories: Product categories worth re-pinging (consumables).
        limit: Cap on rows returned to bound a daily run.

    Returns:
        A list of dicts, each with ``channel``, ``channel_user_id``,
        ``product_name``, ``order_date``.
    """
    sql = """
        WITH eligible AS (
            SELECT
                o.customer_id,
                p.product_name,
                p.category,
                d.year, d.month, d.day,
                o.order_id,
                o.date_id
            FROM dw.fact_order o
            JOIN dw.dim_product p ON p.product_id = o.product_id
            JOIN dw.dim_date d ON d.date_id = o.date_id
            WHERE p.category = ANY($1::text[])
              AND make_date(d.year, d.month, d.day)
                  BETWEEN current_date - $3::int * interval '1 day'
                      AND current_date - $2::int * interval '1 day'
        )
        SELECT
            up.channel,
            up.channel_user_id,
            e.product_name,
            e.year, e.month, e.day
        FROM eligible e
        JOIN agent.user_profile up
          ON up.profile->>'dw_customer_id' = e.customer_id
        ORDER BY e.year DESC, e.month DESC, e.day DESC
        LIMIT $4
    """
    targets: list[dict[str, Any]] = []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql,
            list(categories),
            days_from,
            days_to,
            limit,
        )
    for row in rows:
        targets.append(
            {
                "channel": row["channel"],
                "channel_user_id": row["channel_user_id"],
                "product_name": row["product_name"],
                "order_date": f"{row['year']:04d}-{row['month']:02d}-{row['day']:02d}",
            }
        )
    _log.info("cron.repurchase.targets_built", count=len(targets))
    return targets


def _format(product_name: str, order_date: str) -> str:
    return (
        f"亲，您于 {order_date} 购买的「{product_name}」也快用完了吧？\n"
        "上次的口味/规格还合适吗？\n"
        "如需复购，回我「下单」我帮您安排，老顾客有专属价。"
    )


async def send_repurchase_reminders(
    ctx: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
) -> dict[str, int]:
    """Send a reminder to each target. Returns a ``{delivered, skipped}`` summary."""
    delivered = 0
    skipped = 0
    for t in targets:
        text = _format(t["product_name"], t["order_date"])
        ok = await deliver_proactive(
            channel=t["channel"],
            channel_user_id=t["channel_user_id"],
            text=text,
            purpose=PURPOSE,
            redis=ctx["redis"],
            sends=ctx["sends"],
        )
        if ok:
            delivered += 1
        else:
            skipped += 1
    _log.info(
        "cron.repurchase.run_complete",
        delivered=delivered,
        skipped=skipped,
    )
    return {"delivered": delivered, "skipped": skipped}
