"""Proactive delivery helper.

Cron tasks call :func:`deliver_proactive` to (1) record that a system-
initiated message was sent (so the bus worker can prepend it to the
agent's view on the customer's next turn) and (2) hand the text to the
correct channel adapter for outbound.

The "record" step writes into a per-identity Redis list that
``on_session_start`` (Phase-2 P2 follow-up) drains into the LangGraph
``messages`` log so the AI sees its own outbound history. Phase-2 P2
ships only the writer; the on_session_start drainer is hooked up in a
follow-up commit so this commit can land cleanly without forcing a
checkpointer change.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import redis.asyncio as redis_async

from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("cron.proactive")

PROACTIVE_LOG_KEY_PREFIX = "proactive_log"
"""Redis list key: ``{prefix}:{channel}:{channel_user_id}`` holds proactive
messages awaiting injection into the next conversation turn."""

PROACTIVE_LOG_TTL_SECONDS = 30 * 24 * 60 * 60
"""Proactive messages stay in the log for 30 days. After that we assume
the customer has either seen them in the channel UI or has churned."""


def _log_key(channel: str, channel_user_id: str) -> str:
    return f"{PROACTIVE_LOG_KEY_PREFIX}:{channel}:{channel_user_id}"


SendFn = Callable[[str, str], Awaitable[None]]


async def record_proactive(
    redis: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    text: str,
    purpose: str,
) -> None:
    """Record a proactive message in the per-identity Redis log.

    Uses ``RPUSH`` so messages preserve emission order. The bus worker's
    on_session_start hook will ``LRANGE``/``DEL`` this list when the
    customer next replies, prepending the entries as ``AIMessage`` so
    the agent's context includes its own outbound history.
    """
    key = _log_key(channel, channel_user_id)
    payload = f"{purpose}\t{text}".encode()
    await redis.rpush(key, payload)
    await redis.expire(key, PROACTIVE_LOG_TTL_SECONDS)


async def deliver_proactive(
    *,
    channel: str,
    channel_user_id: str,
    text: str,
    purpose: str,
    redis: redis_async.Redis,
    sends: dict[str, SendFn],
) -> bool:
    """Persist + dispatch a proactive message.

    Returns:
        ``True`` if the message was dispatched, ``False`` if no channel
        adapter is registered for ``channel`` (callers can decide whether
        that's fatal or skip-and-warn).
    """
    with bind_request(channel=channel, channel_user_id=channel_user_id, purpose=purpose):
        send = sends.get(channel)
        if send is None:
            _log.error("cron.proactive.no_send", channel=channel)
            return False
        await record_proactive(
            redis,
            channel=channel,
            channel_user_id=channel_user_id,
            text=text,
            purpose=purpose,
        )
        await send(channel_user_id, text)
        _log.info("cron.proactive.delivered", text_len=len(text))
        return True
